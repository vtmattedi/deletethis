#include <Arduino.h>
#include <driver/i2s.h>
#include <arduinoFFT.h>
#include <WiFi.h>
#include <math.h>

#if __has_include("creds.h")
#include "creds.h"
#else
#error "Copy include/creds.example.h to include/creds.h and fill in WIFI_SSID / WIFI_PASSWORD."
#endif

#ifndef AUDIO_TCP_PORT
#define AUDIO_TCP_PORT 3333
#endif

// ============================================================
// ESP32 DevKit V1 + INMP441
//
// INMP441 VDD -> 3V3
// INMP441 GND -> GND
// INMP441 SCK -> GPIO26
// INMP441 WS  -> GPIO25
// INMP441 SD  -> GPIO33
// INMP441 L/R -> GND
// ============================================================

#define I2S_PORT I2S_NUM_0

#define I2S_BCLK  26
#define I2S_LRCLK 25
#define I2S_DIN   33

// ============================================================
// Serial
//
// CAPTURE mode streams 16000 samples/s * 4 bytes = 64000 B/s.
// At 8N1 that needs at least 640000 baud, so 115200 is not
// enough. 921600 leaves roughly 30% headroom and is supported
// by the usual CP2102 / CH340 adapters.
// ============================================================

static constexpr uint32_t SERIAL_BAUD = 921600;

// ============================================================
// Audio / FFT configuration
// ============================================================

static constexpr uint32_t SAMPLE_RATE = 16000;
static constexpr uint16_t FFT_SAMPLES = 1024;

static_assert(
    (FFT_SAMPLES & (FFT_SAMPLES - 1)) == 0,
    "FFT_SAMPLES must be a power of 2"
);

static constexpr double FULL_SCALE_24BIT = 8388608.0;

// Ignore the first few captures after I2S startup.
static constexpr uint8_t STARTUP_FRAMES_TO_IGNORE = 5;

// I2S driver event queue, used to see DMA overruns.
static constexpr int I2S_EVENT_QUEUE_LENGTH = 8;

static QueueHandle_t i2sEvents = nullptr;

// Total DMA overruns since boot. Each one is audio that was
// captured and then overwritten before it could be sent.
static uint32_t i2sOverruns = 0;

static uint32_t lastOverrunReport = 0;

static constexpr uint32_t OVERRUN_REPORT_INTERVAL_MS = 1000;

// ============================================================
// Streaming protocol
//
// Stream header, once, when capture starts.
// Then repeating: frame header + frame payload.
//
// All multi-byte fields are little-endian, which is the
// native ESP32 layout, so the structs are written verbatim.
// ============================================================

static constexpr uint16_t PROTOCOL_VERSION = 1;

// Sample format identifiers.
// 1 = signed 32-bit little-endian, 24 significant bits,
//     already right-aligned (the raw I2S word >> 8).
static constexpr uint16_t FORMAT_PCM_S32LE = 1;

static constexpr uint16_t BITS_VALID = 24;

// 512 samples = 32 ms per frame at 16 kHz.
static constexpr uint16_t FRAME_SAMPLES = 512;

struct __attribute__((packed)) StreamHeader
{
    char     magic[4];       // "ACD1"
    uint16_t version;
    uint16_t headerSize;
    uint32_t sampleRate;
    uint16_t format;
    uint16_t channels;
    uint16_t bitsValid;
    uint16_t frameSamples;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
};

static_assert(
    sizeof(StreamHeader) == 32,
    "StreamHeader must be 32 bytes"
);

struct __attribute__((packed)) FrameHeader
{
    char     magic[4];       // "ACDF"
    uint32_t sequence;
    uint16_t samples;
    uint16_t dropped;        // frames lost since the last frame
};

static_assert(
    sizeof(FrameHeader) == 12,
    "FrameHeader must be 12 bytes"
);

// ============================================================
// Operating mode
// ============================================================

// ============================================================
// Transport roles
//
//   Serial = commands, status and logs. Always human-readable.
//   TCP    = binary audio. Always machine-readable.
//
// Keeping them separate means nothing has to arbitrate who owns
// Serial: a log line can never land in the middle of a PCM
// frame, because PCM never goes there. That invariant is worth
// more than being able to stream over either pipe.
//
// The serial streaming path below is still compiled, as a
// reference and a fallback if the network is unavailable. Set
// this to 1 to make the `c` command reach it again; the PC
// tools will then work over a serial port as well.
// ============================================================

#define ALLOW_SERIAL_CAPTURE 0

enum class Mode
{
    TEXT,
    CAPTURE,   // unreachable unless ALLOW_SERIAL_CAPTURE
    IDLE
};

static Mode mode = Mode::TEXT;

// ============================================================
// Network
//
// One listening socket, one client, no discovery and no
// application-level reconnect: the client connects, gets a
// stream header and then frames, and if the write fails the
// socket is closed and we listen again.
//
// A TCP client takes over binary streaming from Serial, so the
// 64 kB/s stream is never duplicated. Serial stays a text
// channel while that is happening.
// ============================================================

static WiFiServer audioServer(AUDIO_TCP_PORT);
static WiFiClient audioClient;

static bool serverStarted = false;
static bool wasConnected = false;

// How long a write may make no progress before the client is
// treated as dead. Dropping audio beats stalling I2S forever.
static constexpr uint32_t TCP_WRITE_TIMEOUT_MS = 1000;

// How long to wait for the access point at boot.
static constexpr uint32_t WIFI_CONNECT_TIMEOUT_MS = 20000;

// ============================================================
// Stream sink
//
// The frame emission below does not care whether it is writing
// to Serial or a socket. Both HardwareSerial and WiFiClient
// derive from Print, so this stays a tag and a pointer rather
// than a class hierarchy.
// ============================================================

struct StreamSink
{
    Print *out = nullptr;

    // Null for Serial. Set for TCP, where a write can fail
    // part-way and the peer can vanish.
    WiFiClient *client = nullptr;
};

StreamSink serialSink()
{
    StreamSink sink;
    sink.out = &Serial;

    return sink;
}

StreamSink clientSink(WiFiClient &client)
{
    StreamSink sink;
    sink.out = &client;
    sink.client = &client;

    return sink;
}

bool sinkConnected(const StreamSink &sink)
{
    if (sink.out == nullptr)
        return false;

    if (sink.client == nullptr)
        return true;

    return sink.client->connected();
}

// Writes everything or reports failure. One call to write() is
// never assumed to have sent the whole buffer: over TCP a short
// write is normal, and silently dropping the remainder would
// corrupt the stream in a way the PC could only see as a lost
// frame much later.
bool writeAll(
    StreamSink &sink,
    const void *data,
    size_t size
)
{
    const uint8_t *bytes =
        static_cast<const uint8_t *>(data);

    uint32_t lastProgress = millis();

    while (size > 0)
    {
        if (!sinkConnected(sink))
            return false;

        const size_t written =
            sink.out->write(bytes, size);

        if (written > 0)
        {
            bytes += written;
            size -= written;

            lastProgress = millis();

            continue;
        }

        // No progress. A peer that has stopped reading must not
        // be allowed to block acquisition indefinitely.
        if (
            millis() - lastProgress >
            TCP_WRITE_TIMEOUT_MS
        )
        {
            return false;
        }

        delay(1);
    }

    return true;
}

// ============================================================
// Buffers
// ============================================================

double vReal[FFT_SAMPLES];
double vImag[FFT_SAMPLES];

int32_t i2sSamples[FFT_SAMPLES];

// The same buffer serves CAPTURE mode, which needs only
// FRAME_SAMPLES of it.
static_assert(
    FRAME_SAMPLES <= FFT_SAMPLES,
    "FRAME_SAMPLES must fit in the I2S buffer"
);

ArduinoFFT<double> FFT(
    vReal,
    vImag,
    FFT_SAMPLES,
    SAMPLE_RATE
);

// ============================================================
// Frequency band structure
// ============================================================

struct FrequencyBand
{
    const char *name;

    float lowHz;
    float highHz;

    double energy;
    double peakMagnitude;
    double peakFrequency;
};

FrequencyBand bands[] =
{
    {"30-80",    30.0f,   80.0f,   0, 0, 0},
    {"80-200",   80.0f,   200.0f,  0, 0, 0},
    {"200-500",  200.0f,  500.0f,  0, 0, 0},
    {"500-1k",   500.0f,  1000.0f, 0, 0, 0},
    {"1k-2k",    1000.0f, 2000.0f, 0, 0, 0},
    {"2k-4k",    2000.0f, 4000.0f, 0, 0, 0},
};

static constexpr size_t BAND_COUNT =
    sizeof(bands) / sizeof(bands[0]);

// ============================================================
// Helper functions
// ============================================================

// First bin whose centre frequency is at or above `frequency`.
//
// Band edges are half-open, [low, high), so a bin belongs to exactly
// one band. Rounding to the nearest bin at both edges instead would
// put the bin nearest a shared edge into both neighbouring bands and
// count its energy twice.
//
// The returned value may be FFT_SAMPLES / 2, one past the last usable
// bin, so that it also serves as an exclusive upper limit.
uint16_t binAtOrAbove(float frequency)
{
    if (frequency <= 0.0f)
        return 0;

    const double exact =
        static_cast<double>(frequency) *
        FFT_SAMPLES /
        SAMPLE_RATE;

    const double bin = ceil(exact);

    const uint16_t limit = FFT_SAMPLES / 2;

    if (bin >= limit)
        return limit;

    return static_cast<uint16_t>(bin);
}

double binToFrequency(uint16_t bin)
{
    return
        static_cast<double>(bin) *
        SAMPLE_RATE /
        FFT_SAMPLES;
}

void resetBands()
{
    for (size_t i = 0; i < BAND_COUNT; i++)
    {
        bands[i].energy = 0;
        bands[i].peakMagnitude = 0;
        bands[i].peakFrequency = 0;
    }
}

// ============================================================
// I2S
// ============================================================

void initI2S()
{
    i2s_config_t config = {};

    config.mode = static_cast<i2s_mode_t>(
        I2S_MODE_MASTER |
        I2S_MODE_RX
    );

    config.sample_rate = SAMPLE_RATE;

    config.bits_per_sample =
        I2S_BITS_PER_SAMPLE_32BIT;

    // L/R = GND on INMP441
    config.channel_format =
        I2S_CHANNEL_FMT_ONLY_LEFT;

#if ESP_IDF_VERSION_MAJOR >= 5
    config.communication_format =
        I2S_COMM_FORMAT_STAND_I2S;
#else
    config.communication_format =
        I2S_COMM_FORMAT_I2S;
#endif

    config.intr_alloc_flags =
        ESP_INTR_FLAG_LEVEL1;

    config.dma_buf_count = 8;
    config.dma_buf_len = 256;

    config.use_apll = false;
    config.tx_desc_auto_clear = false;
    config.fixed_mclk = 0;

    // With portMAX_DELAY an i2s_read() never returns short, so a
    // stalled writer does not show up as a failed read: the DMA ring
    // silently overwrites itself and the audio is gone. The event
    // queue is the only place that loss is visible, and over TCP it
    // is the loss that matters.
    esp_err_t err = i2s_driver_install(
        I2S_PORT,
        &config,
        I2S_EVENT_QUEUE_LENGTH,
        &i2sEvents
    );

    if (err != ESP_OK)
    {
        Serial.printf(
            "ERROR: i2s_driver_install(): %d\n",
            err
        );

        while (true)
            delay(1000);
    }

    i2s_pin_config_t pins = {};

    pins.bck_io_num = I2S_BCLK;
    pins.ws_io_num = I2S_LRCLK;
    pins.data_out_num = I2S_PIN_NO_CHANGE;
    pins.data_in_num = I2S_DIN;

    err = i2s_set_pin(
        I2S_PORT,
        &pins
    );

    if (err != ESP_OK)
    {
        Serial.printf(
            "ERROR: i2s_set_pin(): %d\n",
            err
        );

        while (true)
            delay(1000);
    }

    i2s_zero_dma_buffer(I2S_PORT);

    Serial.println("I2S initialized.");
}

// ============================================================
// Capture
// ============================================================

// Reads up to `wanted` samples into i2sSamples and returns
// how many samples were actually read.
size_t readSamples(size_t wanted)
{
    size_t bytesRead = 0;

    esp_err_t err = i2s_read(
        I2S_PORT,
        i2sSamples,
        wanted * sizeof(int32_t),
        &bytesRead,
        portMAX_DELAY
    );

    if (err != ESP_OK)
        return 0;

    return bytesRead / sizeof(int32_t);
}

bool captureAudio()
{
    const size_t sampleCount =
        readSamples(FFT_SAMPLES);

    if (sampleCount == 0)
    {
        Serial.println("I2S read error.");
        return false;
    }

    if (sampleCount != FFT_SAMPLES)
    {
        Serial.printf(
            "Short read: %u / %u samples\n",
            static_cast<unsigned>(sampleCount),
            FFT_SAMPLES
        );

        return false;
    }

    return true;
}

// ============================================================
// Audio preprocessing
// ============================================================

void preprocessAudio(
    double &rms,
    double &peak,
    double &dbFS,
    double &dcOffset
)
{
    double sum = 0.0;

    // --------------------------------------------------------
    // INMP441 sends 24-bit audio in a 32-bit I2S word.
    //
    // The useful signed data is normally in the upper
    // 24 bits, hence >> 8.
    // --------------------------------------------------------

    for (uint16_t i = 0; i < FFT_SAMPLES; i++)
    {
        int32_t sample = i2sSamples[i] >> 8;

        vReal[i] =
            static_cast<double>(sample);

        vImag[i] = 0.0;

        sum += vReal[i];
    }

    dcOffset =
        sum / FFT_SAMPLES;

    double squareSum = 0.0;
    peak = 0.0;

    // Remove DC before FFT.
    for (uint16_t i = 0; i < FFT_SAMPLES; i++)
    {
        const double sample =
            vReal[i] - dcOffset;

        vReal[i] = sample;

        const double absSample =
            fabs(sample);

        if (absSample > peak)
            peak = absSample;

        squareSum += sample * sample;
    }

    rms =
        sqrt(
            squareSum /
            FFT_SAMPLES
        );

    if (rms > 0.0)
    {
        dbFS =
            20.0 *
            log10(
                rms /
                FULL_SCALE_24BIT
            );
    }
    else
    {
        dbFS = -120.0;
    }
}

// ============================================================
// FFT analysis
// ============================================================

void analyzeFFT(
    double &mainPeakFrequency,
    double &mainPeakMagnitude
)
{
    FFT.windowing(
        FFTWindow::Hamming,
        FFTDirection::Forward
    );

    FFT.compute(
        FFTDirection::Forward
    );

    FFT.complexToMagnitude();

    resetBands();

    // --------------------------------------------------------
    // Analyze predefined frequency bands.
    // --------------------------------------------------------

    for (size_t b = 0; b < BAND_COUNT; b++)
    {
        FrequencyBand &band = bands[b];

        uint16_t firstBin =
            binAtOrAbove(
                band.lowHz
            );

        // Exclusive, so the bin on the boundary belongs to the
        // band above and is not counted twice.
        const uint16_t endBin =
            binAtOrAbove(
                band.highHz
            );

        // Skip DC, which carries the microphone's offset.
        if (firstBin < 1)
            firstBin = 1;

        for (
            uint16_t bin = firstBin;
            bin < endBin;
            bin++
        )
        {
            const double magnitude =
                vReal[bin];

            // Magnitude-squared is a better approximation
            // of spectral energy than summing magnitudes.
            band.energy +=
                magnitude *
                magnitude;

            if (
                magnitude >
                band.peakMagnitude
            )
            {
                band.peakMagnitude =
                    magnitude;

                band.peakFrequency =
                    binToFrequency(bin);
            }
        }
    }

    // --------------------------------------------------------
    // Global useful peak
    //
    // Ignore everything below 100 Hz so mains / movement
    // does not permanently win.
    // --------------------------------------------------------

    const uint16_t firstBin =
        binAtOrAbove(100.0f);

    // Same half-open convention as the bands, so the search covers
    // exactly the union of 80-200 .. 2k-4k above 100 Hz.
    const uint16_t endBin =
        binAtOrAbove(4000.0f);

    mainPeakMagnitude = 0.0;
    mainPeakFrequency = 0.0;

    for (
        uint16_t bin = firstBin;
        bin < endBin;
        bin++
    )
    {
        const double magnitude =
            vReal[bin];

        if (
            magnitude >
            mainPeakMagnitude
        )
        {
            mainPeakMagnitude =
                magnitude;

            mainPeakFrequency =
                binToFrequency(bin);
        }
    }
}

// ============================================================
// Output
// ============================================================

void printResults(
    double rms,
    double peak,
    double dbFS,
    double dcOffset,
    double mainPeakFrequency,
    double mainPeakMagnitude
)
{
    Serial.println();
    Serial.println(
        "------------------------------------------------------------"
    );

    Serial.printf(
        "RMS=%10.0f  "
        "Peak=%10.0f  "
        "dBFS=%7.2f  "
        "DC=%10.0f\n",
        rms,
        peak,
        dbFS,
        dcOffset
    );

    Serial.printf(
        "Main >100Hz: %7.1f Hz  "
        "FFT=%12.0f\n",
        mainPeakFrequency,
        mainPeakMagnitude
    );

    Serial.println();
    Serial.println(
        "Band         Energy              Peak Hz      Peak FFT"
    );

    for (size_t i = 0; i < BAND_COUNT; i++)
    {
        const FrequencyBand &band =
            bands[i];

        Serial.printf(
            "%-10s %16.3e   %8.1f   %12.0f\n",
            band.name,
            band.energy,
            band.peakFrequency,
            band.peakMagnitude
        );
    }
}

// ============================================================
// Processing
// ============================================================

void processAudio()
{
    double rms = 0;
    double peak = 0;
    double dbFS = 0;
    double dcOffset = 0;

    double mainPeakFrequency = 0;
    double mainPeakMagnitude = 0;

    preprocessAudio(
        rms,
        peak,
        dbFS,
        dcOffset
    );

    analyzeFFT(
        mainPeakFrequency,
        mainPeakMagnitude
    );

    printResults(
        rms,
        peak,
        dbFS,
        dcOffset,
        mainPeakFrequency,
        mainPeakMagnitude
    );
}

// ============================================================
// CAPTURE mode
// ============================================================

static uint32_t frameSequence = 0;
static uint16_t pendingDropped = 0;

bool sendStreamHeader(StreamSink &sink)
{
    StreamHeader header = {};

    header.magic[0] = 'A';
    header.magic[1] = 'C';
    header.magic[2] = 'D';
    header.magic[3] = '1';

    header.version = PROTOCOL_VERSION;
    header.headerSize = sizeof(StreamHeader);
    header.sampleRate = SAMPLE_RATE;
    header.format = FORMAT_PCM_S32LE;
    header.channels = 1;
    header.bitsValid = BITS_VALID;
    header.frameSamples = FRAME_SAMPLES;

    if (!writeAll(sink, &header, sizeof(header)))
        return false;

    if (sink.client == nullptr)
        Serial.flush();

    return true;
}

bool sendFrame(
    StreamSink &sink,
    const int32_t *samples,
    uint16_t count
)
{
    FrameHeader header = {};

    header.magic[0] = 'A';
    header.magic[1] = 'C';
    header.magic[2] = 'D';
    header.magic[3] = 'F';

    header.sequence = frameSequence++;
    header.samples = count;
    header.dropped = pendingDropped;

    pendingDropped = 0;

    if (!writeAll(sink, &header, sizeof(header)))
        return false;

    return writeAll(
        sink,
        samples,
        static_cast<size_t>(count) * sizeof(int32_t)
    );
}

// Counts DMA overruns into pendingDropped, so the PC sees them
// in the next frame header exactly as it sees a short read.
//
// This is the measurement to watch when streaming over WiFi: if
// a write stalls for longer than the DMA ring holds -- 8 buffers
// of 256 samples, about 128 ms -- audio is lost, and without
// this the sequence numbers would still be contiguous and the
// gap would be invisible.
void drainI2sEvents()
{
    if (i2sEvents == nullptr)
        return;

    i2s_event_t event;

    uint32_t seen = 0;

    while (xQueueReceive(i2sEvents, &event, 0) == pdTRUE)
    {
        if (event.type != I2S_EVENT_RX_Q_OVF)
            continue;

        if (pendingDropped < 0xFFFF)
            pendingDropped++;

        i2sOverruns++;
        seen++;
    }

    if (seen == 0)
        return;

    // Safe to say so on Serial, because Serial never carries
    // audio. Throttled, since an overrun tends to arrive with
    // friends and the log is meant to be readable.
    if (mode == Mode::CAPTURE)
        return;

    const uint32_t now = millis();

    if (now - lastOverrunReport < OVERRUN_REPORT_INTERVAL_MS)
        return;

    lastOverrunReport = now;

    Serial.printf(
        "# i2s: %u overrun(s), %u total -- audio was lost\n",
        seen,
        i2sOverruns
    );
}

// Reads one frame into i2sSamples, right-aligned and ready to
// send. A frame that could not be read whole is counted and
// reported in the next good frame header.
bool readPcmFrame()
{
    drainI2sEvents();

    const size_t sampleCount =
        readSamples(FRAME_SAMPLES);

    if (sampleCount != FRAME_SAMPLES)
    {
        if (pendingDropped < 0xFFFF)
            pendingDropped++;

        return false;
    }

    // Right-align the 24-bit sample in the 32-bit word so the
    // PC side does not have to know about the I2S padding.
    for (uint16_t i = 0; i < FRAME_SAMPLES; i++)
        i2sSamples[i] = i2sSamples[i] >> 8;

    return true;
}

// ============================================================
// WiFi and the audio server
// ============================================================

void setupWifi()
{
    WiFi.persistent(false);
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);          // latency over power here
    WiFi.setAutoReconnect(true);

    Serial.printf("WiFi: connecting to \"%s\"", WIFI_SSID);

    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    const uint32_t started = millis();

    while (
        WiFi.status() != WL_CONNECTED &&
        millis() - started < WIFI_CONNECT_TIMEOUT_MS
    )
    {
        delay(250);
        Serial.print(".");
    }

    Serial.println();

    if (WiFi.status() == WL_CONNECTED)
    {
        Serial.print("WiFi: connected, IP ");
        Serial.println(WiFi.localIP());
        Serial.printf(
            "Audio server: tcp://%s:%u\n",
            WiFi.localIP().toString().c_str(),
            static_cast<unsigned>(AUDIO_TCP_PORT)
        );
    }
    else
    {
        // Not fatal. The Serial path still works, and the
        // station keeps retrying in the background.
        Serial.println(
            "WiFi: not connected. Serial capture still works; "
            "the server starts if the link comes up later."
        );
    }
}

void startAudioServer()
{
    if (serverStarted)
        return;

    audioServer.begin();
    audioServer.setNoDelay(true);

    serverStarted = true;
}

void dropClient(const char *reason)
{
    if (!audioClient)
        return;

    audioClient.stop();

    Serial.printf("# tcp: client gone (%s)\n", reason);
}

// Accepts a client, starts the server once the link is up, and
// notices a peer that has disappeared.
void handleTcpClient()
{
    const bool connected =
        WiFi.status() == WL_CONNECTED;

    if (connected != wasConnected)
    {
        wasConnected = connected;

        if (connected)
        {
            Serial.print("# wifi: up, IP ");
            Serial.println(WiFi.localIP());
        }
        else
        {
            Serial.println("# wifi: down");

            dropClient("wifi down");
        }
    }

    if (!connected)
        return;

    startAudioServer();

    if (audioClient && !audioClient.connected())
        dropClient("disconnected");

    if (audioClient)
    {
        // One client for v1. Anyone else is turned away at once
        // rather than left hanging in the backlog.
        WiFiClient extra = audioServer.accept();

        if (extra)
        {
            extra.println("busy");
            extra.stop();
        }

        return;
    }

    WiFiClient incoming = audioServer.accept();

    if (!incoming)
        return;

    audioClient = incoming;
    audioClient.setNoDelay(true);

    Serial.print("# tcp: client ");
    Serial.print(audioClient.remoteIP());
    Serial.println(" connected, streaming");

    // A new connection is a new stream session, so the sequence
    // restarts and the fresh ACD1 header says so.
    frameSequence = 0;
    pendingDropped = 0;

    i2s_zero_dma_buffer(I2S_PORT);

    StreamSink sink = clientSink(audioClient);

    if (!sendStreamHeader(sink))
        dropClient("header write failed");
}

bool tcpClientStreaming()
{
    return audioClient && audioClient.connected();
}

// ============================================================
// Commands
//
// Single characters, so the PC side stays trivial:
//
//   t  TEXT mode     human-readable RMS / FFT diagnostics
//   c  CAPTURE mode  binary PCM stream
//   s  IDLE          stop everything
//   ?  status line   (ignored while capturing)
// ============================================================

void enterMode(Mode next)
{
    if (next == mode)
        return;

    if (mode == Mode::CAPTURE)
    {
        // Let the host see a clean end of stream before any
        // text is mixed in.
        Serial.flush();
    }

    mode = next;

    switch (mode)
    {
        case Mode::TEXT:
            Serial.println();
            Serial.println("# mode=TEXT");
            break;

        case Mode::CAPTURE:
            if (tcpClientStreaming())
            {
                // Only one binary destination at a time, so the
                // 64 kB/s stream is never sent twice.
                mode = Mode::TEXT;

                Serial.println();
                Serial.println(
                    "# a TCP client is streaming; "
                    "disconnect it first"
                );

                break;
            }

            frameSequence = 0;
            pendingDropped = 0;

            i2s_zero_dma_buffer(I2S_PORT);

            {
                StreamSink sink = serialSink();
                sendStreamHeader(sink);
            }
            break;

        case Mode::IDLE:
            Serial.println();
            Serial.println("# mode=IDLE");
            break;
    }
}

void printStatus()
{
    Serial.printf(
        "# sample_rate=%u fft=%u frame=%u baud=%u\n",
        SAMPLE_RATE,
        FFT_SAMPLES,
        FRAME_SAMPLES,
        SERIAL_BAUD
    );

    if (WiFi.status() == WL_CONNECTED)
    {
        Serial.printf(
            "# wifi=up ip=%s port=%u client=%s\n",
            WiFi.localIP().toString().c_str(),
            static_cast<unsigned>(AUDIO_TCP_PORT),
            tcpClientStreaming() ? "yes" : "no"
        );
    }
    else
    {
        Serial.println("# wifi=down");
    }
}

void handleCommands()
{
    while (Serial.available() > 0)
    {
        const int c = Serial.read();

        switch (c)
        {
            case 't':
            case 'T':
                enterMode(Mode::TEXT);
                break;

            case 'c':
            case 'C':
#if ALLOW_SERIAL_CAPTURE
                enterMode(Mode::CAPTURE);
#else
                Serial.println();
                Serial.println(
                    "# capture over serial disabled; use TCP"
                );

                if (WiFi.status() == WL_CONNECTED)
                {
                    Serial.printf(
                        "# connect to tcp://%s:%u\n",
                        WiFi.localIP().toString().c_str(),
                        static_cast<unsigned>(AUDIO_TCP_PORT)
                    );
                }
                else
                {
                    Serial.println(
                        "# wifi is down, so there is nowhere to "
                        "connect yet"
                    );
                }
#endif
                break;

            case 's':
            case 'S':
                enterMode(Mode::IDLE);
                break;

            case '?':
                if (mode != Mode::CAPTURE)
                    printStatus();
                break;

            default:
                // Ignore whitespace and anything unknown so a
                // serial monitor sending CRLF does not matter.
                break;
        }
    }
}

// ============================================================
// Arduino
// ============================================================

void setup()
{
    Serial.begin(SERIAL_BAUD);

    delay(1000);

    Serial.println();
    Serial.println(
        "ESP32 + INMP441 multi-band FFT test"
    );

    Serial.printf(
        "Sample rate: %u Hz\n",
        SAMPLE_RATE
    );

    Serial.printf(
        "FFT samples: %u\n",
        FFT_SAMPLES
    );

    Serial.printf(
        "FFT resolution: %.3f Hz/bin\n",
        static_cast<double>(SAMPLE_RATE) /
            FFT_SAMPLES
    );

    Serial.printf(
        "Capture duration: %.1f ms\n",
        (
            static_cast<double>(FFT_SAMPLES) /
            SAMPLE_RATE
        ) *
            1000.0
    );

    Serial.printf(
        "Nyquist frequency: %.0f Hz\n",
        SAMPLE_RATE / 2.0
    );

    initI2S();

    Serial.printf(
        "Ignoring first %u startup frames...\n",
        STARTUP_FRAMES_TO_IGNORE
    );

    for (
        uint8_t i = 0;
        i < STARTUP_FRAMES_TO_IGNORE;
        i++
    )
    {
        captureAudio();
    }

    setupWifi();
    startAudioServer();

#if ALLOW_SERIAL_CAPTURE
    Serial.println(
        "Commands: t=text  c=capture  s=stop  ?=status"
    );
#else
    Serial.println(
        "Commands: t=text  s=stop  ?=status"
    );

    Serial.println(
        "Serial is text only. Audio streams over TCP; "
        "connect a client to start it."
    );
#endif

    Serial.println("Starting analysis.");
}

void loop()
{
    handleCommands();
    handleTcpClient();

    // A TCP client outranks everything: it gets the PCM, and
    // Serial stays a text channel. The periodic FFT dump is
    // paused while that happens, because it needs 1024-sample
    // reads and would fight the 512-sample frame cadence for
    // the same I2S buffer.
    if (tcpClientStreaming())
    {
        if (!readPcmFrame())
            return;

        StreamSink sink = clientSink(audioClient);

        if (!sendFrame(sink, i2sSamples, FRAME_SAMPLES))
            dropClient("write failed");

        return;
    }

    switch (mode)
    {
        case Mode::TEXT:
            if (captureAudio())
                processAudio();
            break;

        case Mode::CAPTURE:
            if (readPcmFrame())
            {
                StreamSink sink = serialSink();
                sendFrame(sink, i2sSamples, FRAME_SAMPLES);
            }
            break;

        case Mode::IDLE:
            // Keep the DMA buffers drained so re-entering
            // CAPTURE does not start on stale audio.
            readSamples(FRAME_SAMPLES);
            break;
    }
}
