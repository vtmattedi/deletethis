#include <Arduino.h>
#include <driver/i2s.h>
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
// Audio
//
// The device captures and forwards. It does not analyse: every
// FFT, band and threshold lives in the PC tools, where a rule
// can be changed without a reflash. Serial is for status and
// connection state only.
// ============================================================

static constexpr uint32_t SAMPLE_RATE = 16000;

// Ignore the first few frames after I2S startup.
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

// Only meaningful when ALLOW_SERIAL_CAPTURE is 1.
static bool serialCapture = false;

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

// One frame of audio, right-aligned by readPcmFrame().
int32_t i2sSamples[FRAME_SAMPLES];

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
#if ALLOW_SERIAL_CAPTURE
    if (serialCapture)
        return;
#endif

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

#if ALLOW_SERIAL_CAPTURE
void startSerialCapture()
{
    if (serialCapture)
        return;

    if (tcpClientStreaming())
    {
        // Only one binary destination at a time, so the 64 kB/s
        // stream is never sent twice.
        Serial.println();
        Serial.println(
            "# a TCP client is streaming; disconnect it first"
        );

        return;
    }

    frameSequence = 0;
    pendingDropped = 0;

    i2s_zero_dma_buffer(I2S_PORT);

    StreamSink sink = serialSink();

    if (sendStreamHeader(sink))
        serialCapture = true;
}

void stopSerialCapture()
{
    if (!serialCapture)
        return;

    Serial.flush();
    serialCapture = false;

    Serial.println();
    Serial.println("# serial capture stopped");
}
#endif

void printStatus()
{
    Serial.printf(
        "# sample_rate=%u frame=%u baud=%u uptime=%lus\n",
        SAMPLE_RATE,
        FRAME_SAMPLES,
        SERIAL_BAUD,
        millis() / 1000UL
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
            case 'c':
            case 'C':
#if ALLOW_SERIAL_CAPTURE
                startSerialCapture();
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
#if ALLOW_SERIAL_CAPTURE
                stopSerialCapture();
#endif
                break;

            case '?':
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
    Serial.println("ESP32 + INMP441 audio node");

    Serial.printf(
        "Sample rate: %u Hz\n",
        SAMPLE_RATE
    );

    Serial.printf(
        "Frame: %u samples (%.1f ms)\n",
        FRAME_SAMPLES,
        (
            static_cast<double>(FRAME_SAMPLES) /
            SAMPLE_RATE
        ) * 1000.0
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
        readSamples(FRAME_SAMPLES);
    }

    setupWifi();
    startAudioServer();

#if ALLOW_SERIAL_CAPTURE
    Serial.println("Commands: c=capture  s=stop  ?=status");
#else
    Serial.println("Commands: ?=status");
#endif

    Serial.println(
        "Serial reports status and connection state only. "
        "Audio streams over TCP; all analysis is on the PC."
    );

    Serial.println("Ready.");
}

void loop()
{
    handleCommands();
    handleTcpClient();

    if (tcpClientStreaming())
    {
        if (!readPcmFrame())
            return;

        StreamSink sink = clientSink(audioClient);

        if (!sendFrame(sink, i2sSamples, FRAME_SAMPLES))
            dropClient("write failed");

        return;
    }

#if ALLOW_SERIAL_CAPTURE
    if (serialCapture)
    {
        if (readPcmFrame())
        {
            StreamSink sink = serialSink();

            if (!sendFrame(sink, i2sSamples, FRAME_SAMPLES))
                stopSerialCapture();
        }

        return;
    }
#endif

    // Nobody is listening. Keep reading anyway so the DMA ring
    // does not overflow and so a client that connects starts on
    // fresh audio rather than whatever was left in the buffers.
    drainI2sEvents();
    readSamples(FRAME_SAMPLES);
}
