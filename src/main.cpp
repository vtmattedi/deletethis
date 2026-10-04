#include <Arduino.h>
#include <NightMare.h>
#include <esp_heap_caps.h>

#include "audio/SelfTest.h"
#include "audio/WatsonAcoustic.h"

// ============================================================
// Watson: an acoustic observation device.
//
// ESP32 DevKit V1 + INMP441
//
//   INMP441 VDD -> 3V3        INMP441 SCK -> GPIO26
//   INMP441 GND -> GND        INMP441 WS  -> GPIO25
//   INMP441 L/R -> GND        INMP441 SD  -> GPIO33
//
// It listens and reports three independent observations over NightMare:
//
//   acoustic:fan_detected         state
//   acoustic:compressor_detected  state
//   acoustic:beep                 event
//
// It does not infer whether the air conditioner is on, or whether a command
// worked: the controller combines these with what it sent.
//
// Layout (src/audio, src/tcp):
//   AudioCapture   I2S DMA -> blocks              high-priority task, lossless
//   WatsonCore     blocks -> windows -> detectors analysis task, bounded
//   RawAudioTcp    blocks -> PC debug stream      own task, drops under pressure
//   WatsonAcoustic glue; the only code that talks to NightMare, from loop()
//
// Serial is the NightMare console (CONFIG LIST, > acoustic:fan_detected, ...)
// plus WATSON STATS. It never carries audio.
// ============================================================

// The Arduino default is 8 KB. loop() only runs NightMare's cooperative tick and
// the publication pump; the internal RAM is worth more elsewhere.
SET_LOOP_TASK_STACK_SIZE(7 * 1024);

namespace
{
    // WATSON STATS: acquisition, analysis, detector and TCP counters.
    NightMareResults onWatsonCommand(const NightMareMessage &message)
    {
        NightMareResults result{false, "", NightmareContext()};

        if (message.command != "WATSON")
            return result;

        result.result = true;
        if (message.subcommand == "STATS" || message.subcommand.length() == 0)
            result.response = watson::gWatson.report();
        else if (message.subcommand == "SELFTEST")
        {
            // Golden windows through this device's own FFT, against the
            // float64 reference.
            char text[768];
            watson::runSelfTest(text, sizeof(text));
            result.response = text;
        }
        else
            result.response = "usage: WATSON STATS | WATSON SELFTEST";
        return result;
    }
} // namespace

// NightMareNetwork's OTA. Flash writes stall the cache, so the audio lost while
// an update runs is accounted for separately (see AudioCapture::setMaintenance).
void onOta(OTA_INFO info, int data)
{
    static int lastPercent = -10;
    switch (info)
    {
    case OTA_START:
        lastPercent = -10;
        watson::gWatson.setMaintenance(true);
        Serial.println("# ota: update started");
        break;
    case OTA_PROGRESS:
        if (data >= lastPercent + 10)
        {
            lastPercent = data;
            Serial.printf("# ota: %d%%\n", data);
        }
        break;
    case OTA_END:
        Serial.println("# ota: update finished, restarting");
        break;
    case OTA_ERROR:
        watson::gWatson.setMaintenance(false);
        Serial.printf("# ota: error %d\n", data);
        break;
    }
}

void setup()
{
    Serial.begin(921600);
    delay(300);

    Serial.println();
    Serial.println("Watson acoustic observer");
    Serial.printf("# heap: %lu free at boot\n", (unsigned long)heap_caps_get_free_size(MALLOC_CAP_8BIT));

    if (!watson::gWatson.begin())
    {
        Serial.println("ERROR: audio start failed; halting");
        while (true)
            delay(1000);
    }

    Serial.printf("# heap: %lu free after audio buffers (largest block %lu)\n",
                  (unsigned long)heap_caps_get_free_size(MALLOC_CAP_8BIT),
                  (unsigned long)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));

    setCommandResolver(onWatsonCommand);
    onOTAEvent(onOta);

    // Resources and Configs are declared and bound; NightMare restores the
    // persisted Configs and starts Wi-Fi, MQTT and OTA from here.
    startNightMareESP();

    watson::gWatson.startAnalysis();

    Serial.printf("# heap: %lu free after NightMare start (largest block %lu)\n",
                  (unsigned long)heap_caps_get_free_size(MALLOC_CAP_8BIT),
                  (unsigned long)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
    Serial.println("Ready. WATSON STATS for counters; CONFIG LIST for settings.");
}

void loop()
{
    tickNightMareESP();
    watson::gWatson.pump();

    // Short enough that a beep or a state change is published within a few
    // ms, long enough to leave the core to the analysis task.
    delay(5);
}
