#pragma once

#include <Arduino.h>
#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>
#include <freertos/semphr.h>
#include <freertos/task.h>

#include <atomic>

#include "AudioBuffer.h"
#include "AudioCapture.h"
#include "DetectorParams.h"
#include "WatsonCore.h"

namespace watson
{
    // What an accepted beep carries onto the network. NightMare's Event codecs
    // are the built-in ones (no application-defined types), so on the wire
    // this travels as a JSON object in a ManagedEvent<String>; the struct is
    // the semantic payload.
    struct BeepPayload
    {
        uint32_t event_id;     // 1, 2, 3 ... for this boot
        uint32_t timestamp_ms; // uptime (millis) when the tone ended
        float peak_hz;
        uint16_t duration_ms;
        float contrast_db;
        float level_db;
    };

    // Writes the JSON encoding into `out`; returns its length.
    size_t encodeBeepPayload(const BeepPayload &payload, char *out, size_t capacity);

    struct AnalysisStats
    {
        volatile uint32_t windows = 0;
        volatile uint32_t maxWindowUs = 0;
        volatile uint64_t sumWindowUs = 0;
        volatile uint32_t maxQueueDepth = 0;
        volatile uint32_t beepsLostInQueue = 0;
    };

    // Ties the pieces together. It orchestrates:
    //
    //   AudioCapture (acquisition task) --blocks--> analysis task --> WatsonCore
    //        |                                          |
    //        '--blocks--> RawAudioTcp (own task)        '--observations--> pump()
    //                                                                        |
    //                                                    NightMare Resources/Events
    //
    // NightMare is only ever touched from pump(), which runs in loop() beside
    // tickNightMareESP(). The acquisition and analysis tasks never call it.
    class WatsonAcoustic
    {
    public:
        // Before startNightMareESP(): buffers, the core, NightMare bindings.
        bool begin();

        // After startNightMareESP() has restored the persisted Configs: reads
        // them, then starts the analysis task and I2S acquisition.
        void startAnalysis();

        // loop() context. Applies Config changes, publishes changed
        // observations, fires accepted beeps. Never blocks.
        void pump();

        // The WATSON STATS text.
        String report();

        // A firmware update is (not) running; see AudioCapture::setMaintenance.
        void setMaintenance(bool active) { capture_.setMaintenance(active); }

        // Test hook: pretend the microphone is dead (WATSON SIMFLAT).
        void simulateDeadMic(bool on) { capture_.setSimulateFlat(on); }

        const AnalysisStats &analysis() const { return stats_; }
        const AudioCapture &capture() const { return capture_; }

    private:
        static void analysisTrampoline(void *self);
        void analysisLoop();
        static void stepTrampoline(void *self, const StepResult &step,
                                   const AudioBlock &newest, uint32_t elapsedUs);
        void handleStep(const StepResult &step, const AudioBlock &newest,
                        uint32_t elapsedUs);
        void applyPendingParams();
        void pollConfigs();

        AudioCapture capture_;
        BlockQueue analysisQueue_;
        WatsonCore *core_ = nullptr;
        AnalysisStats stats_;
        TaskHandle_t analysisTask_ = nullptr;

        // Config -> analysis hand-off. The loop task builds and confirms a
        // snapshot; the analysis task takes it between windows.
        SemaphoreHandle_t paramsLock_ = nullptr;
        DetectorParams pending_;
        std::atomic<uint32_t> paramsVersion_{0};
        uint32_t paramsSeen_ = 0;
        DetectorParams applied_;
        DetectorParams lastPolled_;
        uint32_t lastPollMs_ = 0;
        bool hardwareSeen_ = true;     // analysis task's view
        bool hardwareSent_ = true;     // what NightMare was last told
        uint32_t lastPsCheckMs_ = 0;
        uint32_t psRestored_ = 0;
        bool settled_ = false;
        uint32_t overrunsAtSettle_ = 0;
        uint32_t lateAtSettle_ = 0;

        // Analysis -> loop: state is level-triggered (always the latest
        // published value), the beep is edge-triggered (a queue).
        std::atomic<int8_t> fanState_{-1};
        std::atomic<int8_t> compressorState_{-1};
        int8_t fanSent_ = -2;
        int8_t compressorSent_ = -2;
        QueueHandle_t beepQueue_ = nullptr;
        uint32_t beepIds_ = 0;
        uint32_t beepsFired_ = 0;
    };

    extern WatsonAcoustic gWatson;
} // namespace watson
