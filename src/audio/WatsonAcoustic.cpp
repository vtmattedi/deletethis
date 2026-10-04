#include "WatsonAcoustic.h"

#include <NightMare.h>
#include <esp_heap_caps.h>
#include <esp_timer.h>
#include <esp_wifi.h>
#include <math.h>
#include <stdio.h>

#include "../tcp/RawAudioTcp.h"
#include "DetectorConfig.h"

namespace watson
{
    // ------------------------------------------------------------------
    // NightMare declarations. Watson's whole public interface:
    //
    //   acoustic:fan_detected         ManagedSensor<bool>   state
    //   acoustic:compressor_detected  ManagedSensor<bool>   state
    //   acoustic:beep                 ManagedEvent<String>  transient
    //
    // The two states are independent observations and both may be true; there
    // is no combined OFF/FAN/COMPRESSOR resource. Watson does not infer
    // whether the air conditioner is "on" or whether a command worked.
    // ------------------------------------------------------------------
    namespace
    {
        ManagedSensor<bool> fanDetected("acoustic:fan_detected");
        ManagedSensor<bool> compressorDetected("acoustic:compressor_detected");
        ManagedEvent<String> beepEvent("acoustic:beep");

#if ENABLE_TCP
        RawAudioTcp gTcp;
#endif

        constexpr uint32_t kConfigPollMs = 250;
        constexpr uint32_t kSettleMs = 15000;
        constexpr int kBeepQueueLength = 8;
    } // namespace

    WatsonAcoustic gWatson;

    size_t encodeBeepPayload(const BeepPayload &p, char *out, size_t capacity)
    {
        const int n = snprintf(
            out, capacity,
            "{\"event_id\":%lu,\"timestamp_ms\":%lu,\"peak_hz\":%.1f,"
            "\"duration_ms\":%u,\"contrast_db\":%.1f,\"level_db\":%.1f}",
            static_cast<unsigned long>(p.event_id),
            static_cast<unsigned long>(p.timestamp_ms),
            static_cast<double>(p.peak_hz), static_cast<unsigned>(p.duration_ms),
            static_cast<double>(p.contrast_db), static_cast<double>(p.level_db));
        return n < 0 ? 0 : static_cast<size_t>(n);
    }

    // ------------------------------------------------------------------
    // Startup
    // ------------------------------------------------------------------

    bool WatsonAcoustic::begin()
    {
        if (!analysisQueue_.begin(kAnalysisQueueBlocks))
            return false;

        core_ = new WatsonCore();
        if (!core_->begin())
            return false;
        core_->setClock([]() -> uint32_t
                        { return static_cast<uint32_t>(esp_timer_get_time()); });

        paramsLock_ = xSemaphoreCreateMutex();
        beepQueue_ = xQueueCreate(kBeepQueueLength, sizeof(BeepPayload));
        if (paramsLock_ == nullptr || beepQueue_ == nullptr)
            return false;

        capture_.sink(0).queue = &analysisQueue_;

#if ENABLE_TCP
        // The TCP sink stays empty (no queue, disabled) until a client connects.
        capture_.sink(1).enabled = false;
#endif

        // Public interface, bound before startNightMareESP() as NightMare
        // expects. Nothing has a value until the first hold elapses.
        installDetectorConfigHandlers();

        // I2S is started in startAnalysis(), after NightMare is up. Its start-up
        // does flash work (filesystem mount, settings restore) that stalls
        // every task for hundreds of milliseconds -- more than the DMA ring
        // holds -- and audio from those seconds is not worth having.
        return true;
    }

    void WatsonAcoustic::startAnalysis()
    {
        // NightMare has restored the persisted Configs by now.
        DetectorParams params = readDetectorParams();
        if (!paramsConsistent(params))
        {
            Serial.println("# watson: persisted detector settings are inconsistent; "
                           "using defaults");
            params = DetectorParams();
        }
        core_->configure(params);
        applied_ = params;
        lastPolled_ = params;

#if defined(WATSON_NO_AUDIO)
        // Diagnostic build: NightMare only, no I2S, no analysis, no TCP.
        return;
#endif
        xTaskCreatePinnedToCore(&WatsonAcoustic::analysisTrampoline, "analysis", 2560,
                                this, kAnalysisPriority, &analysisTask_,
                                kAnalysisCore);
        capture_.sink(0).task = analysisTask_;

        if (!capture_.begin())
            Serial.println("ERROR: I2S acquisition failed to start");

#if ENABLE_TCP
        gTcp.begin(&capture_.sink(1), AUDIO_TCP_PORT);
#endif
    }

    // ------------------------------------------------------------------
    // Analysis task: bounded work, below acquisition in priority.
    // ------------------------------------------------------------------

    void WatsonAcoustic::analysisTrampoline(void *self)
    {
        static_cast<WatsonAcoustic *>(self)->analysisLoop();
    }

    void WatsonAcoustic::applyPendingParams()
    {
        const uint32_t version = paramsVersion_.load(std::memory_order_acquire);
        if (version == paramsSeen_)
            return;

        DetectorParams next;
        xSemaphoreTake(paramsLock_, portMAX_DELAY);
        next = pending_;
        xSemaphoreGive(paramsLock_);

        paramsSeen_ = version;
        core_->configure(next);
    }

    void WatsonAcoustic::analysisLoop()
    {
        for (;;)
        {
            const uint32_t depth = analysisQueue_.size();
            if (depth > stats_.maxQueueDepth)
                stats_.maxQueueDepth = depth;

            applyPendingParams();
            core_->drain(analysisQueue_, &WatsonAcoustic::stepTrampoline, this);

            ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(100));
        }
    }

    void WatsonAcoustic::stepTrampoline(void *self, const StepResult &step,
                                        const AudioBlock &newest, uint32_t elapsedUs)
    {
        static_cast<WatsonAcoustic *>(self)->handleStep(step, newest, elapsedUs);
    }

    // Hands results to loop(). No NightMare call is made from here.
    void WatsonAcoustic::handleStep(const StepResult &step, const AudioBlock &newest,
                                    uint32_t elapsedUs)
    {
        stats_.windows = stats_.windows + 1;
        stats_.sumWindowUs = stats_.sumWindowUs + elapsedUs;
        if (elapsedUs > stats_.maxWindowUs)
            stats_.maxWindowUs = elapsedUs;

        if (step.fanChanged)
            fanState_.store(core_->fanPublished(), std::memory_order_release);
        if (step.compressorChanged)
            compressorState_.store(core_->compressorPublished(),
                                   std::memory_order_release);

        if (!step.beep)
            return;

        // The newest block's last sample was read at capturedMs and sits at
        // stream time (seq + 1) * hop; the tone ended `lag` before that.
        const int64_t blockEndMs = static_cast<int64_t>(newest.seq + 1) * kHopMs;
        int64_t lag = blockEndMs - step.beepEvent.endMs;
        if (lag < 0)
            lag = 0;
        const uint32_t stamp = newest.capturedMs > static_cast<uint32_t>(lag)
                                   ? newest.capturedMs - static_cast<uint32_t>(lag)
                                   : 0;

        BeepPayload payload;
        payload.event_id = ++beepIds_;
        payload.timestamp_ms = stamp;
        payload.peak_hz = step.beepEvent.peakHz;
        payload.duration_ms = step.beepEvent.durationMs;
        payload.contrast_db = step.beepEvent.contrastDb;
        payload.level_db = step.beepEvent.levelDb;

        if (xQueueSend(beepQueue_, &payload, 0) != pdTRUE)
            stats_.beepsLostInQueue = stats_.beepsLostInQueue + 1;
    }

    // ------------------------------------------------------------------
    // loop() context
    // ------------------------------------------------------------------

    // Builds a snapshot of the Configs, and hands it to analysis once two
    // reads 250 ms apart agree. A CONFIG SET may arrive on another task while
    // this runs, so one read is not trusted on its own.
    void WatsonAcoustic::pollConfigs()
    {
        const uint32_t now = millis();
        if (now - lastPollMs_ < kConfigPollMs)
            return;
        lastPollMs_ = now;

        const DetectorParams current = readDetectorParams();
        const bool stable = current == lastPolled_;
        lastPolled_ = current;

        if (!stable || current == applied_ || !paramsConsistent(current))
            return;

        xSemaphoreTake(paramsLock_, portMAX_DELAY);
        pending_ = current;
        xSemaphoreGive(paramsLock_);
        paramsVersion_.fetch_add(1, std::memory_order_release);
        applied_ = current;
        Serial.println("# watson: detector settings changed");
    }

    void WatsonAcoustic::pump()
    {
        // NightMare's first seconds do flash work (PHY calibration, settings)
        // that stalls every task. Overruns are also counted from the end of
        // that period, which is the number that says whether steady state is
        // lossless.
        if (!settled_ && millis() >= kSettleMs)
        {
            settled_ = true;
            overrunsAtSettle_ = capture_.stats().i2sOverruns;
            lateAtSettle_ = capture_.stats().lateBlocks;
        }

        pollConfigs();

        // The Wi-Fi driver defaults to modem power-save, which delays every
        // packet to the next beacon: a debug stream at 64 KB/s stalls on it,
        // and MQTT publishes are late. This device is mains powered. The mode
        // goes back to the default whenever NightMare reconnects and
        // re-initialises the driver, so it is checked, not set once.
        if (millis() - lastPsCheckMs_ >= 1000)
        {
            lastPsCheckMs_ = millis();
            wifi_ps_type_t mode;
            if (esp_wifi_get_ps(&mode) == ESP_OK && mode != WIFI_PS_NONE)
            {
                esp_wifi_set_ps(WIFI_PS_NONE);
                psRestored_++;
            }
        }
#if ENABLE_TCP
        gTcp.poll();
#endif

#if defined(WATSON_HEAP_TRACE)
        static uint32_t lastTrace = 0;
        if (millis() - lastTrace >= 2000)
        {
            lastTrace = millis();
            Serial.printf("# heap t=%lus 8bit_free=%lu 8bit_largest=%lu 8bit_min=%lu all_free=%lu\n",
                          (unsigned long)(millis() / 1000UL),
                          (unsigned long)heap_caps_get_free_size(MALLOC_CAP_8BIT),
                          (unsigned long)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT),
                          (unsigned long)heap_caps_get_minimum_free_size(MALLOC_CAP_8BIT),
                          (unsigned long)ESP.getFreeHeap());
        }
#endif

        const int8_t fan = fanState_.load(std::memory_order_acquire);
        if (fan != fanSent_ && fan >= 0)
        {
            fanDetected.setValue(fan != 0);
            fanSent_ = fan;
            Serial.printf("# watson: fan_detected=%s\n", fan ? "true" : "false");
        }

        const int8_t compressor = compressorState_.load(std::memory_order_acquire);
        if (compressor != compressorSent_ && compressor >= 0)
        {
            compressorDetected.setValue(compressor != 0);
            compressorSent_ = compressor;
            Serial.printf("# watson: compressor_detected=%s\n",
                          compressor ? "true" : "false");
        }

        BeepPayload payload;
        while (xQueueReceive(beepQueue_, &payload, 0) == pdTRUE)
        {
            char json[160];
            encodeBeepPayload(payload, json, sizeof(json));
            beepEvent.fire(String(json));
            beepsFired_++;
            Serial.printf("# watson: beep #%lu %.0f Hz %u ms %.1f dB\n",
                          static_cast<unsigned long>(payload.event_id),
                          static_cast<double>(payload.peak_hz),
                          static_cast<unsigned>(payload.duration_ms),
                          static_cast<double>(payload.contrast_db));
        }
    }

    // ------------------------------------------------------------------
    // Diagnostics (WATSON STATS)
    // ------------------------------------------------------------------

    String WatsonAcoustic::report()
    {
        const CaptureStats &c = capture_.stats();
        const CoreStats &k = core_->stats();
        const BeepStats &b = core_->beepStats();
        const Smoothed &s = core_->lastSmoothed();
        const Features &f = core_->lastFeatures();

        const uint32_t windows = stats_.windows;
        const uint32_t avgUs =
            windows ? static_cast<uint32_t>(stats_.sumWindowUs / windows) : 0;

        // Built in a static buffer, not by growing a String: this runs when
        // someone asks why things look wrong, which is exactly when the heap
        // may be too fragmented for a String to grow.
        static char buffer[1536];
        size_t used = 0;
        auto add = [&](const char *format, auto... args)
        {
            if (used < sizeof(buffer))
                used += snprintf(buffer + used, sizeof(buffer) - used, format, args...);
        };

        add("acquisition: blocks=%lu i2s_overruns=%lu short_reads=%lu "
                 "read_errors=%lu late_blocks=%lu max_interval_us=%lu@%lums "
                 "last_late=%luus@%lums\n",
                 (unsigned long)c.blocks, (unsigned long)c.i2sOverruns,
                 (unsigned long)c.shortReads, (unsigned long)c.readErrors,
                 (unsigned long)c.lateBlocks, (unsigned long)c.maxIntervalUs,
                 (unsigned long)c.maxIntervalAtMs, (unsigned long)c.lastLateUs,
                 (unsigned long)c.lastLateAtMs);
        add("wifi:        power_save_restored=%lu\n",(unsigned long)psRestored_);
        add("steady:      since %lus: i2s_overruns=%lu late_blocks=%lu%s\n",
            (unsigned long)(kSettleMs / 1000),
            (unsigned long)(settled_ ? c.i2sOverruns - overrunsAtSettle_ : 0),
            (unsigned long)(settled_ ? c.lateBlocks - lateAtSettle_ : 0),
            settled_ ? "" : "  (not settled yet)");
        add("ota:         acquisition pauses=%lu%s\n",
            (unsigned long)c.maintenancePauses,
            capture_.inMaintenance() ? "  (PAUSED: update running)" : "");

        add("analysis:    windows=%lu blocks_lost=%lu resets=%lu "
                 "avg_us=%lu max_us=%lu max_queue=%lu/%d beeps_lost=%lu\n",
                 (unsigned long)windows, (unsigned long)k.blocksLost,
                 (unsigned long)k.discontinuities, (unsigned long)avgUs,
                 (unsigned long)stats_.maxWindowUs,
                 (unsigned long)stats_.maxQueueDepth, kAnalysisQueueBlocks,
                 (unsigned long)stats_.beepsLostInQueue);

        add("fan:         candidate=%d output=%d  mid=%.1f high=%.1f "
                 "std=%.2f history=%.0fms\n",
                 core_->fanCandidate() ? 1 : 0, core_->fanPublished(),
                 (double)s.midDb, (double)s.highDb, (double)s.highStdDb,
                 (double)s.historyMs);

        add("compressor:  candidate=%d output=%d  primary=%.1f lower=%.1f "
                 "upper=%.1f shoulder=%.1f threshold=%.1f sidebands=%s\n",
                 core_->compressorCandidate() ? 1 : 0, core_->compressorPublished(),
                 (double)s.compPrimaryDb, (double)s.compLowerDb,
                 (double)s.compUpperDb, (double)f.compShoulderDb,
                 (double)core_->params().compressorThresholdDb,
                 core_->params().sidebandsEnable ? "on" : "off");

        add("beep:        accepted=%lu fired=%lu weak=%lu too_short=%lu "
                 "too_long=%lu unstable_pitch=%lu  last: peak=%.0fHz "
                 "contrast=%.1f level=%.1f\n",
                 (unsigned long)b.accepted, (unsigned long)beepsFired_,
                 (unsigned long)b.weak, (unsigned long)b.tooShort,
                 (unsigned long)b.tooLong, (unsigned long)b.unstablePitch,
                 (double)f.beepPeakHz, (double)f.beepContrastDb,
                 (double)f.beepBandDb);

        add("core:        fan_windows=%lu "
                 "compressor_windows=%lu  analysis_queue_refused=%lu\n",
                 (unsigned long)k.fanCandidateWindows,
                 (unsigned long)k.compressorCandidateWindows,
                 (unsigned long)analysisQueue_.droppedTotal());

#if ENABLE_TCP
        const TcpStats &t = gTcp.stats();
        add("tcp:         port=%u client=%s frames=%lu dropped=%lu "
                 "timeouts=%lu clients=%lu\n",
                 (unsigned)gTcp.port(), t.connected ? "yes" : "no",
                 (unsigned long)t.framesSent, (unsigned long)t.blocksDropped,
                 (unsigned long)t.writeTimeouts, (unsigned long)t.clients);
#else
        add("tcp:         disabled (ENABLE_TCP=0)\n");
#endif

        add("heap:        8bit_free=%lu 8bit_largest=%lu 8bit_min=%lu "
                 "uptime=%lus  stack_free: acquire=%lu analysis=%lu loop=%lu",
                 (unsigned long)heap_caps_get_free_size(MALLOC_CAP_8BIT),
                 (unsigned long)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT),
                 (unsigned long)heap_caps_get_minimum_free_size(MALLOC_CAP_8BIT),
                 (unsigned long)(millis() / 1000UL),
                 (unsigned long)capture_.stackFreeBytes(),
                 (unsigned long)(analysisTask_ ? uxTaskGetStackHighWaterMark(analysisTask_)
                                               : 0),
                 (unsigned long)uxTaskGetStackHighWaterMark(nullptr));
        return String(buffer);
    }
} // namespace watson
