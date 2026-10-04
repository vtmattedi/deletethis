#pragma once

#include <stdint.h>

#include "AudioBuffer.h"
#include "AudioConfig.h"
#include "BeepDetector.h"
#include "CompressorDetector.h"
#include "DetectorParams.h"
#include "FanDetector.h"
#include "FeatureExtractor.h"
#include "Smoothing.h"

namespace watson
{
    // Everything between "blocks in the acquisition queue" and "the
    // observations changed": windows, features, the shared median, the two
    // independent detectors with their own hold timers, and the beep state
    // machine. No Arduino, no FreeRTOS, no NightMare: the analysis task drives
    // it on the target and the host parity/replay tests drive it unchanged.
    //
    // It reports; it does not decide policy: no AC state, no command success.
    struct CoreStats
    {
        uint32_t windows = 0;
        uint32_t discontinuities = 0; // times the history was dropped
        uint32_t blocksLost = 0;      // refused by the queue, or lost to an I2S overrun
        uint32_t fanCandidateWindows = 0;
        uint32_t compressorCandidateWindows = 0;
    };

    struct StepResult
    {
        bool fanChanged = false;
        bool compressorChanged = false;
        bool beep = false; // `beepEvent` is valid
        BeepEvent beepEvent{};
    };

    // The median-smoothed features the fan and compressor read.
    struct Smoothed
    {
        float compPrimaryDb = kDbFloor;
        float compLowerDb = kDbFloor;
        float compUpperDb = kDbFloor;
        float midDb = kDbFloor;
        float highDb = kDbFloor;
        float highStdDb = 0.0f;
        float historyMs = 0.0f;
    };

    // Called once per analysed window. `newest` is the last block the window
    // covers; `elapsedUs` is the time the window took (0 without a clock).
    using StepFn = void (*)(void *context, const StepResult &step,
                            const AudioBlock &newest, uint32_t elapsedUs);

    class WatsonCore
    {
    public:
        bool begin();

        // Optional microsecond clock, to time windows.
        void setClock(uint32_t (*nowUs)()) { clock_ = nowUs; }

        // New settings. What each change touches is deliberate:
        //   thresholds                   applied at once, nothing reset
        //   median window                median rings + both candidates
        //   history length               stationarity history + fan candidate
        //   compressor band              compressor ring + its candidate
        //   beep settings                the beep run in progress
        // A published fan/compressor value is never reset by a settings change.
        void configure(const DetectorParams &p);
        const DetectorParams &params() const { return params_; }

        // Consumes what it can from the acquisition queue, without blocking.
        //
        // A window is the four oldest queued blocks (2048 samples), read in
        // place; after each one the oldest block is released, so the queue
        // keeps the last three as history and nothing is ever copied. Audio is
        // only ever analysed across blocks that follow each other exactly: a
        // hole -- blocks the queue refused, or an I2S overrun -- drops
        // everything that was being built across it (see discontinuity()) and
        // the next window starts from audio received after it.
        // Returns the number of windows analysed.
        int drain(BlockQueue &queue, StepFn onStep, void *context);

        // Everything that cannot span a hole is dropped. Published
        // observations stay; each must re-earn its hold. A beep spanning the
        // hole is discarded.
        void discontinuity();

        // -1 until the first hold elapses, then 0 / 1.
        int8_t fanPublished() const { return fanHold_.published(); }
        int8_t compressorPublished() const { return compressorHold_.published(); }
        bool fanCandidate() const { return fanHold_.candidate(); }
        bool compressorCandidate() const { return compressorHold_.candidate(); }

        const CoreStats &stats() const { return stats_; }
        const BeepStats &beepStats() const { return beep_.stats(); }
        const Features &lastFeatures() const { return last_; }
        const Smoothed &lastSmoothed() const { return smoothed_; }

    private:
        void onWindow(const Features &f, StepResult &result);
        int medianWindows() const;

        FeatureExtractor extractor_;
        DetectorParams params_;
        BeepDetector beep_;

        MedianRing primary_, lower_, upper_;
        MedianRing mid_, high_, highStd_, historyMs_;
        HoldTimer fanHold_, compressorHold_;

        uint32_t (*clock_)() = nullptr;
        uint32_t windowsSinceReset_ = 0;

        Features last_;
        Smoothed smoothed_;
        CoreStats stats_;
    };
} // namespace watson
