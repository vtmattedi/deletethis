#pragma once

#include <stdint.h>

#include "AudioConfig.h"
#include "DetectorParams.h"
#include "FeatureExtractor.h"

namespace watson
{
    // One accepted beep. Times are milliseconds on the stream clock.
    struct BeepEvent
    {
        int64_t endMs;       // estimated end of the tone
        uint16_t durationMs; // flagged-run length, corrected for window overhang
        float peakHz;        // median of the run's interpolated peaks
        float contrastDb;    // best contrast in the run
        float levelDb;       // best band level in the run
    };

    struct BeepStats
    {
        uint32_t accepted = 0;
        uint32_t weak = 0;
        uint32_t tooShort = 0;
        uint32_t tooLong = 0;
        uint32_t unstablePitch = 0;
    };

    // Did the unit just beep? A beep is an event, not a state: the detector
    // follows a run of windows that look like the piezo tone, judges it when
    // it ends, and emits exactly one event per accepted run
    // (detectors/beep.py).
    //
    //   IDLE --valid window--> TONE --enough invalid windows--> judge
    //                           |  valid windows extend the run
    //                           '--> too long: remembered, nothing emitted
    //
    // It reads each window's raw beep features. Median-smoothing them, like
    // the fan and compressor features, would erase a ~150 ms tone.
    class BeepDetector
    {
    public:
        // New settings; a tone in progress was being judged by the old ones,
        // so it is dropped. Counters are kept.
        void configure(const DetectorParams &p);

        // Forget a tone in progress, e.g. across a hole in the audio: a tone
        // seen on both sides of a gap is not one tone we can measure.
        void reset();

        // Consume one window. True when a beep just finished and `out` is set.
        bool update(const Features &f, int64_t timeMs, BeepEvent &out);

        const BeepStats &stats() const { return stats_; }

    private:
        static constexpr int kMaxPeaks = 48;

        struct Run
        {
            int64_t start = 0;
            int64_t last = 0;
            float peaks[kMaxPeaks];
            int peakCount = 0;
            int windows = 0;
            int missing = 0;
            float contrast = -1e30f;
            float level = -1e30f;
            bool tooLong = false;
        };

        bool valid(const Features &f) const;
        void extend(const Features &f, int64_t timeMs);
        int64_t durationMs(const Run &r) const;
        bool finish(BeepEvent &out);

        DetectorParams p_;
        int gapWindows_ = 1;
        bool active_ = false;
        Run run_;
        int64_t blockedUntil_ = INT64_MIN;
        BeepStats stats_;
    };
} // namespace watson
