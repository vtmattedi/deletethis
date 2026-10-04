#include "SelfTest.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "FeatureExtractor.h"
#include "GoldenVectors.h"

namespace watson
{
    namespace
    {
        // Documented numerical tolerance, firmware vs the float64 reference.
        // Single-precision FFT and libm give <= 0.02 dB on host and device; these
        // leave room for the Xtensa FPU while staying far below anything a
        // threshold (whole dB) can feel.
        constexpr float kDbTolerance = 0.05f;
        constexpr float kHzTolerance = 0.05f;
        constexpr float kStdTolerance = 1e-4f;
    } // namespace

    bool runSelfTest(char *out, size_t capacity)
    {
        size_t used = 0;
        auto put = [&](const char *format, auto... args)
        {
            if (used < capacity)
                used += snprintf(out + used, capacity - used, format, args...);
        };

        FeatureExtractor extractor;
        if (!extractor.begin())
        {
            put("selftest: out of memory");
            return false;
        }

        bool pass = true;
        const size_t count = sizeof(golden::kVectors) / sizeof(golden::kVectors[0]);

        for (size_t v = 0; v < count; v++)
        {
            const golden::Expected &e = golden::kVectors[v];
            const int32_t *blocks[kBlocksPerWindow];
            for (int b = 0; b < kBlocksPerWindow; b++)
                blocks[b] = e.samples + b * kBlockSamples;

            extractor.resetHistory();
            Features f;
            extractor.analyse(blocks, 0, f);

            struct Item
            {
                const char *name;
                float got, want, tolerance;
                bool floored; // reference sat on the -120 dB floor
            };
            const Item items[] = {
                {"rms", f.rmsDb, e.rms, kDbTolerance, true},
                {"500-1k", f.midDb, e.mid, kDbTolerance, true},
                {"1k-2k", f.highDb, e.high, kDbTolerance, true},
                {"primary", f.compPrimaryDb, e.primary, kDbTolerance, true},
                {"lower", f.compLowerDb, e.lower, kDbTolerance, true},
                {"shoulder", f.compShoulderDb, e.shoulder, kDbTolerance, true},
                {"upper", f.compUpperDb, e.upper, kDbTolerance, true},
                {"beep_band", f.beepBandDb, e.beep_band, kDbTolerance, true},
                {"beep_neighbour", f.beepNeighbourDb, e.beep_neighbour, kDbTolerance,
                 true},
                {"beep_contrast", f.beepContrastDb, e.beep_contrast, kDbTolerance,
                 false},
                // The peak is only meaningful where there is a tone; elsewhere
                // it is an argmax over noise.
                {"beep_peak_hz", f.beepPeakHz, e.beep_peak, kHzTolerance, false},
            };

            float worst = 0.0f;
            const char *worstName = "-";
            bool vectorPass = true;
            for (const Item &item : items)
            {
                if (item.floored && item.want < -119.0f)
                    continue;
                if (strcmp(item.name, "beep_peak_hz") == 0 && e.beep_contrast < 10.0f)
                    continue;
                const float diff = fabsf(item.got - item.want);
                if (diff > worst)
                {
                    worst = diff;
                    worstName = item.name;
                }
                if (diff > item.tolerance)
                    vectorPass = false;
            }
            pass = pass && vectorPass;
            put("%-15s %s  worst |diff|=%.5f (%s)  primary=%.2f dB  mid=%.2f dB\n",
                e.name, vectorPass ? "PASS" : "FAIL", static_cast<double>(worst),
                worstName, static_cast<double>(f.compPrimaryDb),
                static_cast<double>(f.midDb));
        }

        const float std = FeatureExtractor::populationStd(golden::kStdInput, 62);
        const bool stdPass = fabsf(std - golden::kStdExpected) <= kStdTolerance;
        pass = pass && stdPass;
        put("%-15s %s  std=%.6f expected=%.6f\n", "fan std (62)",
            stdPass ? "PASS" : "FAIL", static_cast<double>(std),
            static_cast<double>(golden::kStdExpected));

        put("tolerance: %.2f dB, %.2f Hz  ->  %s", static_cast<double>(kDbTolerance),
            static_cast<double>(kHzTolerance), pass ? "PASS" : "FAIL");
        return pass;
    }
} // namespace watson
