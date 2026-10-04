#pragma once

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "AudioConfig.h"

// The tunable detector settings as one plain struct: no Arduino, no NightMare,
// so the detectors and the host-side parity tests can use it unchanged.
// DetectorConfig.{h,cpp} turns these into NightMare Config<T> declarations and
// hands the detectors a validated snapshot.
//
// The defaults are the PC v2 defaults, except the compressor, which is the
// narrow 52-65 Hz feature with a threshold derived for it.

namespace watson
{
    // --- Compressor ---------------------------------------------------
    //
    // A running compressor puts sustained energy at a measured fundamental of
    // ~58.6 Hz (7.5 bins of 7.8125 Hz). The detector is a band-energy test,
    // not a frequency-equality test:
    //
    //   lower sideband   30 - 45 Hz   guard (a false episode sat at ~37-39 Hz)
    //   lower shoulder   45 - 52 Hz   diagnostic only
    //   primary          52 - 65 Hz   the v0 feature
    //   upper sideband   65 - 80 Hz   guard (and the ~74 Hz harmonic)
    //
    // The old PC feature was the whole 30-80 Hz band with a -38 dB threshold.
    // Narrowing the band changes the power it integrates, so -38 does not
    // carry over. -43.0 dB was derived over the reviewed recordings with this
    // firmware's exact feature math (validation/derive_compressor.py):
    // balanced accuracy 0.976 (recall 96.2 %, false positives 1.0 %), the
    // plateau of near-optimal thresholds is -50 ... -41, and a leave-one-group-
    // out search over 87 groups picks -44 ... -42.
    constexpr float kCompressorLowerSidebandHz[2] = {30.0f, 45.0f};
    constexpr float kCompressorShoulderHz[2] = {45.0f, 52.0f};
    constexpr float kCompressorUpperSidebandHz[2] = {65.0f, 80.0f};

    struct DetectorParams
    {
        // --- Shared smoothing (fan and compressor) -------------------
        uint32_t medianMs = 500;   // rolling median of the features
        uint32_t holdMs = 2000;    // each observation's own publication hold
        uint32_t historyMs = 2000; // history the fan stationarity std spans

        // --- Compressor ---------------------------------------------
        float compressorMinHz = 52.0f;
        float compressorMaxHz = 65.0f;
        float compressorThresholdDb = -43.0f;
        bool sidebandsEnable = false;
        // primary - lower >= lowerMarginDb && primary - upper >= upperMarginDb.
        // "0,0" is deliberately neutral; the margins have not been frozen.
        float sidebandLowerMarginDb = 0.0f;
        float sidebandUpperMarginDb = 0.0f;

        // --- Fan -----------------------------------------------------
        float fanMidThresholdDb = -62.0f;  // 500-1k
        float fanHighThresholdDb = -65.0f; // 1k-2k
        bool fanRequireBoth = true;
        float fanStabilityThresholdDb = 6.0f; // 1k-2k_std <= this
        uint32_t fanStabilityMinMs = 1000;

        // --- Beep ----------------------------------------------------
        float beepMinHz = 4050.0f;
        float beepMaxHz = 4180.0f;
        float beepLeftMinHz = 3850.0f;
        float beepLeftMaxHz = 4000.0f;
        float beepRightMinHz = 4230.0f;
        float beepRightMaxHz = 4380.0f;
        float beepEdgeContrastDb = 10.0f;
        float beepMinContrastDb = 15.0f;
        float beepMinLevelDb = -80.0f;
        uint32_t beepMinDurationMs = 60;
        uint32_t beepMaxDurationMs = 300;
        float beepMaxPeakSpreadHz = 50.0f;
        uint32_t beepMaxGapMs = 40;
        uint32_t beepRefractoryMs = 80;

        bool operator==(const DetectorParams &o) const;
        bool operator!=(const DetectorParams &o) const { return !(*this == o); }
    };

    constexpr float kNyquistHz = kSampleRate / 2.0f;

    inline bool finite(float v) { return !isnan(v) && !isinf(v); }

    // A [lo, hi) band that lies inside the spectrum.
    inline bool bandOk(float lo, float hi)
    {
        return finite(lo) && finite(hi) && lo >= 0.0f && lo < hi &&
               hi <= kNyquistHz;
    }

    // Cross-field consistency of a whole snapshot. The Config write handlers
    // check one field against the current others; this is the final word.
    inline bool paramsConsistent(const DetectorParams &p)
    {
        // Windows are 32 ms; the rings are bounded.
        if (p.medianMs < kHopMs || p.medianMs > 1000)
            return false;
        if (p.historyMs < kHopMs || p.historyMs > 4000)
            return false;
        if (p.holdMs > 60000)
            return false;

        if (!bandOk(p.compressorMinHz, p.compressorMaxHz))
            return false;
        if (!finite(p.compressorThresholdDb) ||
            !finite(p.sidebandLowerMarginDb) ||
            !finite(p.sidebandUpperMarginDb))
            return false;

        if (!finite(p.fanMidThresholdDb) || !finite(p.fanHighThresholdDb) ||
            !finite(p.fanStabilityThresholdDb) ||
            p.fanStabilityThresholdDb < 0.0f)
            return false;

        if (!bandOk(p.beepMinHz, p.beepMaxHz) ||
            !bandOk(p.beepLeftMinHz, p.beepLeftMaxHz) ||
            !bandOk(p.beepRightMinHz, p.beepRightMaxHz))
            return false;
        if (!finite(p.beepEdgeContrastDb) || !finite(p.beepMinContrastDb) ||
            !finite(p.beepMinLevelDb) || !finite(p.beepMaxPeakSpreadHz) ||
            p.beepMaxPeakSpreadHz < 0.0f)
            return false;
        if (p.beepEdgeContrastDb > p.beepMinContrastDb)
            return false;
        if (p.beepMinDurationMs == 0 ||
            p.beepMinDurationMs >= p.beepMaxDurationMs ||
            p.beepMaxDurationMs > 1000) // a run keeps at most 48 windows
            return false;
        if (p.beepMaxGapMs > 1000 || p.beepRefractoryMs > 10000)
            return false;
        return true;
    }

    inline bool DetectorParams::operator==(const DetectorParams &o) const
    {
        return medianMs == o.medianMs && holdMs == o.holdMs &&
               historyMs == o.historyMs &&
               compressorMinHz == o.compressorMinHz &&
               compressorMaxHz == o.compressorMaxHz &&
               compressorThresholdDb == o.compressorThresholdDb &&
               sidebandsEnable == o.sidebandsEnable &&
               sidebandLowerMarginDb == o.sidebandLowerMarginDb &&
               sidebandUpperMarginDb == o.sidebandUpperMarginDb &&
               fanMidThresholdDb == o.fanMidThresholdDb &&
               fanHighThresholdDb == o.fanHighThresholdDb &&
               fanRequireBoth == o.fanRequireBoth &&
               fanStabilityThresholdDb == o.fanStabilityThresholdDb &&
               fanStabilityMinMs == o.fanStabilityMinMs &&
               beepMinHz == o.beepMinHz && beepMaxHz == o.beepMaxHz &&
               beepLeftMinHz == o.beepLeftMinHz &&
               beepLeftMaxHz == o.beepLeftMaxHz &&
               beepRightMinHz == o.beepRightMinHz &&
               beepRightMaxHz == o.beepRightMaxHz &&
               beepEdgeContrastDb == o.beepEdgeContrastDb &&
               beepMinContrastDb == o.beepMinContrastDb &&
               beepMinLevelDb == o.beepMinLevelDb &&
               beepMinDurationMs == o.beepMinDurationMs &&
               beepMaxDurationMs == o.beepMaxDurationMs &&
               beepMaxPeakSpreadHz == o.beepMaxPeakSpreadHz &&
               beepMaxGapMs == o.beepMaxGapMs &&
               beepRefractoryMs == o.beepRefractoryMs;
    }

    // Parses the canonical "lower,upper" encoding of
    // acoustic:compressor:sidebands:thresholds. Used by the Config
    // validation handler and by the snapshot builder.
    inline bool parseSidebandMargins(const char *text, float &lower, float &upper)
    {
        if (text == nullptr)
            return false;
        // Canonical form only: strtof would otherwise accept padding.
        for (const char *c = text; *c != '\0'; c++)
            if (*c == ' ' || *c == '\t' || *c == '\r' || *c == '\n')
                return false;
        char *end = nullptr;
        const float a = strtof(text, &end);
        if (end == text || *end != ',')
            return false;
        const char *second = end + 1;
        const float b = strtof(second, &end);
        if (end == second || *end != '\0')
            return false;
        if (!finite(a) || !finite(b))
            return false;
        lower = a;
        upper = b;
        return true;
    }
} // namespace watson
