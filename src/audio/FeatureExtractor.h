#pragma once

#include <stdint.h>

#include "AudioConfig.h"
#include "DetectorParams.h"

namespace watson
{
    constexpr float kDbFloor = -120.0f;
    constexpr int kMaxHistoryWindows = 128; // 4 s at 32 ms
    constexpr int kBlocksPerWindow = kFftSize / kBlockSamples; // 4

    // Everything one analysis window yields: what the fan and compressor
    // detectors read (before the rolling median), the raw per-window beep
    // features, and the stationarity statistic. Definitions are the PC v2
    // ones (tools/audio/features.py): band level is the dB of the summed
    // per-bin mean-square power of the DC-removed, Hamming-windowed window,
    // normalised by (sum of the window)^2, over bins in [low, high).
    struct Features
    {
        // Stream position of the window's first sample.
        uint64_t startSample = 0;

        float rmsDb = kDbFloor;
        float midDb = kDbFloor;  // 500-1000 Hz
        float highDb = kDbFloor; // 1000-2000 Hz

        float compPrimaryDb = kDbFloor;  // 52-65 Hz (configurable)
        float compLowerDb = kDbFloor;    // 30-45 Hz
        float compShoulderDb = kDbFloor; // 45-52 Hz (diagnostic)
        float compUpperDb = kDbFloor;    // 65-80 Hz

        // Raw, never median-smoothed.
        float beepBandDb = kDbFloor;
        float beepPeakHz = 0.0f;
        float beepNeighbourDb = kDbFloor;
        float beepContrastDb = 0.0f;

        // Population std (ddof 0) of highDb over the history, this window
        // included, and how much history stands behind it.
        float highStdDb = 0.0f;
        uint32_t historyMs = 0;
    };

    // Bin ranges [lo, hi) for every band, derived from the Configs. A band is
    // the set of bins whose centre frequency f satisfies lo <= f < hi, which is
    // what the PC extractor's masks select.
    struct BandPlan
    {
        int mid[2], high[2];
        int primary[2], lower[2], shoulder[2], upper[2];
        int tone[2], left[2], right[2], search[2];
    };

    BandPlan makeBandPlan(const DetectorParams &p);

    float powerToDb(float power);

    // Turns one window -- four consecutive acquisition blocks, read in place --
    // into Features. It keeps no samples of its own: the blocks stay in the
    // acquisition queue until the window has moved past them, so the audio is
    // never copied or held twice. What it does keep is the 1k-2k level history
    // for the stationarity statistic.
    class FeatureExtractor
    {
    public:
        // Allocates the window table and the FFT scratch once. Nothing is
        // allocated afterwards.
        bool begin();
        bool ready() const { return window_ != nullptr; }

        // Bands and history length come from the detector settings.
        void configure(const DetectorParams &p);

        // Forget the stationarity history (the audio it came from is gone).
        void resetHistory();

        // `blocks` are right-aligned 24-bit samples, oldest first;
        // `startSample` is the stream position of blocks[0][0].
        void analyse(const int32_t *const blocks[kBlocksPerWindow],
                     uint64_t startSample, Features &out);

        // The population std the extractor uses, exposed for the golden test.
        static float populationStd(const float *values, int n);

    private:
        // One-sided mean-square power of bin k, 1 <= k < kComplexFftSize.
        float binPower(int k) const;
        float bandPower(const int range[2]) const;

        uint32_t *window_ = nullptr; // Hamming as float bits, n = 0 .. N/2
        float *re_ = nullptr;     // packed complex FFT, kComplexFftSize each
        float *im_ = nullptr;
        float powerScale_ = 0.0f; // 2 / (sum w)^2: one-sided doubling included

        BandPlan bands_{};
        float binHz_ = 0.0f;

        float history_[kMaxHistoryWindows];
        int historyCapacity_ = 0;
        int historyCount_ = 0;
        int historyHead_ = 0;
    };
} // namespace watson
