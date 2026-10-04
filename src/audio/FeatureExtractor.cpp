#include "FeatureExtractor.h"

#include <float.h>
#include <math.h>
#include <string.h>

#include "Alloc.h"
#include "Fft.h"

namespace watson
{
    namespace
    {
        // The bins whose centre frequency f = k * binHz satisfies lo <= f < hi.
        // That is `(freqs >= lo) & (freqs < hi)` in the PC extractor.
        inline int firstBinAtOrAbove(float hz, float binHz)
        {
            float k = ceilf(hz / binHz);
            if (k < 0.0f)
                k = 0.0f;
            if (k > static_cast<float>(kBins))
                k = static_cast<float>(kBins);
            return static_cast<int>(k);
        }

        inline void makeBand(int out[2], float lo, float hi, float binHz)
        {
            out[0] = firstBinAtOrAbove(lo, binHz);
            out[1] = firstBinAtOrAbove(hi, binHz);
            if (out[1] < out[0])
                out[1] = out[0];
        }

        // Bin kComplexFftSize and above cannot be unpacked from the half-size
        // transform. The detectors never look that high (it is 8 kHz).
        inline void clip(int band[2])
        {
            // The unpacking needs bin >= 1 (DC is not a band the detectors use).
            if (band[0] < 1)
                band[0] = 1;
            if (band[1] > kComplexFftSize)
                band[1] = kComplexFftSize;
            if (band[0] > band[1])
                band[0] = band[1];
        }

        inline bool empty(const int band[2]) { return band[1] <= band[0]; }
    } // namespace

    float powerToDb(float power)
    {
        // 20*log10(max(sqrt(p), 1e-12)), floored at -120 dBFS, as in
        // features.power_to_db.
        if (!(power > 1e-24f))
            return kDbFloor;
        const float db = 10.0f * log10f(power);
        return db < kDbFloor ? kDbFloor : db;
    }

    BandPlan makeBandPlan(const DetectorParams &p)
    {
        const float binHz = static_cast<float>(kSampleRate) / kFftSize;
        BandPlan b{};
        makeBand(b.mid, 500.0f, 1000.0f, binHz);
        makeBand(b.high, 1000.0f, 2000.0f, binHz);
        makeBand(b.primary, p.compressorMinHz, p.compressorMaxHz, binHz);
        makeBand(b.lower, kCompressorLowerSidebandHz[0],
                 kCompressorLowerSidebandHz[1], binHz);
        makeBand(b.shoulder, kCompressorShoulderHz[0], kCompressorShoulderHz[1],
                 binHz);
        makeBand(b.upper, kCompressorUpperSidebandHz[0],
                 kCompressorUpperSidebandHz[1], binHz);
        makeBand(b.tone, p.beepMinHz, p.beepMaxHz, binHz);
        makeBand(b.left, p.beepLeftMinHz, p.beepLeftMaxHz, binHz);
        makeBand(b.right, p.beepRightMinHz, p.beepRightMaxHz, binHz);
        // The peak is searched over the tone band and both neighbours:
        // [left_min, right_max), like BEEP_SEARCH_HZ.
        makeBand(b.search, p.beepLeftMinHz, p.beepRightMaxHz, binHz);

        clip(b.mid);
        clip(b.high);
        clip(b.primary);
        clip(b.lower);
        clip(b.shoulder);
        clip(b.upper);
        clip(b.tone);
        clip(b.left);
        clip(b.right);
        clip(b.search);
        return b;
    }

    bool FeatureExtractor::begin()
    {
        if (window_ != nullptr)
            return true;

        // The Hamming table is read as integer words and converted, because
        // the FPU's own load cannot address the 32-bit-only memory it lives in.
        window_ = static_cast<uint32_t *>(
            allocWords(sizeof(uint32_t) * (kFftSize / 2 + 1)));
        re_ = static_cast<float *>(allocBuffer(sizeof(float) * kComplexFftSize));
        im_ = static_cast<float *>(allocBuffer(sizeof(float) * kComplexFftSize));

        if (!window_ || !re_ || !im_)
        {
            window_ = nullptr;
            return false;
        }

        // scipy.signal.get_window("hamming", N) is the periodic form,
        // w[n] = 0.54 - 0.46 cos(2 pi n / N), symmetric about N/2: only
        // n = 0..N/2 is stored and the rest is read mirrored.
        double sum = 0.0;
        for (int n = 0; n < kFftSize; n++)
        {
            const int m = n <= kFftSize / 2 ? n : kFftSize - n;
            const double w =
                0.54 - 0.46 * cos(2.0 * M_PI * m / static_cast<double>(kFftSize));
            if (n <= kFftSize / 2)
            {
                const float f = static_cast<float>(w);
                memcpy(&window_[n], &f, sizeof(f));
            }
            sum += w;
        }
        // Per-bin power = |X|^2 / (sum w)^2, doubled for the one-sided
        // spectrum (every bin used is strictly between DC and Nyquist).
        powerScale_ = static_cast<float>(2.0 / (sum * sum));
        binHz_ = static_cast<float>(kSampleRate) / kFftSize;

        fftInit();
        configure(DetectorParams());
        resetHistory();
        return true;
    }

    void FeatureExtractor::configure(const DetectorParams &p)
    {
        bands_ = makeBandPlan(p);

        // round(history_seconds * sample_rate / hop), half to even like
        // Python's round(); at least one window.
        int windows = static_cast<int>(nearbyintf(
            static_cast<float>(p.historyMs) / static_cast<float>(kHopMs)));
        if (windows < 1)
            windows = 1;
        if (windows > kMaxHistoryWindows)
            windows = kMaxHistoryWindows;

        if (windows != historyCapacity_)
        {
            historyCapacity_ = windows;
            historyCount_ = 0;
            historyHead_ = 0;
        }
    }

    void FeatureExtractor::resetHistory()
    {
        historyCount_ = 0;
        historyHead_ = 0;
    }

    float FeatureExtractor::populationStd(const float *values, int n)
    {
        if (n <= 0)
            return 0.0f;
        float mean = 0.0f;
        for (int i = 0; i < n; i++)
            mean += values[i];
        mean /= static_cast<float>(n);
        float var = 0.0f;
        for (int i = 0; i < n; i++)
        {
            const float d = values[i] - mean;
            var += d * d;
        }
        return sqrtf(var / static_cast<float>(n));
    }

    // X[k] of the real 2048-point window from Z = FFT_1024(x[2n] + i x[2n+1]):
    //
    //   E[k] = (Z[k] + conj(Z[M-k])) / 2          spectrum of the even samples
    //   O[k] = (Z[k] - conj(Z[M-k])) / 2i         spectrum of the odd samples
    //   X[k] = E[k] + exp(-2 pi i k / N) O[k]
    float FeatureExtractor::binPower(int k) const
    {
        const int m = kComplexFftSize - k;
        const float zr = re_[k], zi = im_[k];
        const float wr = re_[m], wi = im_[m];

        const float er = 0.5f * (zr + wr);
        const float ei = 0.5f * (zi - wi);
        const float orr = 0.5f * (zi + wi);
        const float oi = -0.5f * (zr - wr);

        float s, c;
        sincosf(static_cast<float>(2.0 * M_PI) * static_cast<float>(k) /
                    static_cast<float>(kFftSize),
                &s, &c);

        const float xr = er + c * orr + s * oi;
        const float xi = ei + c * oi - s * orr;
        return (xr * xr + xi * xi) * powerScale_;
    }

    float FeatureExtractor::bandPower(const int range[2]) const
    {
        float sum = 0.0f;
        for (int k = range[0]; k < range[1]; k++)
            sum += binPower(k);
        return sum;
    }

    void FeatureExtractor::analyse(const int32_t *const blocks[kBlocksPerWindow],
                                   uint64_t startSample, Features &out)
    {
        out = Features();
        out.startSample = startSample;

        // DC removal, as the PC does twice (once for the rms, once inside
        // window_power). The samples are integers, so the sum is exact.
        int64_t total = 0;
        for (int b = 0; b < kBlocksPerWindow; b++)
            for (int i = 0; i < kBlockSamples; i++)
                total += blocks[b][i];
        const float mean = static_cast<float>(static_cast<double>(total) / kFftSize);
        const float scale = 1.0f / kFullScale;

        auto sample = [&](int i)
        {
            return (static_cast<float>(blocks[i / kBlockSamples][i % kBlockSamples]) -
                    mean) *
                   scale;
        };
        auto hamming = [&](int i)
        {
            const uint32_t bits = window_[i <= kFftSize / 2 ? i : kFftSize - i];
            float w;
            memcpy(&w, &bits, sizeof(w));
            return w;
        };

        // Centre, window and pack: z[n] = w[2n] c[2n] + i w[2n+1] c[2n+1].
        float energy = 0.0f;
        for (int n = 0; n < kComplexFftSize; n++)
        {
            const int even = 2 * n;
            const float ce = sample(even);
            const float co = sample(even + 1);
            energy += ce * ce + co * co;
            re_[n] = ce * hamming(even);
            im_[n] = co * hamming(even + 1);
        }
        out.rmsDb = powerToDb(energy / static_cast<float>(kFftSize));

        fftForward(re_, im_);

        auto level = [&](const int band[2])
        {
            return empty(band) ? kDbFloor : powerToDb(bandPower(band));
        };

        out.midDb = level(bands_.mid);
        out.highDb = level(bands_.high);
        out.compPrimaryDb = level(bands_.primary);
        out.compLowerDb = level(bands_.lower);
        out.compShoulderDb = level(bands_.shoulder);
        out.compUpperDb = level(bands_.upper);

        // ---- Beep (features.beep_features) ----
        if (!empty(bands_.tone) && !empty(bands_.left) && !empty(bands_.right))
        {
            const float tone = bandPower(bands_.tone);
            const float left = bandPower(bands_.left);
            const float right = bandPower(bands_.right);

            const float bandDb = powerToDb(tone);
            const float neighbourDb = powerToDb(left > right ? left : right);
            out.beepBandDb = bandDb;
            out.beepNeighbourDb = neighbourDb;
            out.beepContrastDb = bandDb - neighbourDb;

            int peakBin = bands_.search[0];
            float peakPower = -1.0f;
            for (int k = bands_.search[0]; k < bands_.search[1]; k++)
            {
                const float p = binPower(k);
                if (p > peakPower) // first maximum, like np.argmax
                {
                    peakPower = p;
                    peakBin = k;
                }
            }

            if (peakPower > FLT_MIN)
            {
                float peakHz = peakBin * binHz_;

                // Parabola through the log power of the peak bin and its
                // neighbours; a Hamming-windowed tone spreads over a few bins
                // and its log spectrum is close to parabolic there, which
                // places it to a fraction of a bin.
                if (peakBin > 1 && peakBin < kComplexFftSize - 1)
                {
                    const float a = logf(binPower(peakBin - 1) + FLT_MIN);
                    const float b = logf(binPower(peakBin) + FLT_MIN);
                    const float c = logf(binPower(peakBin + 1) + FLT_MIN);
                    const float curvature = a - 2.0f * b + c;
                    if (curvature < -1e-12f)
                    {
                        float offset = 0.5f * (a - c) / curvature;
                        offset = offset < -1.0f ? -1.0f
                                                : (offset > 1.0f ? 1.0f : offset);
                        peakHz = (peakBin + offset) * binHz_;
                    }
                }
                out.beepPeakHz = peakHz;
            }
        }

        // ---- Fan stationarity: std of 1k-2k over the history ----
        history_[historyHead_] = out.highDb;
        historyHead_ = (historyHead_ + 1) % historyCapacity_;
        if (historyCount_ < historyCapacity_)
            historyCount_++;

        out.highStdDb = populationStd(history_, historyCount_);
        out.historyMs = static_cast<uint32_t>(historyCount_) * kHopMs;
    }
} // namespace watson
