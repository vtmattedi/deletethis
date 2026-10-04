#pragma once

#include <stdint.h>

namespace watson
{
    // Rolling median over the most recent windows of one feature.
    //
    // The PC smoother takes the median of every feature over the last
    // round(median_seconds * window_rate) windows (16 at the defaults), using
    // however many exist at the start. np.median averages the two middle
    // values of an even count. Fan and compressor share the median *stage*;
    // here that is one ring per feature, all sized by the same window count.
    class MedianRing
    {
    public:
        static constexpr int kCapacity = 32;

        void reset()
        {
            count_ = 0;
            head_ = 0;
        }

        void push(float v)
        {
            values_[head_] = v;
            head_ = (head_ + 1) % kCapacity;
            if (count_ < kCapacity)
                count_++;
        }

        // Median of the last `window` pushed values.
        float median(int window) const
        {
            int n = count_ < window ? count_ : window;
            if (n <= 0)
                return 0.0f;
            if (n > kCapacity)
                n = kCapacity;

            float tmp[kCapacity];
            for (int i = 0; i < n; i++)
                tmp[i] = values_[(head_ - 1 - i + 2 * kCapacity) % kCapacity];

            for (int i = 1; i < n; i++)
            {
                const float key = tmp[i];
                int j = i - 1;
                while (j >= 0 && tmp[j] > key)
                {
                    tmp[j + 1] = tmp[j];
                    j--;
                }
                tmp[j + 1] = key;
            }

            if (n & 1)
                return tmp[n / 2];
            return 0.5f * (tmp[n / 2 - 1] + tmp[n / 2]);
        }

        int count() const { return count_; }

    private:
        float values_[kCapacity];
        int count_ = 0;
        int head_ = 0;
    };

    // Publishes a boolean once its candidate has held long enough: the PC
    // HoldTimer. Each observation owns one, so a change in the fan can neither
    // reset nor delay the compressor.
    class HoldTimer
    {
    public:
        // Three-valued: nothing published yet, off, on.
        static constexpr int8_t kUnknown = -1;

        // A gap re-earns the hold. The published value stays: it is the last
        // thing actually observed, and a dropout is not evidence it changed.
        void resetCandidate()
        {
            haveCandidate_ = false;
            since_ = 0;
        }

        // The published value can no longer be trusted (the hardware that
        // produced it is gone): back to "nothing published yet".
        void forget()
        {
            resetCandidate();
            published_ = kUnknown;
        }

        void setHoldMs(uint32_t ms) { holdMs_ = ms; }

        // True when the published value just changed.
        bool update(bool candidate, int64_t timeMs)
        {
            if (!haveCandidate_ || candidate != candidate_)
            {
                candidate_ = candidate;
                haveCandidate_ = true;
                since_ = timeMs;
            }

            stableMs_ = timeMs - since_;

            if ((published_ == kUnknown || (published_ != 0) != candidate) &&
                stableMs_ >= static_cast<int64_t>(holdMs_))
            {
                published_ = candidate ? 1 : 0;
                return true;
            }
            return false;
        }

        int8_t published() const { return published_; }
        bool candidate() const { return haveCandidate_ && candidate_; }
        int64_t stableMs() const { return stableMs_; }

    private:
        uint32_t holdMs_ = 2000;
        bool haveCandidate_ = false;
        bool candidate_ = false;
        int64_t since_ = 0;
        int64_t stableMs_ = 0;
        int8_t published_ = kUnknown;
    };
} // namespace watson
