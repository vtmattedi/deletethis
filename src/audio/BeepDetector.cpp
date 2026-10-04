#include "BeepDetector.h"

#include <math.h>

namespace watson
{
    void BeepDetector::configure(const DetectorParams &p)
    {
        p_ = p;
        // round(max_gap_ms / 1000 / hop_seconds), half to even, at least 0.
        const int gap = static_cast<int>(
            nearbyintf(static_cast<float>(p.beepMaxGapMs) /
                       static_cast<float>(kHopMs)));
        gapWindows_ = gap < 0 ? 0 : gap;
        reset();
    }

    void BeepDetector::reset()
    {
        active_ = false;
        blockedUntil_ = INT64_MIN;
    }

    bool BeepDetector::valid(const Features &f) const
    {
        return f.beepContrastDb >= p_.beepEdgeContrastDb &&
               f.beepBandDb >= p_.beepMinLevelDb &&
               f.beepPeakHz >= p_.beepMinHz && f.beepPeakHz <= p_.beepMaxHz;
    }

    // The tone's length, corrected for the window overhang: the flagged
    // windows cover the tone plus about one footprint of overhang, which is the
    // hop's worth of time each window adds beyond the first, less the
    // footprint itself. See kBeepFootprintMs.
    int64_t BeepDetector::durationMs(const Run &r) const
    {
        const int64_t span = (r.last - r.start) + static_cast<int64_t>(kHopMs) -
                             (static_cast<int64_t>(kBeepFootprintMs) -
                              static_cast<int64_t>(kHopMs));
        return span < 0 ? 0 : span;
    }

    void BeepDetector::extend(const Features &f, int64_t timeMs)
    {
        Run &r = run_;
        r.last = timeMs;
        r.missing = 0;
        r.windows++;
        if (r.peakCount < kMaxPeaks)
            r.peaks[r.peakCount++] = f.beepPeakHz;
        else
            r.tooLong = true; // far past any plausible beep; cannot judge
        if (f.beepContrastDb > r.contrast)
            r.contrast = f.beepContrastDb;
        if (f.beepBandDb > r.level)
            r.level = f.beepBandDb;
        if (durationMs(r) > static_cast<int64_t>(p_.beepMaxDurationMs))
            r.tooLong = true;
    }

    bool BeepDetector::update(const Features &f, int64_t timeMs, BeepEvent &out)
    {
        const bool ok = valid(f);

        if (!active_)
        {
            if (ok && timeMs >= blockedUntil_)
            {
                active_ = true;
                run_ = Run();
                run_.start = timeMs;
                run_.last = timeMs;
                extend(f, timeMs);
            }
            return false;
        }

        if (ok)
        {
            extend(f, timeMs);
            return false;
        }

        run_.missing++;
        if (run_.missing <= gapWindows_)
            return false;

        return finish(out);
    }

    bool BeepDetector::finish(BeepEvent &out)
    {
        Run &r = run_;
        active_ = false;

        const int64_t duration = durationMs(r);

        // Windows that still overlap the tone, plus the refractory, are not
        // allowed to start another run.
        blockedUntil_ = r.last + static_cast<int64_t>(kBeepFootprintMs) +
                        static_cast<int64_t>(p_.beepRefractoryMs);

        if (r.tooLong)
        {
            stats_.tooLong++;
            return false;
        }

        if (r.contrast < p_.beepMinContrastDb)
        {
            // Tone-like for a while, but never stood out enough.
            stats_.weak++;
            return false;
        }

        if (duration < static_cast<int64_t>(p_.beepMinDurationMs))
        {
            stats_.tooShort++;
            return false;
        }

        float lo = r.peaks[0];
        float hi = r.peaks[0];
        for (int i = 1; i < r.peakCount; i++)
        {
            if (r.peaks[i] < lo)
                lo = r.peaks[i];
            if (r.peaks[i] > hi)
                hi = r.peaks[i];
        }
        if (hi - lo > p_.beepMaxPeakSpreadHz)
        {
            // A narrow peak whose pitch wanders is not a piezo tone.
            stats_.unstablePitch++;
            return false;
        }

        // Median of the run's peaks (np.median).
        float sorted[kMaxPeaks];
        const int n = r.peakCount;
        for (int i = 0; i < n; i++)
        {
            float key = r.peaks[i];
            int j = i - 1;
            while (j >= 0 && sorted[j] > key)
            {
                sorted[j + 1] = sorted[j];
                j--;
            }
            sorted[j + 1] = key;
        }
        const float median =
            (n & 1) ? sorted[n / 2] : 0.5f * (sorted[n / 2 - 1] + sorted[n / 2]);

        // The tone sits in the middle of the flagged windows.
        const int64_t middle =
            (r.start + r.last + static_cast<int64_t>(kBeepFootprintMs)) / 2;

        out.endMs = middle + duration / 2;
        out.durationMs = static_cast<uint16_t>(duration > 65535 ? 65535 : duration);
        out.peakHz = median;
        out.contrastDb = r.contrast;
        out.levelDb = r.level;

        stats_.accepted++;
        return true;
    }
} // namespace watson
