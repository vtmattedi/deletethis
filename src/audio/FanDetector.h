#pragma once

#include "DetectorParams.h"

namespace watson
{
    // Is the fan running? Two tests, both must pass (detectors/fan.py):
    //
    //   energy      500-1k and 1k-2k are both above their thresholds
    //   stationary  the 1k-2k level is holding still (std <= threshold) and
    //               enough history stands behind that std to trust it
    //
    // Stateless, and independent of the compressor: a running compressor
    // neither suppresses nor implies the fan. The inputs are the
    // median-smoothed features; the publication hold is the caller's.
    class FanDetector
    {
    public:
        static bool energy(float midDb, float highDb, const DetectorParams &p)
        {
            const bool mid = midDb >= p.fanMidThresholdDb;
            const bool high = highDb >= p.fanHighThresholdDb;
            return p.fanRequireBoth ? (mid && high) : (mid || high);
        }

        // NaN compares false, so an unknown spectrum is never a fan.
        static bool stationary(float highStdDb, float historyMs,
                               const DetectorParams &p)
        {
            return highStdDb <= p.fanStabilityThresholdDb &&
                   historyMs >= static_cast<float>(p.fanStabilityMinMs);
        }

        static bool detect(float midDb, float highDb, float highStdDb,
                           float historyMs, const DetectorParams &p)
        {
            return energy(midDb, highDb, p) &&
                   stationary(highStdDb, historyMs, p);
        }
    };
} // namespace watson
