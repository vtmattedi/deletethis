#pragma once

#include "DetectorParams.h"

namespace watson
{
    // Is the compressor running? One band at or above a threshold
    // (detectors/compressor.py), on the narrow primary band around the
    // measured ~58.6 Hz fundamental:
    //
    //   primary >= threshold
    //
    // and, only when sidebands:enable is set, additionally
    //
    //   primary - lower >= lower_margin   (lower = 30-45 Hz)
    //   primary - upper >= upper_margin   (upper = 65-80 Hz)
    //
    // which rejects a broad low-frequency rumble (the known false episode at
    // ~37-39 Hz with a ~74 Hz harmonic) that happens to put energy in the
    // primary band. Disabled by default. No fan term enters the decision.
    class CompressorDetector
    {
    public:
        static bool detect(float primaryDb, float lowerDb, float upperDb,
                           const DetectorParams &p)
        {
            if (!(primaryDb >= p.compressorThresholdDb))
                return false;

            if (!p.sidebandsEnable)
                return true;

            return primaryDb - lowerDb >= p.sidebandLowerMarginDb &&
                   primaryDb - upperDb >= p.sidebandUpperMarginDb;
        }
    };
} // namespace watson
