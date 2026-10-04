#pragma once

#include <stddef.h>

namespace watson
{
    // Runs the golden windows (GoldenVectors.h: recorded audio and the features
    // the float64 reference computes for it) through the feature extractor on
    // this device -- its FFT, its FPU, its libm -- and writes a report into
    // `out`. Returns true when every feature is inside its tolerance.
    //
    // No Arduino, no NightMare: the host build runs it too.
    bool runSelfTest(char *out, size_t capacity);
} // namespace watson
