#pragma once

#include "AudioConfig.h"

namespace watson
{
    // The extractor takes the 2048-point transform of a *real* window by
    // packing it into a 1024-point complex one (even samples real, odd samples
    // imaginary) and unpacking the spectrum afterwards. The result is the same
    // 2048-point spectrum; the scratch is half the size, which matters because
    // internal RAM is what Wi-Fi and the TLS session are also fighting over.
    constexpr int kComplexFftSize = kFftSize / 2;

    // In-place forward complex FFT of kComplexFftSize points, split
    // real/imaginary, unnormalised: X[k] = sum x[n] e^{-2 pi i k n / M}.
    //
    // This is the only seam between the feature extractor and an FFT
    // implementation. The firmware uses the arduinoFFT library (Fft.cpp); the
    // host-side parity tests supply their own, so the extractor itself has no
    // Arduino dependency.
    void fftInit();
    void fftForward(float *re, float *im);
} // namespace watson
