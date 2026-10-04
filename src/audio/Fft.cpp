#include "Fft.h"

#include <arduinoFFT.h>

// arduinoFFT swaps vImag during bit reversal only under COMPLEX_INPUT. Without
// it the transform of a complex input is silently wrong.
#ifndef COMPLEX_INPUT
#error "build with -DCOMPLEX_INPUT (platformio.ini build_flags)"
#endif

namespace watson
{
    // The firmware's FFT is the arduinoFFT library, single precision: the
    // ESP32 has a hardware FPU for float and none for double.
    namespace
    {
        ArduinoFFT<float> gFft;
    }

    void fftInit() {}

    void fftForward(float *re, float *im)
    {
        gFft.compute(re, im, kComplexFftSize, FFTDirection::Forward);
    }
} // namespace watson
