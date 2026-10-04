#pragma once

#include "DetectorParams.h"

namespace watson
{
    // Detector settings that may reasonably be tuned after deployment, as
    // NightMare Config<T> declarations (acoustic:...). They are persistent, set
    // through the CONFIG console command, and validated here before they can
    // reach a detector. Structural DSP parameters (sample rate, FFT size, hop,
    // DMA geometry) are compile-time and deliberately not Configs.
    //
    // Names:
    //   acoustic:median_ms  acoustic:hold_ms  acoustic:history_ms
    //   acoustic:compressor:band_min_hz  :band_max_hz  :threshold_db
    //   acoustic:compressor:sidebands:enable  :sidebands:thresholds ("lower,upper")
    //   acoustic:fan:mid_threshold_db  :high_threshold_db  :require_both
    //   acoustic:fan:stability_threshold_db  :stability_min_ms
    //   acoustic:beep:min_hz ... (see DetectorConfig.cpp)

    // Installs the write-validation handlers. Call once, before
    // startNightMareESP().
    void installDetectorConfigHandlers();

    // The current Config values as one snapshot. Cheap, but not atomic with
    // respect to a concurrent CONFIG SET, so callers that act on it confirm it
    // twice (WatsonAcoustic::pollConfigs).
    DetectorParams readDetectorParams();
} // namespace watson
