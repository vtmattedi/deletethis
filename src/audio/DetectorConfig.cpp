#include "DetectorConfig.h"

#include <NightMare.h>

namespace watson
{
    namespace
    {
        // Firmware defaults live in DetectorParams; every Config is declared
        // from them so the two cannot drift apart.
        const DetectorParams kDefaults;

        // ---- shared smoothing ----
        Config<uint32_t> medianMs("acoustic:median_ms", kDefaults.medianMs);
        Config<uint32_t> holdMs("acoustic:hold_ms", kDefaults.holdMs);
        Config<uint32_t> historyMs("acoustic:history_ms", kDefaults.historyMs);

        // ---- compressor ----
        // Measured fundamental ~58.6 Hz. Bands (see DetectorParams.h):
        //   lower sideband 30-45, lower shoulder 45-52 (diagnostic),
        //   primary 52-65, upper sideband 65-80.
        Config<float> compBandMin("acoustic:compressor:band_min_hz",
                                  kDefaults.compressorMinHz);
        Config<float> compBandMax("acoustic:compressor:band_max_hz",
                                  kDefaults.compressorMaxHz);
        Config<float> compThreshold("acoustic:compressor:threshold_db",
                                    kDefaults.compressorThresholdDb);
        Config<bool> sidebandsEnable("acoustic:compressor:sidebands:enable",
                                     kDefaults.sidebandsEnable);
        // "lower_margin_db,upper_margin_db". NightMare's Config codecs have no
        // fixed pair/array type, so this is a validated String in the
        // canonical "lower,upper" form -- no whitespace. "0,0" is neutral and
        // has no effect while sidebands:enable is false.
        Config<String> sidebandThresholds("acoustic:compressor:sidebands:thresholds",
                                          String("0,0"));

        // ---- fan ----
        Config<float> fanMid("acoustic:fan:mid_threshold_db", kDefaults.fanMidThresholdDb);
        Config<float> fanHigh("acoustic:fan:high_threshold_db",
                              kDefaults.fanHighThresholdDb);
        Config<bool> fanRequireBoth("acoustic:fan:require_both",
                                    kDefaults.fanRequireBoth);
        Config<float> fanStability("acoustic:fan:stability_threshold_db",
                                   kDefaults.fanStabilityThresholdDb);
        Config<uint32_t> fanStabilityMin("acoustic:fan:stability_min_ms",
                                         kDefaults.fanStabilityMinMs);

        // ---- beep ----
        Config<float> beepMinHz("acoustic:beep:min_hz", kDefaults.beepMinHz);
        Config<float> beepMaxHz("acoustic:beep:max_hz", kDefaults.beepMaxHz);
        Config<float> beepLeftMin("acoustic:beep:left_min_hz", kDefaults.beepLeftMinHz);
        Config<float> beepLeftMax("acoustic:beep:left_max_hz", kDefaults.beepLeftMaxHz);
        Config<float> beepRightMin("acoustic:beep:right_min_hz", kDefaults.beepRightMinHz);
        Config<float> beepRightMax("acoustic:beep:right_max_hz", kDefaults.beepRightMaxHz);
        Config<float> beepEdge("acoustic:beep:edge_contrast_db",
                               kDefaults.beepEdgeContrastDb);
        Config<float> beepContrast("acoustic:beep:min_contrast_db",
                                   kDefaults.beepMinContrastDb);
        Config<float> beepLevel("acoustic:beep:min_level_db", kDefaults.beepMinLevelDb);
        Config<uint32_t> beepMinDur("acoustic:beep:min_duration_ms",
                                    kDefaults.beepMinDurationMs);
        Config<uint32_t> beepMaxDur("acoustic:beep:max_duration_ms",
                                    kDefaults.beepMaxDurationMs);
        Config<float> beepSpread("acoustic:beep:max_peak_spread_hz",
                                 kDefaults.beepMaxPeakSpreadHz);
        Config<uint32_t> beepGap("acoustic:beep:max_gap_ms", kDefaults.beepMaxGapMs);
        Config<uint32_t> beepRefractory("acoustic:beep:refractory_ms",
                                        kDefaults.beepRefractoryMs);

        // One handler per field: the request is checked together with the
        // *current* value of every other field, so a write that would leave
        // the detectors in an impossible state (min >= max, edge contrast
        // above the peak contrast, a band beyond Nyquist, NaN) is refused and
        // the old value stays. Related fields that must move together are
        // changed in an order that keeps each intermediate state valid (for
        // example raise a band's max before its min).
        template <typename T, T DetectorParams::*Field>
        bool validated(Config<T> &, const T &requested)
        {
            DetectorParams p = readDetectorParams();
            p.*Field = requested;
            return paramsConsistent(p);
        }

        bool validatedSidebands(Config<String> &, const String &requested)
        {
            float lower, upper;
            return parseSidebandMargins(requested.c_str(), lower, upper);
        }
    } // namespace

    DetectorParams readDetectorParams()
    {
        DetectorParams p;
        p.medianMs = medianMs.value();
        p.holdMs = holdMs.value();
        p.historyMs = historyMs.value();

        p.compressorMinHz = compBandMin.value();
        p.compressorMaxHz = compBandMax.value();
        p.compressorThresholdDb = compThreshold.value();
        p.sidebandsEnable = sidebandsEnable.value();
        float lower = 0.0f, upper = 0.0f;
        if (parseSidebandMargins(sidebandThresholds.value().c_str(), lower, upper))
        {
            p.sidebandLowerMarginDb = lower;
            p.sidebandUpperMarginDb = upper;
        }

        p.fanMidThresholdDb = fanMid.value();
        p.fanHighThresholdDb = fanHigh.value();
        p.fanRequireBoth = fanRequireBoth.value();
        p.fanStabilityThresholdDb = fanStability.value();
        p.fanStabilityMinMs = fanStabilityMin.value();

        p.beepMinHz = beepMinHz.value();
        p.beepMaxHz = beepMaxHz.value();
        p.beepLeftMinHz = beepLeftMin.value();
        p.beepLeftMaxHz = beepLeftMax.value();
        p.beepRightMinHz = beepRightMin.value();
        p.beepRightMaxHz = beepRightMax.value();
        p.beepEdgeContrastDb = beepEdge.value();
        p.beepMinContrastDb = beepContrast.value();
        p.beepMinLevelDb = beepLevel.value();
        p.beepMinDurationMs = beepMinDur.value();
        p.beepMaxDurationMs = beepMaxDur.value();
        p.beepMaxPeakSpreadHz = beepSpread.value();
        p.beepMaxGapMs = beepGap.value();
        p.beepRefractoryMs = beepRefractory.value();
        return p;
    }

    void installDetectorConfigHandlers()
    {
        medianMs.onWrite = validated<uint32_t, &DetectorParams::medianMs>;
        holdMs.onWrite = validated<uint32_t, &DetectorParams::holdMs>;
        historyMs.onWrite = validated<uint32_t, &DetectorParams::historyMs>;

        compBandMin.onWrite = validated<float, &DetectorParams::compressorMinHz>;
        compBandMax.onWrite = validated<float, &DetectorParams::compressorMaxHz>;
        compThreshold.onWrite =
            validated<float, &DetectorParams::compressorThresholdDb>;
        sidebandsEnable.onWrite = validated<bool, &DetectorParams::sidebandsEnable>;
        sidebandThresholds.onWrite = validatedSidebands;

        fanMid.onWrite = validated<float, &DetectorParams::fanMidThresholdDb>;
        fanHigh.onWrite = validated<float, &DetectorParams::fanHighThresholdDb>;
        fanRequireBoth.onWrite = validated<bool, &DetectorParams::fanRequireBoth>;
        fanStability.onWrite =
            validated<float, &DetectorParams::fanStabilityThresholdDb>;
        fanStabilityMin.onWrite =
            validated<uint32_t, &DetectorParams::fanStabilityMinMs>;

        beepMinHz.onWrite = validated<float, &DetectorParams::beepMinHz>;
        beepMaxHz.onWrite = validated<float, &DetectorParams::beepMaxHz>;
        beepLeftMin.onWrite = validated<float, &DetectorParams::beepLeftMinHz>;
        beepLeftMax.onWrite = validated<float, &DetectorParams::beepLeftMaxHz>;
        beepRightMin.onWrite = validated<float, &DetectorParams::beepRightMinHz>;
        beepRightMax.onWrite = validated<float, &DetectorParams::beepRightMaxHz>;
        beepEdge.onWrite = validated<float, &DetectorParams::beepEdgeContrastDb>;
        beepContrast.onWrite = validated<float, &DetectorParams::beepMinContrastDb>;
        beepLevel.onWrite = validated<float, &DetectorParams::beepMinLevelDb>;
        beepMinDur.onWrite = validated<uint32_t, &DetectorParams::beepMinDurationMs>;
        beepMaxDur.onWrite = validated<uint32_t, &DetectorParams::beepMaxDurationMs>;
        beepSpread.onWrite = validated<float, &DetectorParams::beepMaxPeakSpreadHz>;
        beepGap.onWrite = validated<uint32_t, &DetectorParams::beepMaxGapMs>;
        beepRefractory.onWrite = validated<uint32_t, &DetectorParams::beepRefractoryMs>;
    }
} // namespace watson
