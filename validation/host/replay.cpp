// Host-side replay of the firmware's analysis core.
//
// Compiles the *same* src/audio sources that run on the ESP32 (feature
// extractor, detectors, WatsonCore) together with the real arduinoFFT
// library, and feeds them a raw PCM file through the same BlockQueue /
// drain() path the analysis task uses. Output is a per-window CSV that
// validation/parity.py compares with the float64 reference.
//
//   replay <pcm.raw> [key=value ...] [--drop=N,N,...] [--quiet]
//
// <pcm.raw>: little-endian int32, right-aligned 24-bit samples (the device's
// own format), 16 kHz mono.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <set>
#include <string>
#include <vector>

#include "../../src/audio/AudioBuffer.h"
#include "../../src/audio/DetectorParams.h"
#include "../../src/audio/SelfTest.h"
#include "../../src/audio/WatsonCore.h"

using namespace watson;

struct Emit
{
    FILE *out;
    WatsonCore *core;
    bool quiet;
};

static void onStep(void *context, const StepResult &step, const AudioBlock &,
                   uint32_t)
{
    Emit *e = static_cast<Emit *>(context);
    const Features &f = e->core->lastFeatures();
    const Smoothed &s = e->core->lastSmoothed();

    if (!e->quiet)
    {
        fprintf(e->out,
                "W,%llu,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
                "%.6f,%u,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.1f,%d,%d,%d,%d\n",
                static_cast<unsigned long long>(f.startSample / 16), f.rmsDb,
                f.midDb, f.highDb, f.compPrimaryDb, f.compLowerDb,
                f.compShoulderDb, f.compUpperDb, f.beepBandDb, f.beepPeakHz,
                f.beepNeighbourDb, f.beepContrastDb, f.highStdDb, f.historyMs,
                s.compPrimaryDb, s.compLowerDb, s.compUpperDb, s.midDb, s.highDb,
                s.highStdDb, static_cast<double>(s.historyMs),
                e->core->fanCandidate() ? 1 : 0,
                e->core->compressorCandidate() ? 1 : 0, e->core->fanPublished(),
                e->core->compressorPublished());
    }

    if (step.fanChanged)
        fprintf(e->out, "FAN,%llu,%d\n",
                static_cast<unsigned long long>(f.startSample / 16),
                e->core->fanPublished());
    if (step.compressorChanged)
        fprintf(e->out, "COMPRESSOR,%llu,%d\n",
                static_cast<unsigned long long>(f.startSample / 16),
                e->core->compressorPublished());
    if (step.beep)
        fprintf(e->out, "BEEP,%lld,%u,%.4f,%.4f,%.4f\n",
                static_cast<long long>(step.beepEvent.endMs),
                static_cast<unsigned>(step.beepEvent.durationMs),
                static_cast<double>(step.beepEvent.peakHz),
                static_cast<double>(step.beepEvent.contrastDb),
                static_cast<double>(step.beepEvent.levelDb));
}

static bool setParam(DetectorParams &p, const std::string &key, double v)
{
#define F(name, field)  \
    if (key == name)    \
    {                   \
        p.field = static_cast<decltype(p.field)>(v); \
        return true;    \
    }
    F("median_ms", medianMs) F("hold_ms", holdMs) F("history_ms", historyMs)
    F("comp_min_hz", compressorMinHz) F("comp_max_hz", compressorMaxHz)
    F("comp_threshold", compressorThresholdDb) F("sidebands", sidebandsEnable)
    F("side_lower", sidebandLowerMarginDb) F("side_upper", sidebandUpperMarginDb)
    F("fan_mid", fanMidThresholdDb) F("fan_high", fanHighThresholdDb)
    F("fan_both", fanRequireBoth) F("fan_stab", fanStabilityThresholdDb)
    F("fan_stab_ms", fanStabilityMinMs)
    F("beep_edge", beepEdgeContrastDb) F("beep_contrast", beepMinContrastDb)
    F("beep_min_ms", beepMinDurationMs) F("beep_max_ms", beepMaxDurationMs)
#undef F
    return false;
}

int main(int argc, char **argv)
{
    if (argc == 2 && std::string(argv[1]) == "--selftest")
    {
        char text[1024];
        const bool ok = runSelfTest(text, sizeof(text));
        printf("%s\n", text);
        return ok ? 0 : 1;
    }

    if (argc < 2)
    {
        fprintf(stderr, "usage: replay <pcm.raw> [key=value ...] [--drop=N,..] [--quiet]\n");
        return 2;
    }

    DetectorParams params;
    std::set<uint32_t> drops;
    bool quiet = false;

    for (int i = 2; i < argc; i++)
    {
        std::string a = argv[i];
        if (a == "--quiet")
            quiet = true;
        else if (a.rfind("--drop=", 0) == 0)
        {
            size_t pos = 7;
            while (pos < a.size())
            {
                size_t comma = a.find(',', pos);
                drops.insert(static_cast<uint32_t>(
                    atoi(a.substr(pos, comma - pos).c_str())));
                if (comma == std::string::npos)
                    break;
                pos = comma + 1;
            }
        }
        else
        {
            size_t eq = a.find('=');
            if (eq == std::string::npos ||
                !setParam(params, a.substr(0, eq), atof(a.c_str() + eq + 1)))
            {
                fprintf(stderr, "bad argument: %s\n", a.c_str());
                return 2;
            }
        }
    }

    if (!paramsConsistent(params))
    {
        fprintf(stderr, "inconsistent parameters\n");
        return 2;
    }

    FILE *in = fopen(argv[1], "rb");
    if (!in)
    {
        perror(argv[1]);
        return 1;
    }

    WatsonCore core;
    if (!core.begin())
        return 1;
    core.configure(params);

    BlockQueue queue;
    queue.begin(kAnalysisQueueBlocks);

    Emit emit{stdout, &core, quiet};
    AudioBlock block;
    memset(&block, 0, sizeof(block));

    uint32_t seq = 0;
    while (fread(block.samples, sizeof(int32_t), kBlockSamples, in) == kBlockSamples)
    {
        block.seq = seq;
        block.capturedMs = seq * kHopMs;
        block.i2sGap = 0;
        if (drops.count(seq) == 0)
            queue.tryPush(block);
        seq++;
        core.drain(queue, onStep, &emit);
    }
    fclose(in);

    const CoreStats &k = core.stats();
    const BeepStats &b = core.beepStats();
    fprintf(stdout,
            "STATS,windows=%u,discontinuities=%u,blocksLost=%u,beeps=%u,weak=%u,"
            "too_short=%u,too_long=%u,unstable=%u\n",
            k.windows, k.discontinuities, k.blocksLost, b.accepted, b.weak,
            b.tooShort, b.tooLong, b.unstablePitch);
    return 0;
}
