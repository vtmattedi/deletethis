#include "WatsonCore.h"

#include <math.h>

namespace watson
{
    bool WatsonCore::begin()
    {
        if (!extractor_.begin())
            return false;
        configure(DetectorParams());
        return true;
    }

    // round(median_seconds * window_rate), half to even, at least 1.
    int WatsonCore::medianWindows() const
    {
        int n = static_cast<int>(nearbyintf(
            static_cast<float>(params_.medianMs) / static_cast<float>(kHopMs)));
        if (n < 1)
            n = 1;
        if (n > MedianRing::kCapacity)
            n = MedianRing::kCapacity;
        return n;
    }

    void WatsonCore::configure(const DetectorParams &p)
    {
        const DetectorParams old = params_;
        params_ = p;

        extractor_.configure(p);
        fanHold_.setHoldMs(p.holdMs);
        compressorHold_.setHoldMs(p.holdMs);

        if (p.medianMs != old.medianMs)
        {
            primary_.reset();
            lower_.reset();
            upper_.reset();
            mid_.reset();
            high_.reset();
            highStd_.reset();
            historyMs_.reset();
            fanHold_.resetCandidate();
            compressorHold_.resetCandidate();
        }

        if (p.historyMs != old.historyMs)
        {
            // The extractor already restarted the history in configure(); what
            // the fan reads from it is stale.
            highStd_.reset();
            historyMs_.reset();
            fanHold_.resetCandidate();
        }

        if (p.compressorMinHz != old.compressorMinHz ||
            p.compressorMaxHz != old.compressorMaxHz)
        {
            primary_.reset();
            compressorHold_.resetCandidate();
        }

        // Every beep setting may change how a run in progress is judged.
        beep_.configure(p);
    }

    void WatsonCore::discontinuity()
    {
        extractor_.resetHistory();
        primary_.reset();
        lower_.reset();
        upper_.reset();
        mid_.reset();
        high_.reset();
        highStd_.reset();
        historyMs_.reset();
        fanHold_.resetCandidate();
        compressorHold_.resetCandidate();
        beep_.reset();
        windowsSinceReset_ = 0;
        stats_.discontinuities++;
    }

    void WatsonCore::hardwareChanged()
    {
        discontinuity();
        fanHold_.forget();
        compressorHold_.forget();
    }

    int WatsonCore::drain(BlockQueue &queue, StepFn onStep, void *context)
    {
        int analysed = 0;

        for (;;)
        {
            // Look at the oldest window's worth of blocks.
            const AudioBlock *window[kBlocksPerWindow];
            uint32_t have = 0;
            while (have < kBlocksPerWindow &&
                   (window[have] = queue.at(have)) != nullptr)
                have++;

            // Where is the first thing that stops these from being one
            // continuous stretch of trusted audio?
            //   - a block whose samples cannot be trusted (I2S overrun in it)
            //   - a block the queue refused just before it
            // A block is judged only against its predecessor *in the queue*:
            // holes before the oldest block were handled when it became the
            // oldest.
            uint32_t stop = have;
            for (uint32_t i = 0; i < have; i++)
            {
                const bool bad = window[i]->i2sGap;
                const bool hole =
                    i > 0 && (window[i]->lostBefore > 0 ||
                              window[i]->seq != window[i - 1]->seq + 1);
                if (bad || hole)
                {
                    stop = i;
                    break;
                }
            }

            if (stop < have)
            {
                // Blocks before `stop` can never complete a window. An
                // untrusted block itself goes too.
                const uint32_t drop = window[stop]->i2sGap ? stop + 1 : stop;
                uint32_t lost = window[stop]->lostBefore;
                if (stop > 0 && window[stop]->seq > window[stop - 1]->seq + 1 &&
                    window[stop]->seq - window[stop - 1]->seq - 1 > lost)
                    lost = window[stop]->seq - window[stop - 1]->seq - 1;
                stats_.blocksLost += lost + (window[stop]->i2sGap ? 1u : 0u);
                discontinuity();
                for (uint32_t i = 0; i < drop; i++)
                    queue.pop();
                continue;
            }

            if (have < kBlocksPerWindow)
                break; // not enough audio yet

            // The oldest block can itself follow a hole (the queue refused
            // blocks while it was empty, say): the history is not continuous
            // with it.
            if (window[0]->lostBefore > 0 && windowsSinceReset_ > 0)
            {
                stats_.blocksLost += window[0]->lostBefore;
                discontinuity();
            }

            const int32_t *blocks[kBlocksPerWindow];
            for (int i = 0; i < kBlocksPerWindow; i++)
                blocks[i] = window[i]->samples;

            const uint32_t t0 = clock_ ? clock_() : 0;

            Features f;
            extractor_.analyse(
                blocks, static_cast<uint64_t>(window[0]->seq) * kBlockSamples, f);

            StepResult result;
            onWindow(f, result);

            const uint32_t elapsed = clock_ ? clock_() - t0 : 0;
            if (onStep != nullptr)
                onStep(context, result, *window[kBlocksPerWindow - 1], elapsed);

            queue.pop(); // the window moves on by one hop
            windowsSinceReset_++;
            analysed++;
        }

        return analysed;
    }

    void WatsonCore::onWindow(const Features &f, StepResult &result)
    {
        last_ = f;
        stats_.windows++;

        // Time of the window: start sample -> ms (16 samples per ms).
        const int64_t timeMs =
            static_cast<int64_t>(f.startSample / (kSampleRate / 1000));

        // ---- Fan and compressor: shared median, independent from here ----
        primary_.push(f.compPrimaryDb);
        lower_.push(f.compLowerDb);
        upper_.push(f.compUpperDb);
        mid_.push(f.midDb);
        high_.push(f.highDb);
        highStd_.push(f.highStdDb);
        historyMs_.push(static_cast<float>(f.historyMs));

        const int w = medianWindows();
        smoothed_.compPrimaryDb = primary_.median(w);
        smoothed_.compLowerDb = lower_.median(w);
        smoothed_.compUpperDb = upper_.median(w);
        smoothed_.midDb = mid_.median(w);
        smoothed_.highDb = high_.median(w);
        smoothed_.highStdDb = highStd_.median(w);
        smoothed_.historyMs = historyMs_.median(w);

        const bool fan =
            FanDetector::detect(smoothed_.midDb, smoothed_.highDb,
                                smoothed_.highStdDb, smoothed_.historyMs, params_);
        const bool compressor = CompressorDetector::detect(
            smoothed_.compPrimaryDb, smoothed_.compLowerDb,
            smoothed_.compUpperDb, params_);

        if (fan)
            stats_.fanCandidateWindows++;
        if (compressor)
            stats_.compressorCandidateWindows++;

        result.fanChanged = fanHold_.update(fan, timeMs);
        result.compressorChanged = compressorHold_.update(compressor, timeMs);

        // ---- Beep: raw features, never the median ----
        result.beep = beep_.update(f, timeMs, result.beepEvent);
    }
} // namespace watson
