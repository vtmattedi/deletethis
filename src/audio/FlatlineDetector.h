#pragma once

#include <stdint.h>

#include "AudioConfig.h"

namespace watson
{
    // Is a microphone actually delivering audio?
    //
    // An INMP441 that is unplugged, unpowered or has a broken SD wire gives a
    // constant: all zeros with the internal pulldown on the data pin, or a
    // stuck level. A working one always shows noise, however quiet the room
    // (its floor is hundreds of LSB at 24 bits). So a block whose samples span
    // no more than a few LSB is "flat", and a second's worth of flat blocks
    // in a row is a flatline: the hardware is disconnected. A short run of live
    // blocks is enough to say it is back.
    //
    // No Arduino, no FreeRTOS: the host tests drive it too.
    class FlatlineDetector
    {
    public:
        // Peak-to-peak span (24-bit LSB) at or below which a block is flat.
        static constexpr int32_t kFlatSpan = 8;
        // 32 blocks = 1.024 s flat -> disconnected; 16 blocks = 0.5 s live -> back.
        static constexpr uint32_t kBlocksToDisconnect = 32;
        static constexpr uint32_t kBlocksToReconnect = 16;

        static bool blockIsFlat(const int32_t *samples, int count)
        {
            int32_t lo = samples[0], hi = samples[0];
            for (int i = 1; i < count; i++)
            {
                const int32_t v = samples[i];
                if (v < lo)
                    lo = v;
                if (v > hi)
                    hi = v;
            }
            return hi - lo <= kFlatSpan;
        }

        // Feeds one block; true when the connected state just changed.
        bool update(const int32_t *samples, int count)
        {
            if (blockIsFlat(samples, count))
            {
                liveRun_ = 0;
                flatBlocks_++;
                if (flatRun_ < kBlocksToDisconnect)
                    flatRun_++;
                if (connected_ && flatRun_ >= kBlocksToDisconnect)
                {
                    connected_ = false;
                    transitions_++;
                    return true;
                }
            }
            else
            {
                flatRun_ = 0;
                if (liveRun_ < kBlocksToReconnect)
                    liveRun_++;
                if (!connected_ && liveRun_ >= kBlocksToReconnect)
                {
                    connected_ = true;
                    transitions_++;
                    return true;
                }
            }
            return false;
        }

        bool connected() const { return connected_; }
        uint32_t flatBlocks() const { return flatBlocks_; }
        uint32_t transitions() const { return transitions_; }

    private:
        bool connected_ = true;
        uint32_t flatRun_ = 0;
        uint32_t liveRun_ = 0;
        uint32_t flatBlocks_ = 0;
        uint32_t transitions_ = 0;
    };
} // namespace watson
