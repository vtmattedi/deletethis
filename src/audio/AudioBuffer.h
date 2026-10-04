#pragma once

#include <atomic>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "Alloc.h"
#include "AudioConfig.h"


namespace watson
{
    // One acquisition block: exactly one hop of audio.
    struct AudioBlock
    {
        // Global acquisition counter. Contiguous while nothing was lost.
        uint32_t seq;
        // millis() when the block's last sample was read.
        uint32_t capturedMs;
        // Blocks the *queue* refused since the previous block it accepted.
        // Set per queue, so each consumer sees its own losses.
        uint32_t lostBefore;
        // Non-zero when the I2S DMA overran before this block: audio is
        // missing somewhere in or just before it.
        // (Every field is a 32-bit word: queue slots may live in memory that
        // only allows 32-bit access.)
        uint32_t i2sGap;
        // Right-aligned 24-bit samples.
        int32_t samples[kBlockSamples];
    };

    // Single-producer / single-consumer queue of AudioBlocks.
    //
    // The producer (the acquisition task) can never wait on it: when it is
    // full the block is *refused*, counted, and reported to the consumer on
    // the next block that does get in. That is the whole back-pressure policy
    // -- a slow consumer loses audio, acquisition never slows down.
    class BlockQueue
    {
    public:
        ~BlockQueue() { free(slots_); }

        // Allocates once. Never called from the audio path.
        bool begin(int capacity)
        {
            if (slots_ != nullptr)
                return false;
            slots_ = static_cast<AudioBlock *>(
                allocWords(sizeof(AudioBlock) * capacity));
            if (slots_ == nullptr)
                return false;
            capacity_ = capacity;
            head_.store(0, std::memory_order_relaxed);
            tail_.store(0, std::memory_order_relaxed);
            droppedTotal_.store(0, std::memory_order_relaxed);
            pendingLost_ = 0;
            return true;
        }

        // Releases the slots. The caller guarantees the producer is not
        // pushing (the sink was disabled and a block time has passed).
        void end()
        {
            free(slots_);
            slots_ = nullptr;
            capacity_ = 0;
            head_.store(0, std::memory_order_relaxed);
            tail_.store(0, std::memory_order_relaxed);
            pendingLost_ = 0;
        }

        bool allocated() const { return slots_ != nullptr; }

        // Producer side. Copies the block in; false means it was dropped.
        bool tryPush(const AudioBlock &block)
        {
            const uint32_t head = head_.load(std::memory_order_relaxed);
            const uint32_t tail = tail_.load(std::memory_order_acquire);

            if (head - tail >= static_cast<uint32_t>(capacity_))
            {
                if (pendingLost_ < 0xFFFF)
                    pendingLost_++;
                droppedTotal_.fetch_add(1, std::memory_order_relaxed);
                return false;
            }

            // Word-wise on purpose: the slots may be in 32-bit-only memory,
            // and a volatile destination keeps the compiler from turning
            // this back into a (byte-wise) memcpy.
            AudioBlock &slot = slots_[head % capacity_];
            const uint32_t *from = reinterpret_cast<const uint32_t *>(&block);
            volatile uint32_t *to = reinterpret_cast<volatile uint32_t *>(&slot);
            for (size_t i = 0; i < sizeof(AudioBlock) / sizeof(uint32_t); i++)
                to[i] = from[i];
            slot.lostBefore = pendingLost_;
            pendingLost_ = 0;

            head_.store(head + 1, std::memory_order_release);
            return true;
        }

        // Consumer side. The pointer stays valid until pop().
        const AudioBlock *peek() const
        {
            const uint32_t tail = tail_.load(std::memory_order_relaxed);
            const uint32_t head = head_.load(std::memory_order_acquire);
            return head == tail ? nullptr : &slots_[tail % capacity_];
        }

        // The i-th oldest queued block (0 = the one peek() returns), or null.
        // Lets the consumer read a sliding window across several blocks in
        // place, without copying.
        const AudioBlock *at(uint32_t i) const
        {
            const uint32_t tail = tail_.load(std::memory_order_relaxed);
            const uint32_t head = head_.load(std::memory_order_acquire);
            return head - tail > i ? &slots_[(tail + i) % capacity_] : nullptr;
        }

        void pop()
        {
            tail_.store(tail_.load(std::memory_order_relaxed) + 1,
                        std::memory_order_release);
        }

        // Consumer side: throw away what is queued (a new client connects).
        void clear()
        {
            tail_.store(head_.load(std::memory_order_acquire),
                        std::memory_order_release);
        }

        uint32_t size() const
        {
            return head_.load(std::memory_order_acquire) -
                   tail_.load(std::memory_order_acquire);
        }

        int capacity() const { return capacity_; }
        uint32_t droppedTotal() const
        {
            return droppedTotal_.load(std::memory_order_relaxed);
        }

    private:
        AudioBlock *slots_ = nullptr;
        int capacity_ = 0;
        std::atomic<uint32_t> head_{0};
        std::atomic<uint32_t> tail_{0};
        std::atomic<uint32_t> droppedTotal_{0};
        uint32_t pendingLost_ = 0; // producer-only
    };
} // namespace watson
