#pragma once

#include <Arduino.h>
#include <driver/i2s_std.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "AudioBuffer.h"
#include "AudioConfig.h"

namespace watson
{
    // Where acquired blocks go. The consumer owns `task` (set once it exists)
    // and `enabled`; the acquisition task only reads them.
    struct CaptureSink
    {
        BlockQueue *queue = nullptr;
        // Woken after every block that was accepted. Null until the consumer
        // has a task.
        volatile TaskHandle_t task = nullptr;
        // When false the block is not even copied (no TCP client connected).
        volatile bool enabled = true;
    };

    struct CaptureStats
    {
        volatile uint32_t blocks = 0;
        volatile uint32_t i2sOverruns = 0; // DMA buffers lost inside the driver
        volatile uint32_t shortReads = 0;
        volatile uint32_t readErrors = 0;
        // Spacing of completed reads. A block is 32 ms of audio, so anything
        // well above that means the task was held up.
        volatile uint32_t maxIntervalUs = 0;
        volatile uint32_t maxIntervalAtMs = 0; // uptime when it happened
        volatile uint32_t lateBlocks = 0;      // intervals above 2 hops
        volatile uint32_t lastLateAtMs = 0;
        volatile uint32_t lastLateUs = 0;
        volatile uint32_t maintenancePauses = 0; // updates that paused acquisition
    };

    // Owns the I2S peripheral and the acquisition task, and nothing else: it
    // knows nothing about FFTs, detectors, NightMare or sockets. Everything it
    // produces leaves through the sinks' queues, and a full queue never makes
    // it wait.
    class AudioCapture
    {
    public:
        static constexpr int kSinks = 2;

        CaptureSink &sink(int index) { return sinks_[index]; }

        // Starts the peripheral and the task. The queues must already exist.
        bool begin();

        const CaptureStats &stats() const { return stats_; }

        // Maintenance mode: a firmware update is running. Writing flash
        // disables the instruction cache for tens of milliseconds at a time, and
        // the I2S interrupt handler and this task both run from it, so audio is
        // lost in bursts no scheduling can prevent on this chip -- and an
        // update has no use for it. Acquisition is paused: the peripheral is
        // stopped and the task idles. Leaving maintenance mode (an update that
        // failed) restarts it and reports a gap, so the analysis drops what it
        // was building and the published observations stay.
        void setMaintenance(bool active) { maintenance_ = active; }
        bool inMaintenance() const { return maintenance_; }
        uint32_t blocksSeen() const { return seq_; }
        uint32_t stackFreeBytes() const
        {
            return task_ ? uxTaskGetStackHighWaterMark(task_) : 0;
        }

    private:
        static void taskTrampoline(void *self);
        static bool IRAM_ATTR onOverflow(i2s_chan_handle_t handle,
                                         i2s_event_data_t *event, void *user);
        bool initI2s();
        void run();

        i2s_chan_handle_t rx_ = nullptr;
        TaskHandle_t task_ = nullptr;
        CaptureSink sinks_[kSinks];
        CaptureStats stats_;
        volatile uint32_t overflows_ = 0;
        volatile bool maintenance_ = false;
        bool running_ = false;
        uint32_t seq_ = 0;

        int32_t raw_[kBlockSamples];
        AudioBlock block_;
    };
} // namespace watson
