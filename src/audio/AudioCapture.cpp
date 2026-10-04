#include "AudioCapture.h"

#include <esp_timer.h>

namespace watson
{
    namespace
    {
        // INMP441 produces garbage while it settles after the clock starts.
        constexpr int kStartupBlocksToIgnore = 8;
        constexpr uint32_t kReadTimeoutMs = 1000;
    } // namespace

    // Runs in the I2S interrupt when the driver had to drop a DMA buffer
    // because nobody read it in time. Count only.
    bool IRAM_ATTR AudioCapture::onOverflow(i2s_chan_handle_t,
                                            i2s_event_data_t *,
                                            void *user)
    {
        AudioCapture *self = static_cast<AudioCapture *>(user);
        self->overflows_ = self->overflows_ + 1;
        return false;
    }

    bool AudioCapture::initI2s()
    {
        i2s_chan_config_t chan = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
        chan.dma_desc_num = kDmaDescriptors;
        chan.dma_frame_num = kDmaFrames;
        chan.auto_clear = false;

        esp_err_t err = i2s_new_channel(&chan, nullptr, &rx_);
        if (err != ESP_OK)
        {
            Serial.printf("ERROR: i2s_new_channel(): %d\n", err);
            return false;
        }

        // 32-bit Philips slots, left only (INMP441 L/R tied to GND): the
        // same wiring and format the PC tools were built against. The 24-bit
        // sample is left-justified in the word; read() shifts it down.
        i2s_std_config_t cfg = {};
        cfg.clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(kSampleRate);
        cfg.slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
            I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO);
        cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;
        cfg.gpio_cfg.mclk = I2S_GPIO_UNUSED;
        cfg.gpio_cfg.bclk = static_cast<gpio_num_t>(kI2sBclk);
        cfg.gpio_cfg.ws = static_cast<gpio_num_t>(kI2sLrclk);
        cfg.gpio_cfg.dout = I2S_GPIO_UNUSED;
        cfg.gpio_cfg.din = static_cast<gpio_num_t>(kI2sDin);

        err = i2s_channel_init_std_mode(rx_, &cfg);
        if (err != ESP_OK)
        {
            Serial.printf("ERROR: i2s_channel_init_std_mode(): %d\n", err);
            return false;
        }

        // The only place a lost DMA buffer is visible: a blocking read never
        // returns short, the driver just overwrites itself.
        i2s_event_callbacks_t callbacks = {};
        callbacks.on_recv_q_ovf = &AudioCapture::onOverflow;
        err = i2s_channel_register_event_callback(rx_, &callbacks, this);
        if (err != ESP_OK)
        {
            Serial.printf("ERROR: i2s_channel_register_event_callback(): %d\n", err);
            return false;
        }

        err = i2s_channel_enable(rx_);
        if (err != ESP_OK)
        {
            Serial.printf("ERROR: i2s_channel_enable(): %d\n", err);
            return false;
        }

        // Internal pulldown on SD. The INMP441 drives the line only in its own
        // slot and tri-states it otherwise; with the mic unplugged or its SD
        // wire broken the pin would float and pick up noise that looks like
        // audio. Pulled low it reads constant zero, which the flatline
        // detector recognises as "no hardware". Applied after the peripheral
        // is up so the driver's pin setup cannot undo it.
        gpio_set_pull_mode(static_cast<gpio_num_t>(kI2sDin), GPIO_PULLDOWN_ONLY);
        return true;
    }

    bool AudioCapture::begin()
    {
        for (int i = 0; i < kSinks; i++)
            if (sinks_[i].queue == nullptr && i == 0)
                return false; // the analysis sink is mandatory

        if (!initI2s())
            return false;

        BaseType_t ok = xTaskCreatePinnedToCore(
            &AudioCapture::taskTrampoline, "acquire", 2560, this,
            kAcquisitionPriority, &task_, kAcquisitionCore);
        return ok == pdPASS;
    }

    void AudioCapture::taskTrampoline(void *self)
    {
        static_cast<AudioCapture *>(self)->run();
    }

    void AudioCapture::run()
    {
        size_t bytes = 0;

        for (int i = 0; i < kStartupBlocksToIgnore; i++)
            i2s_channel_read(rx_, raw_, sizeof(raw_), &bytes, kReadTimeoutMs);

        uint32_t seenOverflows = overflows_;
        bool pendingGap = false;
        int64_t lastDoneUs = esp_timer_get_time();

        running_ = true;

        for (;;)
        {
            if (maintenance_)
            {
                if (running_)
                {
                    i2s_channel_disable(rx_);
                    running_ = false;
                    stats_.maintenancePauses = stats_.maintenancePauses + 1;
                }
                vTaskDelay(pdMS_TO_TICKS(20));
                continue;
            }
            if (!running_)
            {
                // Back from maintenance: restart the peripheral. What was in
                // the DMA ring is stale, and the audio in between is gone.
                if (i2s_channel_enable(rx_) != ESP_OK)
                {
                    vTaskDelay(pdMS_TO_TICKS(100));
                    continue;
                }
                running_ = true;
                pendingGap = true;
                lastDoneUs = esp_timer_get_time();
                seenOverflows = overflows_;
            }

            bytes = 0;
            const esp_err_t err = i2s_channel_read(rx_, raw_, sizeof(raw_), &bytes,
                                                   kReadTimeoutMs);
            const int64_t nowUs = esp_timer_get_time();

            if (err != ESP_OK || bytes != sizeof(raw_))
            {
                // A partial read leaves the stream out of step with the block
                // grid, so the block is discarded and the loss reported.
                if (err != ESP_OK)
                    stats_.readErrors = stats_.readErrors + 1;
                else
                    stats_.shortReads = stats_.shortReads + 1;
                pendingGap = true;
                lastDoneUs = nowUs;
                continue;
            }

            const uint32_t interval = static_cast<uint32_t>(nowUs - lastDoneUs);
            lastDoneUs = nowUs;
            if (interval > stats_.maxIntervalUs)
            {
                stats_.maxIntervalUs = interval;
                stats_.maxIntervalAtMs = static_cast<uint32_t>(nowUs / 1000);
            }
            if (interval > 2 * 1000UL * kHopMs)
            {
                stats_.lateBlocks = stats_.lateBlocks + 1;
                stats_.lastLateAtMs = static_cast<uint32_t>(nowUs / 1000);
                stats_.lastLateUs = interval;
            }

            const uint32_t overflows = overflows_;
            const bool gap = pendingGap || overflows != seenOverflows;
            if (overflows != seenOverflows)
            {
                stats_.i2sOverruns = stats_.i2sOverruns + (overflows - seenOverflows);
            }
            seenOverflows = overflows;
            pendingGap = false;

            // Right-align the 24-bit sample in the 32-bit word, so nothing
            // downstream needs to know about the I2S padding.
            if (simulateFlat_)
            {
                // Test hook (WATSON SIMFLAT): behave as if the mic were dead.
                for (int i = 0; i < kBlockSamples; i++)
                    block_.samples[i] = 0;
            }
            else
            {
                for (int i = 0; i < kBlockSamples; i++)
                    block_.samples[i] = raw_[i] >> 8;
            }

            // Is a microphone there at all? A flat block is one whose samples
            // barely move. The change of state is reported as a gap too, so
            // the analysis drops what it was building around it.
            if (flatline_.update(block_.samples, kBlockSamples))
            {
                hardwareConnected_ = flatline_.connected();
                stats_.hardwareTransitions = stats_.hardwareTransitions + 1;
                pendingGap = true;
            }
            stats_.flatBlocks = flatline_.flatBlocks();

            block_.seq = seq_++;
            block_.capturedMs = static_cast<uint32_t>(nowUs / 1000);
            block_.i2sGap = gap;
            stats_.blocks = stats_.blocks + 1;

            // Fan out. A queue that cannot take the block loses it; nothing
            // here ever waits on a consumer.
            for (int i = 0; i < kSinks; i++)
            {
                CaptureSink &sink = sinks_[i];
                if (sink.queue == nullptr || !sink.enabled)
                    continue;
                if (sink.queue->tryPush(block_) && sink.task != nullptr)
                    xTaskNotifyGive(sink.task);
            }
        }
    }
} // namespace watson
