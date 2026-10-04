#include "RawAudioTcp.h"

#if ENABLE_TCP

#include <Network/WiFiRadio/NmWifiRadio.h>
#include <errno.h>
#include <esp_heap_caps.h>
#include <esp_netif.h>
#include <fcntl.h>
#include <lwip/netdb.h>
#include <lwip/sockets.h>
#include <string.h>

#include "../audio/Alloc.h"

namespace watson
{
    namespace
    {
        constexpr uint16_t kProtocolVersion = 1;
        constexpr uint16_t kFormatPcmS32le = 1;
        constexpr uint16_t kBitsValid = 24;
        // A write that makes no progress for this long means the peer is dead.
        // It is long on purpose: one lost Wi-Fi packet stalls TCP for a
        // retransmission timeout (a second or more), which is not a dead
        // client. Nothing waits on this -- while the write is stalled the
        // queue fills and blocks are dropped (and reported in `dropped`),
        // acquisition carries on.
        constexpr uint32_t kWriteTimeoutMs = 8000;
        constexpr uint32_t kStreamStackBytes = 3584;
        // Free internal RAM that must remain after a client's allocations.
        constexpr size_t kTcpHeapReserve = 24 * 1024;
        // If a connected client drives free internal RAM below this (lwIP
        // queues unsent data when the peer or the link is slow), it is dropped:
        // the debug stream is never allowed to starve NightMare's own session.
        constexpr size_t kTcpHeapFloor = 9 * 1024;

        struct __attribute__((packed)) StreamHeader
        {
            char magic[4];
            uint16_t version;
            uint16_t headerSize;
            uint32_t sampleRate;
            uint16_t format;
            uint16_t channels;
            uint16_t bitsValid;
            uint16_t frameSamples;
            uint32_t reserved0;
            uint32_t reserved1;
            uint32_t reserved2;
        };
        static_assert(sizeof(StreamHeader) == 32, "StreamHeader must be 32 bytes");

        struct __attribute__((packed)) FrameHeader
        {
            char magic[4];
            uint32_t sequence;
            uint16_t samples;
            uint16_t dropped;
        };
        static_assert(sizeof(FrameHeader) == 12, "FrameHeader must be 12 bytes");

        void setNonBlocking(int fd)
        {
            const int flags = fcntl(fd, F_GETFL, 0);
            fcntl(fd, F_SETFL, flags | O_NONBLOCK);
        }

        void tellAndClose(int fd, const char *message)
        {
            send(fd, message, strlen(message), MSG_DONTWAIT);
            close(fd);
        }
    } // namespace

    bool RawAudioTcp::begin(CaptureSink *sink, uint16_t port)
    {
        sink_ = sink;
        port_ = port;
        if (sink_ == nullptr)
            return false;

        sink_->enabled = false; // nobody is connected yet
        sink_->queue = nullptr;
        return true;
    }

    // The IP link is the Wi-Fi station's, owned by NightMareNetwork; ask the
    // interface rather than any Arduino WiFi object.
    bool RawAudioTcp::linkUp()
    {
        esp_netif_t *netif = WiFiRadio_stationNetif();
        if (netif == nullptr)
            return false;
        esp_netif_ip_info_t ip;
        return esp_netif_get_ip_info(netif, &ip) == ESP_OK && ip.ip.addr != 0;
    }

    bool RawAudioTcp::startListening()
    {
        const int fd = socket(AF_INET, SOCK_STREAM, IPPROTO_IP);
        if (fd < 0)
            return false;

        int reuse = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

        sockaddr_in addr = {};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
        addr.sin_port = htons(port_);

        if (bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0 ||
            listen(fd, 1) != 0)
        {
            close(fd);
            return false;
        }

        setNonBlocking(fd);
        listenFd_ = fd;
        Serial.printf("# tcp: raw PCM server on port %u\n", port_);
        return true;
    }

    // loop() context. Cheap when idle: one non-blocking accept().
    void RawAudioTcp::poll()
    {
        if (sink_ == nullptr || !linkUp())
            return;

        if (listenFd_ < 0 && !startListening())
            return;

        const int fd = accept(listenFd_, nullptr, nullptr);
        if (fd < 0)
            return;

        // One client at a time; anyone else is turned away at once rather
        // than left in the backlog.
        if (streaming_)
        {
            stats_.refused = stats_.refused + 1;
            tellAndClose(fd, "busy\r\n");
            return;
        }

        // The debug stream must never take the memory NightMare's own
        // connection (a TLS session is ~40 KB of internal RAM) needs: refuse
        // unless a comfortable reserve would remain after this client.
        // (The queue itself sits in 32-bit-only memory that NightMare cannot
        // use, so only the frame buffer and the task stack count here.)
        const size_t needed = sizeof(FrameHeader) + sizeof(int32_t) * kBlockSamples +
                              kStreamStackBytes;
        if (heap_caps_get_free_size(MALLOC_CAP_8BIT) < needed + kTcpHeapReserve)
        {
            stats_.refused = stats_.refused + 1;
            tellAndClose(fd, "low memory\r\n");
            return;
        }

        // Everything this client needs is allocated now and returned when it
        // leaves.
        frame_ = static_cast<uint8_t *>(
            allocBuffer(sizeof(FrameHeader) + sizeof(int32_t) * kBlockSamples));
        if (frame_ == nullptr || !queue_.begin(kTcpQueueBlocks))
        {
            free(frame_);
            frame_ = nullptr;
            stats_.refused = stats_.refused + 1;
            tellAndClose(fd, "no memory\r\n");
            return;
        }

        int one = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        setNonBlocking(fd);

        clientFd_ = fd;
        sequence_ = 0;
        streaming_ = true;

        if (xTaskCreatePinnedToCore(&RawAudioTcp::streamTrampoline, "tcp-pcm",
                                    kStreamStackBytes, this, kTcpPriority, nullptr,
                                    kTcpCore) != pdPASS)
        {
            close(clientFd_);
            clientFd_ = -1;
            queue_.end();
            free(frame_);
            frame_ = nullptr;
            streaming_ = false;
            stats_.refused = stats_.refused + 1;
        }
    }

    void RawAudioTcp::streamTrampoline(void *self)
    {
        static_cast<RawAudioTcp *>(self)->stream();
        vTaskDelete(nullptr);
    }

    // Writes everything or reports failure. A short write is normal over TCP;
    // silently dropping the remainder would corrupt the stream in a way the PC
    // could only see as a lost frame much later.
    bool RawAudioTcp::sendAll(const void *data, size_t size)
    {
        const uint8_t *bytes = static_cast<const uint8_t *>(data);
        uint32_t lastProgress = millis();

        while (size > 0)
        {
            const ssize_t n = send(clientFd_, bytes, size, MSG_DONTWAIT);
            if (n > 0)
            {
                bytes += n;
                size -= n;
                lastProgress = millis();
                continue;
            }

            if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
                sessionWaits_++;

            if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK)
            {
                Serial.printf("# tcp: send() failed, errno %d\n", errno);
                return false;
            }

            if (millis() - lastProgress > kWriteTimeoutMs)
            {
                stats_.writeTimeouts = stats_.writeTimeouts + 1;
                Serial.println("# tcp: send() made no progress for 8 s");
                return false;
            }

            fd_set writable;
            FD_ZERO(&writable);
            FD_SET(clientFd_, &writable);
            timeval wait = {0, 50 * 1000};
            select(clientFd_ + 1, nullptr, &writable, nullptr, &wait);
        }
        return true;
    }

    bool RawAudioTcp::sendStreamHeader()
    {
        StreamHeader header = {};
        memcpy(header.magic, "ACD1", 4);
        header.version = kProtocolVersion;
        header.headerSize = sizeof(StreamHeader);
        header.sampleRate = kSampleRate;
        header.format = kFormatPcmS32le;
        header.channels = 1;
        header.bitsValid = kBitsValid;
        header.frameSamples = kBlockSamples;
        return sendAll(&header, sizeof(header));
    }

    bool RawAudioTcp::sendBlock(const AudioBlock &block)
    {
        // `dropped` counts whole frames that never made it: blocks this queue
        // refused, and one for an I2S overrun (audio the driver lost).
        uint32_t dropped = block.lostBefore + (block.i2sGap ? 1u : 0u);
        if (dropped > 0xFFFF)
            dropped = 0xFFFF;

        FrameHeader header = {};
        memcpy(header.magic, "ACDF", 4);
        header.sequence = sequence_++;
        header.samples = kBlockSamples;
        header.dropped = static_cast<uint16_t>(dropped);

        memcpy(frame_, &header, sizeof(header));
        // Word-wise: the block may be in 32-bit-only memory.
        int32_t *payload = reinterpret_cast<int32_t *>(frame_ + sizeof(header));
        for (int i = 0; i < kBlockSamples; i++)
            payload[i] = block.samples[i];

        return sendAll(frame_, sizeof(header) + sizeof(int32_t) * kBlockSamples);
    }

    // Everything the client owned goes back, in the order that keeps the
    // acquisition task safe: stop feeding the queue, let an in-flight push
    // (at most one block time) finish, then free it.
    void RawAudioTcp::finishClient(const char *why, bool abort)
    {
        if (clientFd_ >= 0)
        {
            if (abort)
            {
                // The peer is gone or not reading. A plain close() would keep
                // the unsent data -- several KB of internal RAM -- queued in
                // lwIP until its retransmission timers give up, which starves
                // the next client and NightMare's own connection. Zero linger
                // resets the connection and frees it now.
                linger reset = {1, 0};
                setsockopt(clientFd_, SOL_SOCKET, SO_LINGER, &reset, sizeof(reset));
            }
            close(clientFd_);
            clientFd_ = -1;
        }

        sink_->enabled = false;
        vTaskDelay(pdMS_TO_TICKS(2 * kHopMs + 10));
        sink_->task = nullptr;
        sink_->queue = nullptr;
        queue_.end();
        free(frame_);
        frame_ = nullptr;

        stats_.connected = false;
        stats_.clientsLost = stats_.clientsLost + 1;
        Serial.printf("# tcp: client gone (%s) after %lu ms: sent %lu frames, "
                      "queue refused %lu, send-would-block %lu\n",
                      why, (unsigned long)(millis() - sessionStartMs_),
                      (unsigned long)(stats_.framesSent - sessionFramesAtStart_),
                      (unsigned long)queue_.droppedTotal(),
                      (unsigned long)sessionWaits_);
        streaming_ = false;
    }

    void RawAudioTcp::stream()
    {
        stats_.clients = stats_.clients + 1;
        stats_.connected = true;
        sessionStartMs_ = millis();
        sessionFramesAtStart_ = stats_.framesSent;
        sessionWaits_ = 0;
        Serial.println("# tcp: client connected, streaming");

        if (!sendStreamHeader())
        {
            finishClient("header write failed", true);
            return;
        }

        sink_->task = xTaskGetCurrentTaskHandle();
        sink_->queue = &queue_;
        sink_->enabled = true;

        uint32_t lastLinkCheck = millis();

        for (;;)
        {
            // The client never sends, so a readable socket is a closed one.
            char probe;
            const ssize_t r = recv(clientFd_, &probe, 1, MSG_DONTWAIT);
            if (r == 0 || (r < 0 && errno != EAGAIN && errno != EWOULDBLOCK))
            {
                finishClient("disconnected", false);
                return;
            }

            if (heap_caps_get_free_size(MALLOC_CAP_8BIT) < kTcpHeapFloor)
            {
                finishClient("low memory", true);
                return;
            }

            if (millis() - lastLinkCheck > 1000)
            {
                lastLinkCheck = millis();
                if (!linkUp())
                {
                    finishClient("link down", true);
                    return;
                }
            }

            const AudioBlock *block = queue_.peek();
            if (block == nullptr)
            {
                ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(100));
                continue;
            }

            const bool ok = sendBlock(*block);
            queue_.pop();
            stats_.blocksDropped = queue_.droppedTotal();

            if (!ok)
            {
                finishClient("write failed", true);
                return;
            }
            stats_.framesSent = stats_.framesSent + 1;
        }
    }
} // namespace watson

#endif // ENABLE_TCP
