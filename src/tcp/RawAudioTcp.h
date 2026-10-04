#pragma once
#define ENABLE_TCP 1
// Raw PCM debug server. Build it out with -DENABLE_TCP=0 (platformio.ini
// build_flags) or by changing the default below; nothing else in the firmware
// refers to it when it is off.
#ifndef ENABLE_TCP
#define ENABLE_TCP 1
#endif

#if ENABLE_TCP

#include <Arduino.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "../audio/AudioBuffer.h"
#include "../audio/AudioCapture.h"

#ifndef AUDIO_TCP_PORT
#define AUDIO_TCP_PORT 3333
#endif

namespace watson
{
    // Wire contract (what tools/audio/acstream.py reads), little-endian:
    //
    //   once on connect   "ACD1" header, 32 bytes: version 1, 16 kHz, format 1
    //                     (signed 32-bit LE, 24 valid bits, right-aligned),
    //                     mono, 512 samples per frame
    //   then, repeating   "ACDF" frame header, 12 bytes: sequence, samples,
    //                     dropped (frames lost since the previous one),
    //                     followed by 512 x int32 PCM
    //
    // One client, TCP port 3333 by default, no audio/control multiplexing: the
    // server never reads from the client. TCP is for debugging and
    // calibration only.
    //
    // Everything it needs lives only while a client is connected: its queue,
    // the frame buffer and the streaming task (about 18 KB of internal RAM,
    // which Wi-Fi and the TLS session also want). An idle debug server costs
    // the production firmware a listening socket and nothing else.
    //
    // It consumes from its own queue on its own task. If the client or the
    // network cannot keep up the queue fills and blocks are *dropped* (and
    // reported in the next frame's `dropped`); acquisition never notices.
    struct TcpStats
    {
        volatile uint32_t clients = 0;
        volatile uint32_t framesSent = 0;
        volatile uint32_t blocksDropped = 0; // never sent, counted in `dropped`
        volatile uint32_t writeTimeouts = 0;
        volatile uint32_t clientsLost = 0;
        volatile uint32_t refused = 0; // second client, or no memory
        volatile bool connected = false;
    };

    class RawAudioTcp
    {
    public:
        // `sink` is the acquisition-side handle this module controls: the
        // queue it points at, the task to wake, and whether anyone listens.
        bool begin(CaptureSink *sink, uint16_t port);

        // loop() context, non-blocking: opens the listening socket once the
        // Wi-Fi link is up, and starts a streaming task for a new client.
        void poll();

        const TcpStats &stats() const { return stats_; }
        uint16_t port() const { return port_; }

    private:
        static void streamTrampoline(void *self);
        void stream();
        bool linkUp();
        bool startListening();
        void finishClient(const char *why, bool abort);
        bool sendAll(const void *data, size_t size);
        bool sendStreamHeader();
        bool sendBlock(const AudioBlock &block);

        BlockQueue queue_;
        CaptureSink *sink_ = nullptr;
        uint16_t port_ = 0;
        int listenFd_ = -1;
        int clientFd_ = -1;
        volatile bool streaming_ = false;
        uint32_t sequence_ = 0;
        uint32_t sessionStartMs_ = 0;
        uint32_t sessionFramesAtStart_ = 0;
        uint32_t sessionWaits_ = 0;
        TcpStats stats_;
        // 12-byte frame header + one block, sent in a single write.
        uint8_t *frame_ = nullptr;
    };
} // namespace watson

#endif // ENABLE_TCP
