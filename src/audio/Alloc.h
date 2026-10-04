#pragma once

#include <stdlib.h>

#if defined(ESP_PLATFORM)
#include <esp_heap_caps.h>
#endif

namespace watson
{
    // One-time allocations (queues, FFT scratch). Never used in the
    // steady-state audio path.
    //
    // allocBuffer: ordinary byte-addressable internal RAM.
    inline void *allocBuffer(size_t bytes)
    {
#if defined(ESP_PLATFORM)
        return heap_caps_malloc(bytes, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
#else
        return malloc(bytes);
#endif
    }

    // allocWords: memory that is only ever read and written as aligned 32-bit
    // words (float and int32 arrays). On the ESP32 that may come from the
    // instruction-RAM heap -- tens of KB that byte-addressable allocations
    // (the Wi-Fi driver, a TLS session) cannot use at all -- which leaves the
    // scarce byte-addressable heap to them. Callers must not use memcpy or
    // sub-word fields on it; see AudioBlock.
    inline void *allocWords(size_t bytes)
    {
#if defined(ESP_PLATFORM)
        void *p = heap_caps_malloc(bytes, MALLOC_CAP_EXEC | MALLOC_CAP_32BIT);
        if (p != nullptr)
            return p;
        return heap_caps_malloc(bytes, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
#else
        return malloc(bytes);
#endif
    }
} // namespace watson
