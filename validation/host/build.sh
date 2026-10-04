#!/usr/bin/env bash
# Builds the host replay tool in a gcc container: the firmware's own analysis
# sources plus the real arduinoFFT, no Arduino/ESP-IDF involved.
#   validation/host/build.sh        -> validation/host/replay (Linux ELF)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
fft="$root/.pio/libdeps/esp32dev/arduinoFFT/src"
[ -d "$fft" ] || { echo "run 'pio pkg install' first (arduinoFFT missing)"; exit 1; }
MSYS_NO_PATHCONV=1 docker run --rm -v "$root:/work" -w /work/validation/host gcc:13 \
  g++ -std=c++17 -O2 -Wall -Wno-unused-function -ffp-contract=off -DCOMPLEX_INPUT \
      -I/work/.pio/libdeps/esp32dev/arduinoFFT/src \
      replay.cpp \
      ../../src/audio/FeatureExtractor.cpp ../../src/audio/WatsonCore.cpp \
      ../../src/audio/BeepDetector.cpp ../../src/audio/Fft.cpp \
      ../../src/audio/SelfTest.cpp \
      /work/.pio/libdeps/esp32dev/arduinoFFT/src/arduinoFFT.cpp \
      -o replay -lm
echo built "$here/replay"
