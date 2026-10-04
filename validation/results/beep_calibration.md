# Beep footprint calibration

21 reviewed beep events; the PC's 1024-point pipeline finds 69 beeps in them. Firmware features (2048-point), PC BeepDetector, different footprints:

| footprint (ms) | PC beeps reproduced | extra beeps | duration diff mean / sd (ms) |
|---:|---:|---:|---:|
| 64 | 69 / 69 | 0 | +10.2 / 21.4 |
| 80 | 63 / 69 | 0 | -1.8 / 15.9 |
| 96 | 57 / 69 | 0 | -20.8 / 15.3 |
| 128 | 44 / 69 | 0 | -50.9 / 15.7 |

64 ms reproduces every beep with no extras; a larger footprint merges beeps that are ~250 ms apart (the refractory outlasts the gap) and under-reads durations. The firmware keeps 64 ms (`kBeepFootprintMs`, src/audio/AudioConfig.h).
