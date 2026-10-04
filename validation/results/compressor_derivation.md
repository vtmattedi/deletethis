# Compressor threshold derivation (52-65 Hz primary band)

117 labelled files, 75371 scored windows (27830 compressor, 47541 not), 87 time-groups. Feature: dB of the summed 52-65 Hz power, 2048-point Hamming, median over 0.5 s.

## Distributions (dBFS, median-smoothed)

| feature | compressor ON p5 / p50 / p95 | OFF p50 / p95 / p99 |
|---|---|---|
| 52-65 Hz (primary) | -40.7 / -32.6 / -30.1 | -59.1 / -53.1 / -43.0 |
| 30-80 Hz (old) | -40.4 / -32.4 / -29.9 | -50.8 / -43.2 / -34.8 |
| 30-45 Hz (lower sideband) | -63.9 / -53.6 / -47.3 | -53.8 / -47.9 / -43.0 |
| 65-80 Hz (upper sideband) | -60.7 / -52.9 / -48.6 | -60.2 / -49.9 / -42.4 |

## Threshold sweep on the primary band

| threshold dB | balanced accuracy | recall | false-positive |
|---:|---:|---:|---:|
| -52 | 0.9657 | 0.9660 | 0.0347 |
| -51 | 0.9698 | 0.9644 | 0.0248 |
| -50 | 0.9715 | 0.9634 | 0.0204 |
| -49 | 0.9725 | 0.9631 | 0.0180 |
| -48 | 0.9733 | 0.9630 | 0.0164 |
| -47 | 0.9738 | 0.9629 | 0.0152 |
| -46 | 0.9744 | 0.9629 | 0.0141 |
| -45 | 0.9745 | 0.9629 | 0.0138 |
| -44 | 0.9749 | 0.9628 | 0.0129 |
| -43 | 0.9759 | 0.9621 | 0.0102 |
| -42 | 0.9760 | 0.9610 | 0.0091 |
| -41 | 0.9749 | 0.9590 | 0.0091 |
| -40 | 0.9608 | 0.9306 | 0.0091 |
| -39 | 0.9190 | 0.8470 | 0.0091 |
| -38 | 0.8533 | 0.7157 | 0.0091 |
| -37 | 0.8339 | 0.6770 | 0.0091 |
| -36 | 0.8337 | 0.6764 | 0.0091 |
| -35 | 0.8337 | 0.6764 | 0.0091 |

## Choice

* best balanced accuracy at -42.5 dB (0.9761)
* plateau (within 0.5 % of best): -50.0 ... -41.0 dB
* leave-one-group-out (87 groups): chosen -44.0 ... -42.0 dB, median -42.5; held-out balanced accuracy 0.9743
* **chosen: -43.0 dB** (the best, rounded down to a whole dB, away from the recall cliff above -41 dB)
* at -43.0 dB: recall 0.9621, false-positive 0.0102
* old rule for comparison, 30-80 Hz at -38 dB: balanced accuracy 0.8637

## By interference at the chosen threshold

| interference | ON windows | OFF windows | recall | false-positive |
|---|---:|---:|---:|---:|
| clean | 18223 | 25742 | 0.971 | 0.001 |
| other | 0 | 4668 | - | 0.053 |
| printer | 6596 | 10537 | 0.971 | 0.002 |
| talking | 2331 | 4662 | 1.000 | 0.000 |
| talking+tv | 0 | 1556 | - | 0.000 |
| tv | 680 | 376 | 0.500 | 0.500 |

## Sideband deltas for the future guard (primary - sideband, dB)

Calibration data only: the guard ships disabled with margins `0,0`.

| windows | primary-lower p1 / p5 / p50 | primary-upper p1 / p5 / p50 |
|---|---|---|
| true compressor, detected (26774) | 8.4 / 15.7 / 21.4 | 7.2 / 17.2 / 20.6 |
| not compressor, detected (false positives) (483) | -1.4 / 3.1 / 21.2 | -5.4 / -4.6 / 20.4 |
