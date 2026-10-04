# Labelled replay: firmware vs PC v2

117 labelled files (recordings + reviewed v1 events). Per-window, on windows where the detector has published.

| observation | category | detector | windows | recall | false-positive |
|---|---|---|---:|---:|---:|
| fan | all | firmware | 75344 | 0.968 | 0.066 |
| fan | all | pc_v2 | 75576 | 0.967 | 0.058 |
| fan | clean | firmware | 43965 | 0.994 | 0.031 |
| fan | clean | pc_v2 | 44115 | 0.994 | 0.033 |
| fan | talking | firmware | 6966 | 0.931 | 0.000 |
| fan | talking | pc_v2 | 6982 | 0.930 | 0.000 |
| fan | printer | firmware | 17133 | 0.981 | 0.021 |
| fan | printer | pc_v2 | 17179 | 0.981 | 0.021 |
| fan | other | firmware | 7280 | 0.000 | 0.200 |
| fan | other | pc_v2 | 7300 | 0.000 | 0.160 |
| compressor | all | firmware | 75371 | 0.961 | 0.012 |
| compressor | all | pc_v2 | 75299 | 0.769 | 0.012 |
| compressor | clean | firmware | 43965 | 0.969 | 0.000 |
| compressor | clean | pc_v2 | 44079 | 0.909 | 0.000 |
| compressor | talking | firmware | 6993 | 1.000 | 0.000 |
| compressor | talking | pc_v2 | 6741 | 0.479 | 0.000 |
| compressor | printer | firmware | 17133 | 0.971 | 0.005 |
| compressor | printer | pc_v2 | 17179 | 0.500 | 0.005 |
| compressor | other | firmware | 7280 | 0.500 | 0.075 |
| compressor | other | pc_v2 | 7300 | 0.500 | 0.075 |
