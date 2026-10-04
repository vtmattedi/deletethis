"""Compute (and cache) the firmware feature matrix for every labelled WAV."""
from __future__ import annotations
import pickle, time
from pathlib import Path
import numpy as np
import watson_ref as ref
import dataset as ds
import evaluation

CACHE = Path(__file__).resolve().parent / ".cache"
NAMES = None

def windows_for(path: Path):
    """(times, matrix, names) of every firmware analysis window of a WAV."""
    CACHE.mkdir(exist_ok=True)
    key = CACHE / (path.stem + ".pkl")
    if key.exists() and key.stat().st_mtime > path.stat().st_mtime and key.stat().st_mtime > Path(ref.__file__).stat().st_mtime:
        return pickle.load(open(key, "rb"))
    samples = ref.load_wav(path)
    ex = ref.RefExtractor()
    times, rows, names = [], [], None
    for start in range(0, samples.size - ref.HOP + 1, ref.HOP):
        for w in ex.push(samples[start:start + ref.HOP]):
            v = w.values
            if names is None: names = list(v)
            times.append(w.time); rows.append([v[n] for n in names])
    out = (np.array(times), np.array(rows), names)
    pickle.dump(out, open(key, "wb"))
    return out

def rolling_median(col: np.ndarray, width: int) -> np.ndarray:
    out = np.empty_like(col)
    for i in range(col.size):
        out[i] = np.median(col[max(0, i - width + 1): i + 1])
    return out

def all_segments():
    """Yield Segment objects with per-window truth and feature matrices."""
    segs = []
    for name, state, interf, path in ds.recording_segments():
        t, m, n = windows_for(path)
        fan, comp = ds.truth_columns(t, state)
        segs.append(dict(name=name, group="rec:" + name[:15], kind="recording", interference=interf, times=t, m=m, names=n, fan=fan, comp=comp, state=state))
    v1 = ds.v1_records()
    groups = evaluation.group_events(v1)
    for r in v1:
        t, m, n = windows_for(r.wav_path)
        truth = evaluation.truth_labels(r, t)
        obs = evaluation.observation_truth(truth)
        segs.append(dict(name=r.id, group=f"v1:{groups[r.id]}", kind="v1", interference=r.interference_label, times=t, m=m, names=n, fan=obs["fan"], comp=obs["compressor"], state=f"{r.actual_from}->{r.actual_to}"))
    return segs

if __name__ == "__main__":
    t0 = time.time()
    s = all_segments()
    print(len(s), "segments", time.time() - t0, "s")
