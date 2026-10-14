"""ASAP three-channel conditions: normalized BPM, MIDI attacks, annotated beats.

Uses the existing WAV/NPZ pairs without modifying them. Python 3.8 compatible.
"""
import math
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

SAMPLE_RATE = 8000
SAMPLES = 65536
LATENT_RATE = 500
LATENT_LENGTH = 4096
ONSET_SIGMA_MS = 10.0
BEAT_SIGMA_MS = 20.0
CONDITION_SCHEMA = "bpm_onset10ms_beat20ms_maxgaussian_v1"


def checked_times(values, name, duration, minimum=0):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < minimum or not np.isfinite(values).all():
        raise ValueError("Invalid " + name)
    if np.any(values < 0) or np.any(values >= duration) or np.any(np.diff(values) <= 0):
        raise ValueError(name + " must be unique, increasing, and inside the clip.")
    return values


def event_envelope(events, length, sigma_ms):
    envelope = np.zeros(length, dtype=np.float32)
    sigma = sigma_ms * .001 * LATENT_RATE
    for seconds in events:
        centre = seconds * LATENT_RATE
        left = max(0, int(math.floor(centre - 4 * sigma)))
        right = min(length, int(math.ceil(centre + 4 * sigma)) + 1)
        index = np.arange(left, right)
        pulse = np.exp(-.5 * ((index - centre) / sigma) ** 2)
        envelope[left:right] = np.maximum(envelope[left:right], pulse)
    return envelope


def make_condition(onsets, beats, bpm, length=LATENT_LENGTH):
    bpm = float(bpm)
    if not math.isfinite(bpm) or not 60 <= bpm <= 180 or length < 1:
        raise ValueError("Expected BPM in [60,180] and a positive latent length.")
    duration = length / float(LATENT_RATE)
    onsets = checked_times(onsets, "onset_seconds", duration)
    beats = checked_times(beats, "beat_seconds", duration, minimum=2)
    if not np.allclose(np.diff(beats), 60.0 / bpm, atol=2.0 / SAMPLE_RATE, rtol=0):
        raise ValueError("Saved beat spacing does not match target BPM.")
    # Require the recorded beat grid to cover the clip, allowing one period at edges.
    if beats[0] > 60.0 / bpm + 2.0 / SAMPLE_RATE or duration - beats[-1] > 60.0 / bpm + 2.0 / SAMPLE_RATE:
        raise ValueError("Beat grid does not cover the conditioning interval.")
    condition = np.empty((3, length), dtype=np.float32)
    condition[0] = (bpm - 120.0) / 60.0
    condition[1] = event_envelope(onsets, length, ONSET_SIGMA_MS)
    condition[2] = event_envelope(beats, length, BEAT_SIGMA_MS)
    return condition


def load_condition(path):
    with np.load(str(path), allow_pickle=False) as data:
        bpm = float(data["target_bpm"])
        old = np.asarray(data["condition"], dtype=np.float32)
        condition = make_condition(data["onset_seconds"], data["beat_seconds"], bpm)
    if old.shape != (2, LATENT_LENGTH) or not np.isfinite(old).all():
        raise ValueError("Expected the original two-channel condition in " + str(path))
    if not np.allclose(old, condition[:2], atol=2e-6, rtol=0):
        raise ValueError("Saved BPM/onset channels disagree with timestamps in " + str(path))
    # Preserve the original two channels bit for bit after validating the convention.
    condition[:2] = old
    return torch.from_numpy(condition), bpm


class ConditionedASAPDataset(Dataset):
    def __init__(self, root, passage_group_id=None):
        self.filenames = sorted(Path(root).glob("bpm_*/passage_*/audio.wav"))
        if passage_group_id:
            prefix = "passage_{}_bpm_".format(passage_group_id)
            self.filenames = [p for p in self.filenames if p.parent.name.startswith(prefix)]
        if not self.filenames:
            raise ValueError("No selected WAV/NPZ pairs under " + str(root))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        path = self.filenames[index]
        audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
        if rate != SAMPLE_RATE or audio.shape != (SAMPLES, 2) or not np.isfinite(audio).all():
            raise ValueError("Expected finite stereo 8-kHz, 65536-sample audio: " + str(path))
        condition, _ = load_condition(path.with_name("conditioning.npz"))
        return torch.from_numpy(audio.T.copy()), condition, str(path)
