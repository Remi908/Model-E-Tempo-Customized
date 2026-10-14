#!/usr/bin/env python3
"""Create tempo-labelled real MAESTRO clips using ASAP beat annotations.

Python 3.8+. Does not need PyTorch, librosa, a GPU, or a VAE checkpoint.

Python 3.8 dependencies:
  python3.8 -m pip install numpy==1.24.4 scipy==1.10.1 soundfile==0.12.1 mido==1.2.10
Also install the native Rubber Band CLI (2.x or newer):
  sudo apt-get install rubberband-cli
or, in a conda environment: conda install -c conda-forge rubberband
Without admin/conda access, ask your cluster administrator for rubberband-cli,
or provide an existing executable using --rubberband /absolute/path/rubberband.

Example (MAESTRO root MUST contain original year folders and metadata CSV):
  python3.8 preprocess_maestro_asap.py \
    --maestro-root /data/maestro-v3.0.0 \
    --asap-dir ./asap_annotations --download-asap \
    --output ./maestro_asap_pilot --bpms 90 105 120 \
    --train-per-bpm 10 --validation-per-bpm 3

First use --scan-only with a DIFFERENT output directory to inspect coverage.
No recordings are downloaded. --download-asap downloads two annotation files
from a pinned public ASAP revision. MAESTRO v2 is the documented ASAP source;
matching v3 files are allowed but their annotation alignment needs review.

Outputs:
  train|validation/bpm_090/<id>/audio.wav       exactly 65536 samples, stereo 8k
  train|validation/bpm_090/<id>/performance.mid cropped MIDI with carry-in tags
  train|validation/bpm_090/<id>/conditioning.npz
  parents_9s/<split>/bpm_090/<id>/audio.wav     exactly 72000 samples, stereo 8k
  parents_9s/.../performance.mid, events.json
  metadata/<id>.json                          provenance, time map, diagnostics
  review/<id>/*                               click overlays (limited by quota)
  manifest.jsonl, summary.json, run_config.json, matched_sources.csv
  rejected.jsonl                             reasons for skipped sources/crops
  split_conflicts.json                       excluded ambiguous composition groups

Train ONLY on output/train, validate ONLY on output/validation. Do not recursively
load the whole output root: it also contains parents and diagnostic audio.

conditioning.npz is authoritative for onset conditioning: onset_seconds contains
only true attacks inside the training window. Cropped MIDI restores held keys
at zero, tagged CARRY_IN; do NOT blindly turn those note-ons into onset labels.
Original acoustic sustain/reverb cannot be recreated exactly from cropped MIDI.
Arrays use no pickle: np.load(path, allow_pickle=False). condition has shape
[2,4096]: normalized scalar BPM, then MAX-combined Gaussian onset pulses.
Use the same --onset-sigma-ms and condition construction at train/inference time.
The script uses no fixed maximum onset count and does not truncate note lists.

Beat labels are annotation-backed; acoustic alignment is NOT automatically
declared verified. Review source/warped click overlays before using the pilot.
The onset support diagnostic is heuristic, not a tempo correctness test.
The script does not reuse the old model-E screening manifest or estimated BPMs.
Version 1.1.0 excludes all members of composer/title groups that cross splits
instead of stopping the entire dataset build. It never reassigns official splits.
Both MAESTRO and ASAP title groupings are checked against all available metadata,
including unrequested splits and recordings not downloaded to this machine.
Version 1.2.0 also supports Rubber Band 2.x: R2 is built in, so no -2 flag is
passed. Versions 3+ receive an explicit engine flag. Timing calibration remains
mandatory. --engine 3 requires Rubber Band 3 or newer.

Sources:
  https://github.com/fosfrancesco/asap-dataset
  https://magenta.withgoogle.com/datasets/maestro
  https://breakfastquay.com/rubberband/usage.txt
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from collections import Counter, defaultdict, deque
from functools import lru_cache
from pathlib import Path

ASAP_REVISION = "afc815c75c42e83a79c03feb6da8a35e77d4c6b8"
SR = 8000
SAMPLES = 65536
TRAIN_SECONDS = SAMPLES / SR
PARENT_SAMPLES = 72000
PARENT_SECONDS = PARENT_SAMPLES / SR
LATENT_SAMPLES = SAMPLES // 16
TRAIN_OFFSET = 0.25  # Training starts 0.25 s into the 9 s parent.
FIRST_BEAT_IN_TRAIN = 0.10  # Do not cut the initial attack at the exact boundary.
SCHEMA_VERSION = 2
SCRIPT_VERSION = "1.2.0"


class SkipClip(Exception):
    pass


def dependencies():
    global np, sf, mido, signal
    try:
        import numpy as np
        import soundfile as sf
        import mido
        from scipy import signal
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: %s. For Python 3.8 run: python3.8 -m pip install "
            "numpy==1.24.4 scipy==1.10.1 soundfile==0.12.1 mido==1.2.10" % exc
        ) from exc


def dump_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def optional_float(value, default):
    if value is None or str(value).strip().lower() in ("", "nan", "none"):
        return default
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Non-finite metadata time")
    return result


def download_annotations(directory):
    directory.mkdir(parents=True, exist_ok=True)
    base = "https://raw.githubusercontent.com/fosfrancesco/asap-dataset/" + ASAP_REVISION
    for name in ("metadata.csv", "asap_annotations.json"):
        destination = directory / name
        if destination.exists():
            continue
        print("Downloading ASAP " + name, flush=True)
        request = urllib.request.Request(base + "/" + name,
                                         headers={"User-Agent": "maestro-asap-preprocessor/1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        # Validate before committing downloaded bytes.
        if name.endswith("json"):
            json.loads(data)
        else:
            if "maestro_audio_performance" not in data.decode("utf-8-sig").splitlines()[0]:
                raise RuntimeError("Unexpected ASAP metadata response")
        temporary = directory / (name + ".part")
        temporary.write_bytes(data)
        os.replace(str(temporary), str(destination))


def csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def normal_path(value):
    return str(value).replace("\\", "/").replace("{maestro}/", "").lstrip("./")


def read_catalog(args):
    metadata = Path(args.maestro_metadata).resolve() if args.maestro_metadata else None
    if metadata is None:
        found = sorted(args.maestro_root.glob("maestro-v*.csv"))
        if len(found) != 1:
            raise RuntimeError("Expected one maestro-v*.csv in --maestro-root. "
                               "Point --maestro-root at the original dataset directory, "
                               "or specify --maestro-metadata explicitly.")
        metadata = found[0]
    maestro = csv_rows(metadata)
    if not maestro or not {"audio_filename", "midi_filename", "split"}.issubset(maestro[0]):
        raise RuntimeError("MAESTRO metadata must be its original CSV with file names and split")
    indexed = {}
    # Titles sometimes name a multi-track collection or several movements.
    # Conservatively exclude ambiguous groups, rather than aborting all sources.
    compositions = defaultdict(list)
    for row in maestro:
        key = normal_path(row["audio_filename"])
        if key in indexed:
            raise RuntimeError("Duplicate MAESTRO audio path: " + key)
        indexed[key] = row
        composer = row.get("canonical_composer", "").strip().casefold()
        title = row.get("canonical_title", "").strip().casefold()
        if not composer or not title:
            raise RuntimeError("MAESTRO metadata lacks canonical composer/title for split checks")
        compositions[(composer, title)].append(row)
    maestro_conflicts = {key: group for key, group in compositions.items()
                         if len({r["split"] for r in group}) > 1}
    blocked_audio = {normal_path(r["audio_filename"])
                     for group in maestro_conflicts.values() for r in group}
    annotations_path = args.asap_dir / "asap_annotations.json"
    annotations = json.loads(annotations_path.read_text())
    rows = csv_rows(args.asap_dir / "metadata.csv")
    sources, rejected = [], Counter()
    # Check the complete ASAP/MAESTRO overlap BEFORE requested-split and local
    # file filters. A missing validation WAV must not hide a train/val conflict.
    asap_compositions = defaultdict(list)
    for row in rows:
        relative = normal_path(row.get("maestro_audio_performance", ""))
        if relative in indexed:
            group_key = (row["composer"].strip().casefold(), row["title"].strip().casefold())
            asap_compositions[group_key].append((row, indexed[relative]))
    asap_conflicts = {key: group for key, group in asap_compositions.items()
                     if len({original["split"] for _, original in group}) > 1}
    conflict_report = dict(
        policy="exclude_all_members_of_conflicting_groups_preserve_other_official_splits",
        note="Composer/title strings can be ambiguous. These are conservative exclusions, not proof of musical leakage.",
        checked_splits=sorted({r["split"] for r in maestro}),
        maestro_conflicting_groups=[dict(
            composer=group[0]["canonical_composer"], title=group[0]["canonical_title"],
            splits=sorted({r["split"] for r in group}),
            recordings=[dict(audio_filename=r["audio_filename"], split=r["split"]) for r in group])
            for _, group in sorted(maestro_conflicts.items())],
        asap_conflicting_groups=[dict(
            composer=group[0][0]["composer"], title=group[0][0]["title"],
            splits=sorted({original["split"] for _, original in group}),
            recordings=[dict(source_id=r["midi_performance"], audio_filename=original["audio_filename"],
                             split=original["split"]) for r, original in group])
            for _, group in sorted(asap_conflicts.items())],
        excluded_candidate_sources=[])
    for row in rows:
        relative = normal_path(row.get("maestro_audio_performance", ""))
        if not relative or relative not in indexed:
            rejected["not_in_supplied_maestro_metadata"] += 1
            continue
        original = indexed[relative]
        split = original["split"]
        if split not in args.splits:
            continue
        asap_key = (row["composer"].strip().casefold(), row["title"].strip().casefold())
        conflict_reasons = []
        if relative in blocked_audio:
            conflict_reasons.append("maestro_composition_split_conflict")
        if asap_key in asap_conflicts:
            conflict_reasons.append("asap_composition_split_conflict")
        if conflict_reasons:
            for reason in conflict_reasons:
                rejected[reason] += 1
            conflict_report["excluded_candidate_sources"].append(dict(
                source_id=row["midi_performance"], audio_filename=relative,
                split=split, reasons=conflict_reasons))
            continue
        audio = args.maestro_root / original["audio_filename"]
        midi = args.maestro_root / original["midi_filename"]
        if not audio.is_file() or not midi.is_file():
            rejected["original_audio_or_midi_missing"] += 1
            continue
        key = row["midi_performance"]
        if key not in annotations:
            rejected["asap_annotation_missing"] += 1
            continue
        annotation = annotations[key]
        if not annotation.get("score_and_performance_aligned", False):
            rejected["score_performance_alignment_flag_false"] += 1
            continue
        source = dict(row)
        source.update(audio=str(audio.resolve()), midi=str(midi.resolve()), split=split,
                      annotation=annotation, source_id=key,
                      maestro_relative_audio=relative)
        sources.append(source)
    return sources, rejected, metadata, conflict_report


def annotated_beats(source):
    annotation = source["annotation"]
    beats = np.asarray(annotation["performance_beats"], dtype=np.float64)
    if len(beats) < 10 or not np.isfinite(beats).all() or np.any(np.diff(beats) <= 0):
        raise SkipClip("insufficient_or_nonmonotonic_beats")
    offset = optional_float(source.get("start"), 0.0)
    return beats, beats + offset


def meter_at(annotation, seconds):
    values = sorted((float(t), v) for t, v in annotation["perf_time_signatures"].items())
    active = [v for t, v in values if t <= seconds + 1e-7]
    if not active:
        raise SkipClip("meter_not_known_before_window")
    signature, beat_count = active[-1]
    if signature not in ("2/4", "3/4", "4/4") or int(beat_count) != int(signature[0]):
        raise SkipClip("not_simple_quarter_note_meter")
    return signature


def make_plan(source, k, bpm, args, samplerate, file_frames):
    local, absolute = annotated_beats(source)
    period = 60.0 / bpm
    target = (np.arange(len(local)) - k) * period + TRAIN_OFFSET + FIRST_BEAT_IN_TRAIN
    # Warp an extended continuous passage; both output crops lie inside it.
    a = int(np.searchsorted(target, -args.guard_seconds, side="right") - 1)
    z = int(np.searchsorted(target, PARENT_SECONDS + args.guard_seconds, side="left"))
    if a < 0 or z >= len(local):
        raise SkipClip("insufficient_annotated_context")
    start, end = local[a], local[z]
    annotation = source["annotation"]
    signature = meter_at(annotation, start)
    for t, value in annotation["perf_time_signatures"].items():
        if start < float(t) <= end and value[0] != signature:
            raise SkipClip("meter_change_inside_context")
    for t, kind in annotation["performance_beats_type"].items():
        if start <= float(t) <= end and kind == "bR":
            raise SkipClip("uncertain_beat_bR")
    intervals = np.diff(local[a:z + 1])
    ratios = period / intervals  # output duration / input duration
    if np.min(ratios) < args.min_duration_ratio or np.max(ratios) > args.max_duration_ratio:
        raise SkipClip("local_stretch_outside_limits")
    end_limit = optional_float(source.get("end"), file_frames / samplerate)
    if absolute[a] < 0 or absolute[z] > min(end_limit, file_frames / samplerate):
        raise SkipClip("annotation_outside_recording_bounds")
    input_frames = np.rint(absolute[a:z + 1] * samplerate).astype(np.int64)
    output_frames = np.rint(np.arange(z - a + 1) * period * samplerate).astype(np.int64)
    if np.any(np.diff(input_frames) <= 0) or np.any(np.diff(output_frames) <= 0):
        raise SkipClip("invalid_rounded_time_map")
    parent_start_sample = int(round(-target[a] * SR))
    return dict(k=int(k), a=a, z=z, bpm=float(bpm), meter=signature,
                input_frames=input_frames, output_frames=output_frames,
                samplerate=int(samplerate), parent_start_sample=parent_start_sample,
                source_anchor_seconds=float(absolute[k]),
                source_median_bpm=float(60.0 / np.median(intervals)),
                duration_ratio_min=float(np.min(ratios)),
                duration_ratio_max=float(np.max(ratios)))


def map_times(times, plan):
    # Actual sample-rounded anchor positions used by the audio engine.
    xp = plan["input_frames"].astype(np.float64) / plan["samplerate"]
    fp = plan["output_frames"].astype(np.float64) / plan["samplerate"]
    values = np.asarray(times, dtype=np.float64)
    mapped = np.interp(values, xp, fp)
    # Extrapolation only matters for carry-in/terminal note state, outside crops.
    mapped = np.where(values < xp[0], fp[0] + (values - xp[0]) *
                      (fp[1] - fp[0]) / (xp[1] - xp[0]), mapped)
    mapped = np.where(values > xp[-1], fp[-1] + (values - xp[-1]) *
                      (fp[-1] - fp[-2]) / (xp[-1] - xp[-2]), mapped)
    return mapped


@lru_cache(maxsize=4)
def read_midi(path):
    midi = mido.MidiFile(path)
    if midi.type == 2 or midi.ticks_per_beat <= 0:
        raise SkipClip("unsupported_MIDI_timing_format")
    time = 0.0
    tempo = 500000
    active = defaultdict(deque)
    notes, controls = [], []
    for message in mido.merge_tracks(midi.tracks):
        time += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
        if message.type == "set_tempo":
            tempo = message.tempo
        elif message.type == "note_on" and message.velocity > 0:
            active[(message.channel, message.note)].append((time, message.velocity))
        elif message.type == "note_off" or (message.type == "note_on" and message.velocity == 0):
            queue = active[(message.channel, message.note)]
            if queue:
                start, velocity = queue.popleft()
                if time > start:
                    notes.append(dict(start=start, end=time, pitch=message.note,
                                      velocity=velocity, channel=message.channel))
        elif message.type in ("control_change", "program_change", "pitchwheel", "aftertouch"):
            controls.append((time, message.copy(time=0)))
    if any(active.values()):
        raise SkipClip("unclosed_MIDI_notes")
    notes.sort(key=lambda n: (n["start"], n["pitch"]))
    if not notes:
        raise SkipClip("empty_MIDI")
    return notes, controls


def window_events(notes, controls, plan, offset, duration):
    starts = map_times([n["start"] for n in notes], plan) - offset
    ends = map_times([n["end"] for n in notes], plan) - offset
    selected, true_onsets = [], []
    for original, start, end in zip(notes, starts, ends):
        if end <= 0 or start >= duration:
            continue
        note = dict(original)
        note.update(start=float(max(0.0, start)), end=float(min(duration, end)),
                    carry_in=bool(start < 0), source_start_seconds=original["start"],
                    source_end_seconds=original["end"])
        if note["end"] > note["start"]:
            selected.append(note)
            if not note["carry_in"]:
                true_onsets.append(note["start"])
    # Restore latest controller/program state at the boundary; retain all later changes.
    initial, inside = {}, []
    times = map_times([t for t, _ in controls], plan) - offset
    for new_time, (_, message) in zip(times, controls):
        key = (message.type, message.channel, getattr(message, "control", None))
        if new_time < 0:
            initial[key] = message
        elif new_time < duration:
            inside.append((float(new_time), message))
    control_events = [(0.0, msg) for msg in initial.values()] + inside
    beats = plan["output_frames"].astype(np.float64) / plan["samplerate"] - offset
    beats = beats[(beats >= -1e-9) & (beats < duration)]
    beats = np.maximum(beats, 0.0)
    return selected, control_events, np.unique(true_onsets), beats


def write_midi(path, notes, controls, bpm, meter, duration):
    midi = mido.MidiFile(type=1, ticks_per_beat=9600)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    tempo = int(round(60000000.0 / bpm))
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    track.append(mido.MetaMessage("time_signature", numerator=int(meter[0]), denominator=4, time=0))
    track.append(mido.MetaMessage("text", text="Onset conditioning: use NPZ; CARRY_IN is not an attack", time=0))
    events = []
    for seconds, message in controls:
        events.append((seconds, 0, message))
    for note in notes:
        if note["carry_in"]:
            events.append((0.0, 1, mido.MetaMessage("text", text="CARRY_IN channel=%d note=%d" %
                                                 (note["channel"], note["pitch"]))))
        events.append((note["start"], 3, mido.Message("note_on", channel=note["channel"],
                      note=note["pitch"], velocity=note["velocity"])))
        events.append((note["end"], 2, mido.Message("note_off", channel=note["channel"],
                      note=note["pitch"], velocity=0)))
    # End pedal-held sound in the exported excerpt; this is an export boundary event.
    for channel in sorted({n["channel"] for n in notes}):
        for controller in (64, 66, 67, 123):
            events.append((duration, 4, mido.Message("control_change", channel=channel,
                                                  control=controller, value=0)))
    last_tick = 0
    for seconds, priority, message in sorted(events, key=lambda item: (item[0], item[1])):
        tick = int(round(mido.second2tick(seconds, midi.ticks_per_beat, tempo)))
        track.append(message.copy(time=max(0, tick - last_tick)))
        last_tick = tick
    end_tick = int(round(mido.second2tick(duration, midi.ticks_per_beat, tempo)))
    track.append(mido.MetaMessage("end_of_track", time=max(0, end_tick - last_tick)))
    midi.save(str(path))


def condition_array(onsets, bpm, sigma_ms):
    condition = np.zeros((2, LATENT_SAMPLES), dtype=np.float32)
    condition[0] = 2.0 * (bpm - 60.0) / 120.0 - 1.0
    sigma_frames = sigma_ms * 0.001 * SR / 16
    for seconds in onsets:
        centre = seconds * SR / 16
        left = max(0, int(math.floor(centre - 4 * sigma_frames)))
        right = min(LATENT_SAMPLES, int(math.ceil(centre + 4 * sigma_frames)) + 1)
        positions = np.arange(left, right)
        pulse = np.exp(-0.5 * ((positions - centre) / sigma_frames) ** 2)
        condition[1, left:right] = np.maximum(condition[1, left:right], pulse)
    return condition


def click_overlay(audio, events, frequency):
    result = audio.astype(np.float32).copy()
    length = int(0.025 * SR)
    t = np.arange(length) / SR
    click = (0.16 * np.sin(2 * np.pi * frequency * t) * np.exp(-t / 0.005)).astype(np.float32)
    for seconds in events:
        start = int(round(seconds * SR))
        if 0 <= start < len(result):
            count = min(length, len(result) - start)
            result[start:start + count] += click[:count, None]
    peak = float(np.max(np.abs(result)))
    if peak > 0.98:
        result *= 0.98 / peak
    return result


def onset_support_diagnostic(audio, onsets):
    # Heuristic only: detector peaks are not guaranteed note or beat ground truth.
    if len(onsets) == 0:
        return {"status": "no_onsets"}
    _, times, spectra = signal.stft(audio, fs=SR, nperseg=256, noverlap=216,
                                   axis=0, boundary="zeros", padded=True)
    power = np.mean(np.abs(spectra) ** 2, axis=1)
    log_power = np.log1p(1000 * power)
    flux = np.sum(np.maximum(np.diff(log_power, axis=-1, prepend=log_power[:, :1]), 0), axis=0)
    peaks, _ = signal.find_peaks(flux, prominence=max(float(np.max(flux)) * 0.04, 1e-9), distance=3)
    detected = times[peaks]
    distances = [float(np.min(np.abs(detected - t))) for t in onsets] if len(detected) else []
    return dict(status="heuristic_only", midi_unique_onsets=int(len(onsets)),
                detected_flux_peaks=int(len(detected)),
                onset_near_flux_peak_fraction_50ms=float(np.mean(np.asarray(distances) <= 0.05)) if distances else 0.0,
                median_nearest_flux_peak_ms=float(np.median(distances) * 1000) if distances else None,
                note="Not one-to-one matching; not an acceptance gate or proof of alignment.")


def resample_to_8k(audio, rate):
    divisor = math.gcd(SR, int(rate))
    return signal.resample_poly(audio, SR // divisor, int(rate) // divisor, axis=0).astype(np.float32)


@lru_cache(maxsize=8)
def check_rubberband(executable):
    path = shutil.which(executable)
    if path is None:
        raise RuntimeError("Rubber Band CLI not found. Install rubberband-cli or pass --rubberband PATH. "
                           "pip install pyrubberband alone does not install the native executable.")
    result = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=20)
    version = (result.stdout + result.stderr).strip()
    if result.returncode:
        raise RuntimeError("Cannot run Rubber Band: " + version)
    return path, version


@lru_cache(maxsize=8)
def rubberband_engine_options(executable, engine):
    """R2 was the only engine before v3; those CLIs do not accept -2."""
    _, version = check_rubberband(executable)
    match = re.search(r"\b(\d+)\.\d+(?:\.\d+)?\b", version)
    if match is None:
        raise RuntimeError("Cannot determine Rubber Band version from: " + repr(version))
    major = int(match.group(1))
    if major < 2:
        raise RuntimeError("Rubber Band 2 or newer is required; found " + version)
    if engine not in (2, 3):
        raise ValueError("Rubber Band engine must be 2 or 3")
    if major < 3:
        if engine == 3:
            raise RuntimeError("--engine 3 requires Rubber Band 3 or newer; found " +
                               version + ". Use --engine 2 with this installation.")
        return ()
    return ("-" + str(engine),)


def run_stretcher(audio, rate, source_frames, target_frames, args, executable, work):
    input_path, output_path = work / "input.wav", work / "warped.wav"
    sf.write(str(input_path), audio, rate, subtype="FLOAT")
    # CLI anchors are sample indexes, so terminal anchors are N-1.
    cli_input, cli_output = source_frames.copy(), target_frames.copy()
    cli_input[-1] -= 1
    cli_output[-1] -= 1
    map_path = work / "time_map.txt"
    np.savetxt(str(map_path), np.column_stack([cli_input, cli_output]), fmt="%d")
    command = [executable] + list(rubberband_engine_options(executable, args.engine)) + [
               "--quiet", "--ignore-clipping",
               "--timemap", str(map_path), "--duration",
               "%.12f" % (target_frames[-1] / rate), str(input_path), str(output_path)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
    if result.returncode:
        details = (result.stderr + "\n" + result.stdout).strip()
        if len(details) > 6000:
            details = details[:3000] + "\n... output truncated ...\n" + details[-3000:]
        raise RuntimeError("Rubber Band failed (exit %d). Command: %s\n%s" %
                           (result.returncode, " ".join(shlex.quote(s) for s in command), details))
    warped, output_rate = sf.read(str(output_path), dtype="float32", always_2d=True)
    if output_rate != rate or warped.shape[1] != 2 or not np.isfinite(warped).all():
        raise RuntimeError("Invalid time-stretcher output")
    error = len(warped) - int(target_frames[-1])
    if abs(error) > max(2, round(0.005 * rate)):
        raise RuntimeError("Unexpected warped duration: %d sample difference" % error)
    return warped, error


def check_backend_timing(executable, args, rate):
    """Test actual acoustic event placement, not just requested map coordinates.

    A short deterministic signal with beat and offbeat attacks tests both slower
    and faster mapping. This cannot guarantee alignment of every piano transient.
    """
    source_seconds = np.array([0., .63, 1.24, 1.90, 2.52, 3.18, 3.80, 4.43, 5.05, 5.70, 6.32])
    source_frames = np.rint(source_seconds * rate).astype(np.int64)
    source_seconds = source_frames / rate
    attacks = []
    for i in range(1, len(source_seconds) - 2):
        attacks.extend([source_seconds[i], source_seconds[i] + .4 * (source_seconds[i+1] - source_seconds[i])])
    audio = np.zeros((int(source_frames[-1]), 2), dtype=np.float32)
    t = np.arange(round(.06 * rate)) / rate
    pulse = (.5 * np.cos(2*np.pi*440*t) * np.exp(-t/.012)).astype(np.float32)
    for attack in attacks:
        pos = round(attack * rate)
        audio[pos:pos+len(pulse), 0] += pulse
        audio[pos:pos+len(pulse), 1] += .8 * pulse
    reports = []
    for period in (.55, .72):
        target_frames = np.rint(np.arange(len(source_frames)) * period * rate).astype(np.int64)
        expected = np.interp(attacks, source_seconds, target_frames / rate)
        with tempfile.TemporaryDirectory(prefix="rubberband_timing_") as temporary:
            output, _ = run_stretcher(audio, rate, source_frames, target_frames, args, executable, Path(temporary))
        envelope = np.convolve(output[:, 0] ** 2, np.ones(max(1, round(.002*rate))) /
                               max(1, round(.002*rate)), mode="same")
        peaks, _ = signal.find_peaks(envelope, height=.002, distance=round(.1*rate))
        if not len(peaks):
            raise RuntimeError("Rubber Band timing check produced no detectable attacks")
        errors = np.array([np.min(np.abs(peaks/rate - attack)) for attack in expected])
        median, p95 = np.quantile(errors, [.5, .95]) * 1000
        reports.append(dict(target_period_seconds=period, median_attack_error_ms=float(median),
                            p95_attack_error_ms=float(p95)))
        if median > 20 or p95 > 40:
            raise RuntimeError("Rubber Band timing check failed at %d Hz (median %.1f ms, p95 %.1f ms). "
                               "Use --engine 2 and check the installed Rubber Band version. "
                               "Do not train on unchecked timing labels." % (rate, median, p95))
    return dict(sample_rate=rate, engine=args.engine, checks=reports,
                note="Synthetic backend calibration only; recorded-piano alignment still needs review.")


def process_clip(source, plan, args, executable, work, review):
    info = sf.info(source["audio"])
    start = int(plan["input_frames"][0])
    stop = int(plan["input_frames"][-1])
    with sf.SoundFile(source["audio"]) as handle:
        handle.seek(start)
        audio = handle.read(stop - start, dtype="float32", always_2d=True)
    if audio.shape != (stop - start, 2) or not np.isfinite(audio).all():
        raise SkipClip("invalid_source_audio_shape_or_values")
    if np.max(np.abs(audio)) < 1e-5:
        raise SkipClip("silent_source")
    input_anchors = plan["input_frames"] - start
    output_anchors = plan["output_frames"].copy()
    warped, native_length_error = run_stretcher(audio, info.samplerate, input_anchors,
                                               output_anchors, args, executable, work)
    warped_8k = resample_to_8k(warped, info.samplerate)
    parent_start = plan["parent_start_sample"]
    parent = warped_8k[parent_start:parent_start + PARENT_SAMPLES].copy()
    if parent.shape != (PARENT_SAMPLES, 2):
        raise SkipClip("insufficient_warped_audio_no_silence_padding_performed")
    peak = float(np.max(np.abs(parent)))
    gain = min(1.0, 0.98 / max(peak, 1e-12))
    parent *= gain
    train_start = int(round(TRAIN_OFFSET * SR))
    train = parent[train_start:train_start + SAMPLES].copy()
    rms = float(np.sqrt(np.mean(train.astype(np.float64) ** 2)))
    if rms < args.min_rms:
        raise SkipClip("training_audio_below_min_rms")
    notes, controls = read_midi(source["midi"])
    parent_offset = parent_start / SR
    parent_events = window_events(notes, controls, plan, parent_offset, PARENT_SECONDS)
    train_events = window_events(notes, controls, plan, parent_offset + TRAIN_OFFSET, TRAIN_SECONDS)
    train_notes, train_controls, onsets, beats = train_events
    if len(onsets) < args.min_onsets:
        raise SkipClip("too_few_MIDI_onsets")
    if len(beats) < 3:
        raise SkipClip("too_few_beats")
    beat_error = float(np.max(np.abs(np.diff(beats) - 60.0 / plan["bpm"])))
    if beat_error > 2.0 / info.samplerate:
        raise RuntimeError("Internal transformed beat-spacing check failed")
    for name in ("training", "parent"):
        (work / name).mkdir()
    for name, values, event_data, duration in (
        ("training", train, train_events, TRAIN_SECONDS),
        ("parent", parent, parent_events, PARENT_SECONDS),
    ):
        folder = work / name
        ns, cs, ons, bs = event_data
        sf.write(str(folder / "audio.wav"), values, SR, subtype="PCM_16")
        write_midi(folder / "performance.mid", ns, cs, plan["bpm"], plan["meter"], duration)
        dump_json(folder / "events.json", dict(note_onsets_seconds=ons.tolist(),
                  beat_times_seconds=bs.tolist(), notes=ns,
                  controllers=[dict(time=t, message=m.dict()) for t, m in cs]))
    condition = condition_array(onsets, plan["bpm"], args.onset_sigma_ms)
    np.savez_compressed(str(work / "training" / "conditioning.npz"),
                        condition=condition, onset_seconds=onsets, beat_seconds=beats,
                        target_bpm=np.float64(plan["bpm"]),
                        note_start=np.asarray([n["start"] for n in train_notes]),
                        note_end=np.asarray([n["end"] for n in train_notes]),
                        note_pitch=np.asarray([n["pitch"] for n in train_notes], dtype=np.int16),
                        note_velocity=np.asarray([n["velocity"] for n in train_notes], dtype=np.int16),
                        note_carry_in=np.asarray([n["carry_in"] for n in train_notes], dtype=bool))
    if review:
        folder = work / "review"
        folder.mkdir()
        sf.write(str(folder / "training_with_beats.wav"), click_overlay(train, beats, 1200), SR)
        sf.write(str(folder / "training_with_onsets.wav"), click_overlay(train, onsets, 1800), SR)
        source_8k = resample_to_8k(audio, info.samplerate)
        source_beats = (plan["input_frames"] - start) / info.samplerate
        sf.write(str(folder / "source_with_annotated_beats.wav"),
                 click_overlay(source_8k, source_beats, 1200), SR)
        source_onsets = [n["start"] - start/info.samplerate for n in notes
                         if start/info.samplerate <= n["start"] < stop/info.samplerate]
        sf.write(str(folder / "source_with_midi_onsets.wav"),
                 click_overlay(source_8k, source_onsets, 1800), SR)
    written = sf.info(str(work / "training" / "audio.wav"))
    if (written.frames, written.samplerate, written.channels) != (SAMPLES, SR, 2):
        raise RuntimeError("Saved waveform shape check failed")
    diagnostics = dict(structural_checks_passed=True, human_review_status="pending",
                       acoustic_alignment_verified=False,
                       annotation_file_version_alignment_verified=False,
                       target_bpm=plan["bpm"], beat_unit="quarter_note",
                       label_basis="ASAP_annotated_beats_mapped_to_constant_tempo",
                       mathematical_beat_spacing_max_error_ms=1000 * beat_error,
                       native_warp_length_error_samples=int(native_length_error),
                       rms=rms, peak=float(np.max(np.abs(train))), gain_applied=gain,
                       channel_dc=np.mean(train, axis=0).astype(float).tolist(),
                       onset_diagnostic=onset_support_diagnostic(train, onsets),
                       warning="Annotations and a valid time map do not independently verify acoustic timing.")
    return onsets, beats, diagnostics


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version="%(prog)s " + SCRIPT_VERSION)
    parser.add_argument("--maestro-root", type=Path, required=True)
    parser.add_argument("--maestro-metadata", help="Original MAESTRO CSV; default: find it in root")
    parser.add_argument("--asap-dir", type=Path, default=Path("asap_annotations"))
    parser.add_argument("--download-asap", action="store_true")
    parser.add_argument("--output", type=Path, required=True, help="New directory; never overwrites an existing run")
    parser.add_argument("--bpms", type=float, nargs="+", default=[90, 105, 120])
    parser.add_argument("--splits", nargs="+", choices=["train", "validation", "test"], default=["train", "validation"])
    parser.add_argument("--train-per-bpm", type=int, default=10)
    parser.add_argument("--validation-per-bpm", type=int, default=3)
    parser.add_argument("--test-per-bpm", type=int, default=3)
    parser.add_argument("--max-per-source-per-bpm", type=int, default=1)
    parser.add_argument("--max-anchors-per-performance", type=int, default=100)
    parser.add_argument("--min-duration-ratio", type=float, default=0.8)
    parser.add_argument("--max-duration-ratio", type=float, default=1.25)
    parser.add_argument("--guard-seconds", type=float, default=1.0)
    parser.add_argument("--min-onsets", type=int, default=8)
    parser.add_argument("--min-rms", type=float, default=0.001)
    parser.add_argument("--onset-sigma-ms", type=float, default=10.0)
    parser.add_argument("--review-per-bpm", type=int, default=1, help="Click-overlay sets per split/BPM")
    parser.add_argument("--rubberband", default="rubberband")
    parser.add_argument("--engine", type=int, choices=[2, 3], default=2,
                        help="Default 2 passed timing tests; either engine must pass backend calibration")
    parser.add_argument("--timeout", type=int, default=180, help="Seconds allowed per time-stretch call")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--scan-only", action="store_true", help="Check file/annotation/MIDI eligibility; write no audio")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    dependencies()
    args.maestro_root = args.maestro_root.expanduser().resolve()
    args.asap_dir = args.asap_dir.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.bpms = sorted(set(args.bpms))
    if not args.bpms or any(not math.isfinite(b) or b < 60 or b > 180 for b in args.bpms):
        raise RuntimeError("BPM values must be finite and between 60 and 180")
    if any(abs(b - round(b, 2)) > 1e-8 for b in args.bpms):
        raise RuntimeError("BPM values may have at most two decimal places")
    if not (0 < args.min_duration_ratio <= args.max_duration_ratio):
        raise RuntimeError("Invalid duration-ratio limits")
    if args.onset_sigma_ms <= 0 or args.guard_seconds < 0.25 or args.min_rms < 0:
        raise RuntimeError("Use positive sigma, guard >= 0.25 seconds, and nonnegative min RMS")
    if min(args.train_per_bpm, args.validation_per_bpm, args.test_per_bpm,
           args.max_per_source_per_bpm, args.max_anchors_per_performance, args.min_onsets) <= 0:
        raise RuntimeError("Counts must be positive")
    if args.output.exists():
        raise RuntimeError("Output already exists; choose a new --output directory: " + str(args.output))
    if args.download_asap:
        download_annotations(args.asap_dir)
    for name in ("metadata.csv", "asap_annotations.json"):
        if not (args.asap_dir / name).is_file():
            raise RuntimeError("Missing ASAP " + name + "; use --download-asap or supply the files")
    sources, catalog_rejections, maestro_metadata, conflict_report = read_catalog(args)
    conflict_count = len(conflict_report["excluded_candidate_sources"])
    if conflict_report["maestro_conflicting_groups"] or conflict_report["asap_conflicting_groups"]:
        print("Split check: %d MAESTRO and %d ASAP ambiguous groups; "
              "excluded %d requested candidate rows. Remaining official splits are preserved." %
              (len(conflict_report["maestro_conflicting_groups"]),
               len(conflict_report["asap_conflicting_groups"]), conflict_count), flush=True)
    executable, version = (None, "not used in scan-only mode") if args.scan_only else check_rubberband(args.rubberband)
    engine_options = () if args.scan_only else rubberband_engine_options(executable, args.engine)
    if not sources:
        args.output.mkdir(parents=True)
        dump_json(args.output / "split_conflicts.json", conflict_report)
        dump_json(args.output / "summary.json", dict(status="no_eligible_sources",
                  catalog_rejections=dict(catalog_rejections)))
        raise RuntimeError("No eligible matched source files. Check the original MAESTRO root and metadata.")
    args.output.mkdir(parents=True)
    dump_json(args.output / "split_conflicts.json", conflict_report)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(schema_version=SCHEMA_VERSION, script_version=SCRIPT_VERSION, python=sys.version,
                  rubberband_version=version, rubberband_engine_options=list(engine_options),
                  asap_download_revision_if_fetched=ASAP_REVISION,
                  asap_metadata_sha256=digest(args.asap_dir / "metadata.csv"),
                  asap_annotations_sha256=digest(args.asap_dir / "asap_annotations.json"),
                  maestro_metadata=str(maestro_metadata), maestro_metadata_sha256=digest(maestro_metadata),
                  script_sha256=digest(__file__), sample_rate=SR, samples=SAMPLES,
                  parent_samples=PARENT_SAMPLES, train_offset_seconds=TRAIN_OFFSET,
                  first_beat_in_train_seconds=FIRST_BEAT_IN_TRAIN)
    dump_json(args.output / "run_config.json", config)
    with (args.output / "matched_sources.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_id", "split", "audio", "midi", "start", "end"])
        writer.writeheader()
        writer.writerows({k: s.get(k, "") for k in writer.fieldnames} for s in sources)
    rng = random.Random(args.seed)
    rng.shuffle(sources)
    quotas = {s: getattr(args, s + "_per_bpm") for s in args.splits}
    counts, per_source, reviews, rejected_counts = Counter(), Counter(), Counter(), Counter()
    backend_reports = {}
    manifests = args.output / ("candidate_manifest.jsonl" if args.scan_only else "manifest.jsonl")
    status, fatal_error = "completed", None
    print("Preprocessor v%s: matched %d annotated performances. Labels use quarter-note beats." %
          (SCRIPT_VERSION, len(sources)), flush=True)
    print("Parent: 9.000 s; training: 8.192 s; no VAE changes.", flush=True)
    if not args.scan_only:
        print("Rubber Band %s: R%d (%s). Timing calibration is enabled." %
              (version, args.engine, " ".join(engine_options) if engine_options else "built-in engine; no selector flag"),
              flush=True)
    print("Review source/warped alignment, especially when using MAESTRO v3.", flush=True)
    with manifests.open("w") as manifest, (args.output / "rejected.jsonl").open("w") as rejection_log:
        def reject(source, reason, **extra):
            rejected_counts[reason] += 1
            rejection_log.write(json.dumps(dict(source_id=source["source_id"], reason=reason, **extra)) + "\n")
            rejection_log.flush()
        try:
            for source in sources:
                split = source["split"]
                if all(counts[(split, b)] >= quotas[split] for b in args.bpms):
                    continue
                try:
                    info = sf.info(source["audio"])
                    if info.channels != 2 or info.samplerate < 22050:
                        raise SkipClip("source_must_be_original_stereo_audio_at_least_22050Hz")
                    local, absolute = annotated_beats(source)
                    notes, controls = read_midi(source["midi"])
                    downbeats = np.asarray(source["annotation"]["performance_downbeats"])
                    candidates = [int(np.argmin(np.abs(local - t))) for t in downbeats
                                  if np.min(np.abs(local - t)) < 0.001]
                    candidates = sorted(set(candidates))
                    rng.shuffle(candidates)
                    # The same source beat anchor is considered across all target tempos.
                    for k in candidates[:args.max_anchors_per_performance]:
                        for bpm in args.bpms:
                            bucket = (split, bpm)
                            performance_key = (source["audio"], bpm)
                            if counts[bucket] >= quotas[split] or per_source[performance_key] >= args.max_per_source_per_bpm:
                                continue
                            try:
                                plan = make_plan(source, k, bpm, args, info.samplerate, info.frames)
                                offset = plan["parent_start_sample"] / SR + TRAIN_OFFSET
                                _, _, onsets, beats = window_events(notes, controls, plan, offset, TRAIN_SECONDS)
                                if len(onsets) < args.min_onsets:
                                    raise SkipClip("too_few_MIDI_onsets")
                                stem = hashlib.sha256((source["source_id"] + "|" + str(k)).encode()).hexdigest()[:12]
                                bpm_slug = ("%06.2f" % bpm).replace(".", "p") if bpm % 1 else "%03d" % int(bpm)
                                clip_id = "passage_" + stem + "_bpm_" + bpm_slug
                                training_relative = Path(split) / ("bpm_" + bpm_slug) / clip_id
                                parent_relative = Path("parents_9s") / training_relative
                                metadata_relative = Path("metadata") / (clip_id + ".json")
                                record = dict(clip_id=clip_id, split=split, target_bpm=bpm,
                                              sample_rate=SR, channels=2, num_samples=SAMPLES,
                                              duration_seconds=TRAIN_SECONDS, source_audio=source["audio"],
                                              source_midi=source["midi"], source_id=source["source_id"],
                                              source_passage_number=int(k), passage_group_id=stem,
                                              source_anchor_seconds=plan["source_anchor_seconds"],
                                              source_pulse_bpm=plan["source_median_bpm"],
                                              audio_path=str(args.output / training_relative / "audio.wav"),
                                              midi_path=str(args.output / training_relative / "performance.mid"),
                                              conditioning_path=str(args.output / training_relative / "conditioning.npz"),
                                              parent_audio_path=str(args.output / parent_relative / "audio.wav"),
                                              metadata_path=str(args.output / metadata_relative),
                                              midi_onsets_seconds=onsets.tolist(), midi_onset_count=len(onsets),
                                              beat_times_seconds=beats.tolist(), beat_unit="quarter_note",
                                              crop_start_in_parent_seconds=TRAIN_OFFSET,
                                              label_basis="ASAP_annotated_beats_mapped_to_constant_tempo",
                                              human_review_status="pending", acoustic_alignment_verified=False,
                                              scan_only=bool(args.scan_only))
                                if not args.scan_only:
                                    if info.samplerate not in backend_reports:
                                        print("Checking Rubber Band timing at %d Hz..." % info.samplerate, flush=True)
                                        backend_reports[info.samplerate] = check_backend_timing(executable, args, info.samplerate)
                                        dump_json(args.output / "backend_timing_checks.json", backend_reports)
                                    make_review = reviews[bucket] < args.review_per_bpm
                                    with tempfile.TemporaryDirectory(prefix="maestro_asap_") as temporary:
                                        work = Path(temporary)
                                        onsets, beats, diagnostics = process_clip(source, plan, args, executable, work, make_review)
                                        details = dict(record, diagnostics=diagnostics,
                                                       source_midi_sha256=digest(source["midi"]),
                                                       source_metadata_start=source.get("start", ""),
                                                       source_metadata_end=source.get("end", ""),
                                                       duration_ratio_min=plan["duration_ratio_min"],
                                                       duration_ratio_max=plan["duration_ratio_max"],
                                                       meter=plan["meter"], onset_sigma_ms=args.onset_sigma_ms,
                                                       onset_pulse_aggregation="max", condition_shape=[2, LATENT_SAMPLES],
                                                       source_anchor_frames=plan["input_frames"].tolist(),
                                                       destination_anchor_frames=plan["output_frames"].tolist(),
                                                       anchor_sample_rate=info.samplerate,
                                                       parent_start_sample_in_warped_8k=plan["parent_start_sample"])
                                        for src, dst in ((work / "training", args.output / training_relative),
                                                         (work / "parent", args.output / parent_relative)):
                                            dst.parent.mkdir(parents=True, exist_ok=True)
                                            shutil.copytree(str(src), str(dst))
                                        if make_review:
                                            shutil.copytree(str(work / "review"), str(args.output / "review" / clip_id))
                                            reviews[bucket] += 1
                                        (args.output / metadata_relative).parent.mkdir(exist_ok=True)
                                        dump_json(args.output / metadata_relative, details)
                                        record["structural_checks_passed"] = True
                                manifest.write(json.dumps(record, allow_nan=False) + "\n")
                                manifest.flush()
                                counts[bucket] += 1
                                per_source[performance_key] += 1
                                print("[%s %g BPM %d/%d] %s%s" %
                                      (split, bpm, counts[bucket], quotas[split], clip_id,
                                       " (candidate; audio not processed)" if args.scan_only else ""), flush=True)
                            except SkipClip as exc:
                                reject(source, str(exc), beat_index=k, target_bpm=bpm)
                        if all(counts[(split, b)] >= quotas[split] or
                               per_source[(source["audio"], b)] >= args.max_per_source_per_bpm for b in args.bpms):
                            break
                except SkipClip as exc:
                    reject(source, str(exc))
        except BaseException as exc:
            status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            fatal_error = type(exc).__name__ + ": " + str(exc)
            raise
        finally:
            summary = dict(status=status, error=fatal_error, scan_only=args.scan_only,
                           clips=sum(counts.values()), matched_sources=len(sources),
                           counts={"%s/%g" % key: value for key, value in counts.items()},
                           missing_quotas={"%s/%g" % (s, b): quotas[s] - counts[(s, b)]
                                           for s in args.splits for b in args.bpms if counts[(s, b)] < quotas[s]},
                           catalog_rejections=dict(catalog_rejections), crop_rejections=dict(rejected_counts),
                           split_conflicts_report="split_conflicts.json",
                           excluded_split_conflict_candidate_rows=conflict_count,
                           next_action=("Resolve the error above before training; rerun with a new output directory."
                                        if fatal_error else
                                        "Review the original and warped beat/onset overlays; acoustic timing is not auto-verified."))
            dump_json(args.output / "summary.json", summary)
    print("Results: " + str(args.output), flush=True)
    if summary["missing_quotas"]:
        print("Coverage shortfall: " + json.dumps(summary["missing_quotas"]) +
              ". The script did not silently increase stretch limits or duplicate samples.")
    if not counts:
        raise RuntimeError("No clips passed. Inspect summary.json and rejected.jsonl before changing limits.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        sys.exit(1)
