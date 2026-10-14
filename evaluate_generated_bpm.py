#!/usr/bin/env python3
"""Fully automatic folder-level tempo evaluation, with no manual annotations.

Same BPM arithmetic as the MAESTRO/ASAP preprocessing:
    BPM = 60 / median(diff(beat_seconds))
Here beat_seconds are AUTOMATIC PREDICTIONS from the generated WAV, not ASAP
annotations. This is NOT the identical labeling process and is not ground truth.
No audio time-warping, requested grid, MIDI notes, target BPM or filename is
supplied to the tracker. A dataset label assigned by time-warping cannot measure
whether a newly generated recording followed its conditioning.

Recommended backend: Beat This! (official pretrained models final0/final1/final2).
Use a separate Python 3.10+ environment with torch>=2, torchaudio, beat-this,
numpy and soundfile. CPU works; --device cuda is optional with a CUDA torch build.
Models download on first use. --models also accepts local Beat This checkpoints.

Compatibility backend: --backend librosa works with Python 3.8 and
librosa>=0.10,<0.12, numpy, scipy and soundfile. This is the previously ambiguous
classical estimator; it is provided as a baseline, not a guaranteed improvement.

Example:
  python evaluate_generated_bpm.py --input-dir model_e_30s_existing \
      --output model_e_automatic_bpm
  python3.8 evaluate_generated_bpm.py --input-dir model_e_30s_existing \
      --backend librosa --output model_e_automatic_bpm_librosa

If input-dir/references contains audio.wav + conditioning.npz pairs, they are
automatically evaluated as known-label controls. --reference-dir accepts another
directory of such pairs, e.g. the preprocessed validation set. Parameters and
model selection never change to make their predictions fit those labels.

Outputs: generated_results.csv/json, reference_results.csv/json,
reference_summary.json, per_clip/*.json, summary.txt, run_config.json.
The primary model is fixed BEFORE evaluation: first --models entry, or Librosa
prior 120. Alternative models/priors are diagnostics, never chosen by target error.
Reports show strict, half-time and double-time matches separately at 4% tolerance.
Half/double candidates and tolerant scores never replace the raw measured BPM.
This reporting change does not resolve the musical beat level or prove control.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

VERSION = "1.1.0"
FORMULA = "60 / median(diff(automatically_detected_beat_seconds))"
COMPARISON_TOLERANCE_PERCENT = 4.0


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def number(value):
    x = float(value)
    if not math.isfinite(x) or x <= 0:
        raise ValueError("BPM must be positive and finite")
    return x


def load_audio(path):
    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    if len(audio) == 0 or sr <= 0 or not np.isfinite(audio).all():
        raise ValueError("Empty or nonfinite audio")
    center = audio.astype(np.float64) - audio.mean(axis=0, dtype=np.float64)
    energy = np.mean(center ** 2, axis=0)
    mono = center.mean(axis=1)
    mode = "channel_mean"
    # Protect a beat detector from nearly anti-phase stereo cancellation.
    if np.mean(mono ** 2) < 0.0625 * np.mean(energy):
        mono = center[:, int(np.argmax(energy))]
        mode = "highest_rms_channel_due_to_stereo_cancellation"
    rms = float(np.sqrt(np.mean(center ** 2)))
    return center.astype(np.float32), mono.astype(np.float32), sr, dict(
        duration_seconds=len(audio) / sr, sample_rate=int(sr), samples=len(audio),
        channels=audio.shape[1], dc_per_channel=audio.mean(axis=0, dtype=np.float64).tolist(),
        rms_dbfs=float(20 * np.log10(max(rms, 1e-12))),
        peak=float(np.max(np.abs(audio))),
        samples_at_or_above_1_fraction=float(np.mean(np.abs(audio) >= 1)),
        mono_method=mode, silence_or_constant_signal=bool(rms < 1e-7))


def clean_times(values, duration):
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
        raise ValueError("Tracker returned nonfinite, duplicate or unordered beats")
    # Network padding can return positions just outside the actual recording.
    return values[(values >= 0) & (values < duration)]


def beat_stats(beats, duration):
    n = len(beats)
    base = dict(measured_bpm=None, beat_count=n, interval_cv=None,
                interval_iqr_ms=None, annotated_or_detected_span_fraction=0.0,
                first_beat_seconds=float(beats[0]) if n else None,
                last_beat_seconds=float(beats[-1]) if n else None)
    if n < 3:
        return base
    intervals = np.diff(beats)
    base.update(measured_bpm=float(60.0 / np.median(intervals)),
                interval_cv=float(np.std(intervals) / np.mean(intervals)),
                interval_iqr_ms=float(1000 * (np.percentile(intervals, 75) - np.percentile(intervals, 25))),
                annotated_or_detected_span_fraction=float((beats[-1] - beats[0]) / duration))
    return base


def match_beats(reference, measured, tolerance=0.070):
    i = j = matches = 0
    while i < len(reference) and j < len(measured):
        d = measured[j] - reference[i]
        if d < -tolerance:
            j += 1
        elif d > tolerance:
            i += 1
        else:
            matches += 1
            i += 1
            j += 1
    return dict(matches=matches, reference_count=len(reference), detected_count=len(measured),
                f1=2 * matches / (len(reference) + len(measured)) if len(reference) + len(measured) else None,
                tolerance_ms=1000 * tolerance)


def target_errors(bpm, target):
    """Compare AFTER inference. No target-dependent correction of the estimate.

    The half/double score permits factors (0.5, 1, 2) only. It is not standard
    ACC2, which also permits factors (1/3, 3). Matches describe agreement with
    a supplied target, not independently verified quarter-note tempo.
    """
    if bpm is not None:
        bpm = number(bpm)
    if target is not None:
        target = number(target)
    result = dict(
        half_bpm_candidate=bpm / 2 if bpm is not None else None,
        double_bpm_candidate=bpm * 2 if bpm is not None else None,
        absolute_bpm_error=None, relative_error_percent=None,
        half_double_tolerant_error_percent=None,
        raw_to_target_ratio=None, log2_raw_to_target_ratio=None,
        strict_match=None, half_double_tolerant_match=None,
        scoring_factor_for_half_double_metric=None,
        comparison_tolerance_percent=COMPARISON_TOLERANCE_PERCENT,
        tempo_comparison="unavailable" if bpm is None else "no_target")
    if bpm is None or target is None:
        return result
    strict_error = 100 * abs(bpm - target) / target
    # Identity first makes the tie policy explicit; factors are for SCORING only.
    factor = min((1.0, 0.5, 2.0), key=lambda f: abs(bpm * f - target))
    tolerant_error = 100 * abs(bpm * factor - target) / target
    strict_match = strict_error <= COMPARISON_TOLERANCE_PERCENT
    tolerant_match = tolerant_error <= COMPARISON_TOLERANCE_PERCENT
    relation = "strict_match" if strict_match else (
        "double_time_match" if tolerant_match and factor == 0.5 else
        "half_time_match" if tolerant_match and factor == 2.0 else "mismatch")
    result.update(absolute_bpm_error=abs(bpm - target), relative_error_percent=strict_error,
                  half_double_tolerant_error_percent=tolerant_error,
                  raw_to_target_ratio=bpm / target, log2_raw_to_target_ratio=math.log2(bpm / target),
                  strict_match=strict_match, half_double_tolerant_match=tolerant_match,
                  scoring_factor_for_half_double_metric=factor, tempo_comparison=relation)
    return result


class Tracker:
    def __init__(self, args):
        self.backend = args.backend
        self.model_hashes = {}
        if args.backend == "beat-this":
            try:
                import torch
                if int(torch.__version__.split(".")[0]) < 2:
                    raise RuntimeError("Beat This requires torch>=2. Use a separate evaluation environment; do not upgrade the training environment.")
                from beat_this.inference import Audio2Beats
            except ImportError as exc:
                raise RuntimeError("Install torch, torchaudio and beat-this in a separate Python 3.10+ evaluation environment, or explicitly use --backend librosa in the existing environment.") from exc
            if args.device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable for this evaluation environment; use --device cpu")
            torch.set_num_threads(args.threads)
            self.names = list(dict.fromkeys(args.models))
            self.models = {}
            for name in self.names:
                print("Loading Beat This model:", name, flush=True)
                self.models[name] = Audio2Beats(checkpoint_path=name, device=args.device,
                                              float16=False, dbn=False)
                weight_path = Path(name) if Path(name).is_file() else (
                    Path(torch.hub.get_dir()) / "checkpoints" / ("beat_this-" + name + ".ckpt"))
                if weight_path.is_file():
                    self.model_hashes[name] = digest(weight_path)
        else:
            import librosa
            self.librosa = librosa
            self.names = ["librosa_prior_120", "librosa_prior_60", "librosa_prior_90", "librosa_prior_180"]
        self.primary = self.names[0]

    def predict(self, audio, mono, sr, duration):
        # No target, reference grid, path or filename is accepted here.
        tracks = {}
        if self.backend == "beat-this":
            for name, model in self.models.items():
                beats, downbeats = model(mono, sr)
                tracks[name] = dict(beat_seconds=clean_times(beats, duration).tolist(),
                                    downbeat_seconds=clean_times(downbeats, duration).tolist())
        else:
            librosa = self.librosa
            # Preserve the audit's time resolution at any actual input sample rate.
            hop = max(1, int(round(sr * 0.005)))
            n_fft = max(32, int(round(sr * 0.064)))
            power = np.mean([np.abs(librosa.stft(audio[:, c], n_fft=n_fft,
                            hop_length=hop)) ** 2 for c in range(audio.shape[1])], axis=0)
            mel = librosa.feature.melspectrogram(S=power, sr=sr, n_fft=n_fft,
                                                n_mels=64, fmax=sr/2)
            env = librosa.onset.onset_strength(S=librosa.power_to_db(mel, ref=1.0),
                                              sr=sr, n_fft=n_fft, hop_length=hop)
            env = env[np.arange(len(env)) * hop / sr < duration]
            for name in self.names:
                prior = int(name.rsplit("_", 1)[1])
                if float(np.max(env)) < 1e-10:
                    estimated, frames = 0, np.empty(0, dtype=int)
                else:
                    estimated, frames = librosa.beat.beat_track(onset_envelope=env, sr=sr,
                                        hop_length=hop, start_bpm=prior, trim=True)
                estimated = float(np.asarray(estimated).reshape(-1)[0])
                beats = librosa.frames_to_time(frames, sr=sr, hop_length=hop)
                tracks[name] = dict(beat_seconds=clean_times(beats, duration).tolist(),
                                    downbeat_seconds=[],
                                    onset_autocorrelation_bpm=estimated if estimated > 0 else None)
        for track in tracks.values():
            track.update(beat_stats(np.asarray(track["beat_seconds"]), duration))
        return tracks


def measure(path, tracker):
    audio, mono, sr, stats = load_audio(path)
    duration = stats["duration_seconds"]
    flags = []
    if stats["silence_or_constant_signal"] or duration < 4:
        flags.append("silence_or_constant_signal" if stats["silence_or_constant_signal"] else "audio_shorter_than_4_seconds")
        tracks = {name: dict(beat_seconds=[], downbeat_seconds=[], **beat_stats([], duration))
                  for name in tracker.names}
    else:
        tracks = tracker.predict(audio, mono, sr, duration)
    primary = tracks[tracker.primary]
    values = [v["measured_bpm"] for v in tracks.values() if v["measured_bpm"] is not None]
    if primary["measured_bpm"] is None:
        flags.append("primary_tracker_found_fewer_than_three_beats")
    elif primary["beat_count"] < 8:
        flags.append("few_detected_beats")
    if duration < 15:
        flags.append("short_context_less_than_15_seconds")
    if primary["interval_cv"] is not None and primary["interval_cv"] > 0.10:
        flags.append("variable_beat_intervals")
    if primary["annotated_or_detected_span_fraction"] < 0.75:
        flags.append("limited_beat_coverage")
    spread = 100 * (max(values)-min(values)) / np.median(values) if values else None
    if len(tracks) < 2:
        flags.append("single_model_only")
    elif len(values) != len(tracks):
        flags.append("some_trackers_found_no_tempo")
    if spread is not None and spread > 5:
        flags.append("models_or_priors_disagree")
    if any(abs(max(x,y)/min(x,y)-2) < .08 for i,x in enumerate(values) for y in values[i+1:]):
        flags.append("half_double_time_disagreement")
    segments = []
    b = np.asarray(primary["beat_seconds"])
    for start in np.arange(0, duration, 10.0):
        end = min(start+10, duration)
        local = b[(b >= start) & (b < end)]
        segments.append(dict(start_seconds=float(start), end_seconds=float(end),
                             **beat_stats(local, float(end-start))))
    return dict(file=str(path.resolve()), audio_sha256=digest(path), audio_statistics=stats,
                primary_model=tracker.primary, measured_bpm=primary["measured_bpm"],
                status="unavailable" if primary["measured_bpm"] is None else
                       ("ambiguous_automatic_estimate" if "models_or_priors_disagree" in flags
                        else "automatic_estimate_unverified"),
                flags=flags, candidate_spread_percent=float(spread) if spread is not None else None,
                tracks=tracks, ten_second_interval_summaries=segments,
                window_note="Summaries of the same full-recording beat sequence, not independently measured windows",
                beat_unit_note="Detected pulse; correspondence to quarter notes is inferred, not verified",
                ground_truth_verified=False)


def discover(root, recursive, include_raw, output):
    candidates = sorted(root.rglob("*") if recursive else root.iterdir())
    selected, skipped = [], []
    for p in candidates:
        if not p.is_file() or p.suffix.lower() not in (".wav", ".flac", ".ogg"):
            continue
        rel = p.relative_to(root)
        if output == p.resolve() or output in p.resolve().parents:
            continue
        if any(x.lower() in ("references", "review", "annotations") for x in rel.parts[:-1]):
            skipped.append(dict(file=str(p), reason="reference_or_review_directory")); continue
        if p.name.lower().startswith("review_") or p.stem.lower().endswith(("_with_beats", "_with_onsets")):
            skipped.append(dict(file=str(p), reason="click_overlay")); continue
        if not include_raw and p.stem.endswith("_raw"):
            corrected = p.with_name(p.stem[:-4] + "_dc_removed" + p.suffix)
            if corrected.is_file():
                skipped.append(dict(file=str(p), reason="DC_corrected_counterpart_selected")); continue
        selected.append(p)
    return selected, skipped


def metadata_target(path):
    # Called only AFTER waveform inference. Filenames never supply guessed BPMs.
    matches = []
    for p in path.parent.glob("*_metadata.json"):
        try:
            m = read_json(p)
            for key in ("generated_audio", "raw_audio"):
                if key in m and (p.parent / m[key]).resolve() == path.resolve():
                    if key == "generated_audio" and m.get("generated_audio_sha256"):
                        if digest(path) != m["generated_audio_sha256"]:
                            return None, "generation_metadata_checksum_mismatch"
                    matches.append(number(m["target_bpm"]))
            # Support the fixed validation demos from train_model_e_expanded.py.
            # Names are matched to explicit metadata records, after prediction.
            if p.name == "demo_metadata.json":
                for record in m.get("clips", []):
                    bpm = number(record["requested_bpm"])
                    stem = "seed_{}_{:g}bpm".format(int(record["seed"]), bpm)
                    if path.name in (stem + "_raw.wav", stem + "_dc_removed.wav"):
                        matches.append(bpm)
        except (ValueError, TypeError, KeyError, OSError):
            continue
    if len(set(matches)) > 1:
        return None, "conflicting_metadata_targets"
    return (matches[0], None) if matches else (None, None)


def compact_row(report, names):
    primary = report.get("tracks", {}).get(report.get("primary_model"), {})
    row = dict(file=report["file"], duration_seconds=report.get("audio_statistics", {}).get("duration_seconds"),
               target_bpm=report.get("target_bpm"), measured_bpm=report.get("measured_bpm"),
               status=report["status"], **target_errors(report.get("measured_bpm"), report.get("target_bpm")),
               beat_count=primary.get("beat_count"), interval_cv=primary.get("interval_cv"),
               coverage_fraction=primary.get("annotated_or_detected_span_fraction"),
               flags=";".join(report.get("flags", [])), error=report.get("error"))
    for name in names:
        row["bpm_" + name] = report.get("tracks", {}).get(name, {}).get("measured_bpm")
    return row


def write_csv(path, reports, names):
    if not reports:
        Path(path).write_text("file,measured_bpm,status\n", encoding="utf-8"); return
    rows = [compact_row(r, names) for r in reports]
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def process(path, tracker):
    try:
        return measure(path, tracker)
    except Exception as exc:
        return dict(file=str(path.resolve()), status="error", measured_bpm=None,
                    flags=["file_processing_failed"], error=type(exc).__name__ + ": " + str(exc))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, help="Default: input-dir/automatic_bpm_<backend>")
    p.add_argument("--backend", choices=("beat-this", "librosa"), default="beat-this")
    p.add_argument("--models", nargs="+", default=["final0", "final1", "final2"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4, help="PyTorch CPU thread count")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--include-raw", action="store_true", help="Include raw WAV even when its DC-corrected counterpart exists")
    p.add_argument("--reference-dir", type=Path, help="Reference audio.wav + conditioning.npz pairs; default input-dir/references if present")
    args = p.parse_args()
    if args.threads < 1:
        p.error("threads must be at least 1")
    root = args.input_dir.resolve()
    if not root.is_dir():
        p.error("input-dir does not exist: " + str(root))
    output = (args.output or root / ("automatic_bpm_" + args.backend)).resolve()
    if output.exists():
        p.error("Output already exists. Choose a new directory to preserve earlier results.")
    files, skipped = discover(root, args.recursive, args.include_raw, output)
    if not files:
        p.error("No eligible WAV, FLAC or OGG files found")
    refdir = args.reference_dir or (root / "references" if (root / "references").is_dir() else None)
    if refdir is not None and not refdir.is_dir():
        p.error("reference-dir does not exist")
    references = sorted(x for x in refdir.rglob("audio.wav") if (x.parent / "conditioning.npz").is_file()) if refdir else []
    try:
        tracker = Tracker(args)
    except Exception as exc:
        p.exit(2, "ERROR loading tracker: %s\n" % exc)
    output.mkdir(parents=True)
    (output / "per_clip").mkdir()
    packages = {}
    for name in ("numpy", "soundfile", "librosa", "torch", "torchaudio", "beat-this"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    dump(output / "run_config.json", dict(version=VERSION, python=sys.version, packages=packages,
         backend=args.backend, primary_model=tracker.primary, candidate_models=tracker.names,
         model_file_sha256=tracker.model_hashes,
         input_dir=str(root), reference_dir=str(refdir) if refdir else None,
         device=args.device, formula=FORMULA, expected_bpm_supplied_to_estimator=False,
         comparison_tolerance_percent=COMPARISON_TOLERANCE_PERCENT,
         half_double_scoring_factors=[0.5, 1.0, 2.0], raw_bpm_is_never_replaced=True,
         audio_preprocessing="DC removal; no time stretching or gain normalization; Beat This internally resamples for inference",
         diagnostics_note="5% candidate spread, CV 0.10 and coverage 75% are heuristics, not calibrated confidence",
         skipped_files=skipped))
    refs = []
    for i, path in enumerate(references, 1):
        print("Reference %d/%d: %s" % (i,len(references),path), flush=True)
        r = process(path, tracker)
        try:
            with np.load(str(path.parent / "conditioning.npz"), allow_pickle=False) as d:
                target = number(d["target_bpm"])
                expected = clean_times(d["beat_seconds"], float(sf.info(str(path)).duration))
            if len(expected) < 3 or not np.allclose(np.diff(expected), 60/target, atol=0.002, rtol=0):
                raise ValueError("Reference label does not match a constant quarter-note beat grid")
            r["target_bpm"] = target
            r["reference_grid_bpm"] = beat_stats(expected, float(sf.info(str(path)).duration))["measured_bpm"]
            for track in r.get("tracks", {}).values():
                track.update(target_errors(track["measured_bpm"], target))
                track["reference_beat_match"] = match_beats(expected, track["beat_seconds"])
        except Exception as exc:
            r.setdefault("flags", []).append("reference_label_invalid")
            r["reference_error"] = str(exc)
        refs.append(r)
    summaries = {}
    for name in tracker.names:
        valid_refs = [r for r in refs if r.get("target_bpm") is not None
                      and "reference_label_invalid" not in r.get("flags", [])]
        tracks = [r.get("tracks", {}).get(name,{}) for r in valid_refs]
        errors = [x["relative_error_percent"] for x in tracks if x.get("relative_error_percent") is not None]
        by_bpm = {}
        for bpm in sorted(set(r["target_bpm"] for r in valid_refs)):
            group = [r.get("tracks", {}).get(name, {}) for r in valid_refs if r["target_bpm"] == bpm]
            by_bpm[str(bpm)] = {
                "reference_count": len(group),
                "comparison_counts": {label: sum(x.get("tempo_comparison") == label for x in group)
                                      for label in ("strict_match", "double_time_match", "half_time_match", "mismatch")},
                "unavailable_count": sum(x.get("measured_bpm") is None for x in group)}
        summaries[name] = dict(reference_count=len(refs), valid_label_count=len(tracks),
            measured_count=len(errors), median_absolute_percentage_error=float(np.median(errors)) if errors else None,
            strict_error_over_4_percent_count=sum(x > COMPARISON_TOLERANCE_PERCENT for x in errors),
            comparison_counts={label: sum(x.get("tempo_comparison") == label for x in tracks)
                               for label in ("strict_match", "double_time_match", "half_time_match", "mismatch")},
            per_target_bpm=by_bpm,
            interpretation="Descriptive original-reference check, not proof of reliability on generated audio")
    dump(output / "reference_summary.json", summaries)
    dump(output / "reference_results.json", refs)
    write_csv(output / "reference_results.csv", refs, tracker.names)
    generated = []
    for i, path in enumerate(files, 1):
        r = process(path, tracker)
        target, warning = metadata_target(path)
        r["target_bpm"] = target
        r["target_source"] = "generation_metadata_comparison_only" if target is not None else None
        r.update(target_errors(r.get("measured_bpm"), target))
        if warning:
            r["flags"].append(warning)
        if not refs:
            r["flags"].append("no_known_label_reference_check")
        elif summaries[tracker.primary]["strict_error_over_4_percent_count"]:
            r["flags"].append("primary_tracker_misreads_some_original_references")
        if refs and summaries[tracker.primary]["measured_count"] < len(refs):
            r["flags"].append("some_reference_checks_unavailable")
        generated.append(r)
        dump(output / "per_clip" / ("%04d.json" % i), r)
        value = "UNAVAILABLE" if r.get("measured_bpm") is None else "%.3f" % r["measured_bpm"]
        print("[%d/%d] %s: raw BPM=%s; %s; %s" % (
              i,len(files),path.name,value,r["tempo_comparison"],r["status"]), flush=True)
    dump(output / "generated_results.json", generated)
    write_csv(output / "generated_results.csv", generated, tracker.names)
    lines = ["FULLY AUTOMATIC GENERATED-AUDIO TEMPO EVALUATION", "Backend: " + args.backend,
             "Primary model: " + tracker.primary, "Formula: " + FORMULA, "",
             "Comparison tolerance: 4%; half/double factors: 0.5, 1, 2", "",
             "file | requested | raw BPM | half/double error % | comparison | estimate status"]
    for r in generated:
        value = "UNAVAILABLE" if r.get("measured_bpm") is None else "%.3f" % r["measured_bpm"]
        tolerant = r.get("half_double_tolerant_error_percent")
        tolerant_text = "UNAVAILABLE" if tolerant is None else "%.3f" % tolerant
        lines.append("%s | %s | %s | %s | %s | %s" % (
            Path(r["file"]).name,str(r.get("target_bpm")),value,tolerant_text,
            r["tempo_comparison"],r["status"]))
    lines += ["", "Inspect candidate BPMs and original-reference errors before interpreting these estimates.",
              "double_time_match means raw BPM is about twice the target; half_time_match means about half.",
              "These are comparison categories, not corrected BPMs or verified musical beat levels.",
              "Half/double candidate columns are arithmetic alternatives, not independent measurements.",
              "A half/double-tolerant match does not establish strict 60-versus-120 BPM control.",
              "Agreement among models is not ground truth; all can choose the same wrong metrical level.",
              "This replaces annotation lookup with automatic beat prediction, not the original ASAP labeling process.",
              "Expected BPMs, conditioning grids and filenames never enter beat prediction.",
              "No automatic half/double correction, audio warping or best-target model selection is applied.",
              "Known-reference evaluation is a local sanity check, not a benchmark of the pretrained tracker's generalization."]
    (output / "summary.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print("Results:", output)
    if any(r["status"] == "error" for r in generated):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
