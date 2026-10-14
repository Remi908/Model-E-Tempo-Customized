#!/usr/bin/env python3
"""Controlled tempo sweep from a train_model_e_conditioned.py checkpoint.

Python 3.8 compatible. Save beside the model_e_conditioned*.py modules.
--seconds 8.192 (default): same 65536-sample length as training.
--seconds 30: generate 262144 samples in one pass, then save 240000 samples.
The same MIDI onset motif is scaled in beat units at every target BPM.
Eight-beat motif repetition is in the CONDITION only; audio is never looped
or time-stretched. This tests joint BPM/onset/beat control, not scalar-only
control or guaranteed preservation of melody. Saved timing is a request,
not measured audio BPM. Identical seeds share initial noise within a sweep.
Use --prepare-only to check conditions without loading checkpoint weights.
"""
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from model_e_conditioned_scaling import CONVENTION
from model_e_conditioning import make_condition, CONDITION_SCHEMA
from model_e_conditioned_runtime import sample, validate_checkpoint, require_guidance_training

VERSION = "conditioned-tempo-sweep-1.0.0"
SR = 8000
TRAIN_SAMPLES = 65536
GENERATE_SAMPLES = TRAIN_SAMPLES
LATENT_LENGTH = GENERATE_SAMPLES // 16
SAVE_SAMPLES = TRAIN_SAMPLES
GEN_SECONDS = GENERATE_SAMPLES / SR
REPO = Path("/data/VAE/HarmonAI/sample-generator-main-1")
DATA = Path("/data/VAE/maestro_fresh/maestro_asap_9bpm_v1/validation")


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def times(values, name, duration, minimum=0):
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1 or len(v) < minimum or not np.isfinite(v).all():
        raise ValueError("Invalid " + name)
    if np.any(v < 0) or np.any(v >= duration) or np.any(np.diff(v) <= 0):
        raise ValueError(name + " must be unique, increasing, and inside the clip.")
    return v


def select_reference_folder(args, source_bpm):
    root = args.dataset_root / ("bpm_%03d" % source_bpm)
    if args.passage_group_id != "auto":
        return root / ("passage_%s_bpm_%03d" % (args.passage_group_id, source_bpm))
    # Select by filename and available timing context, never by generated quality.
    for source in sorted(root.glob("passage_*/conditioning.npz")):
        if not source.with_name("audio.wav").is_file():
            continue
        with np.load(str(source), allow_pickle=False) as data:
            beats = times(data["beat_seconds"], "candidate beats", TRAIN_SAMPLES / SR, 3)
            onsets = times(data["onset_seconds"], "candidate onsets", TRAIN_SAMPLES / SR, 1)
        if args.events_json or (len(beats) > args.loop_beats and np.any(
                (onsets >= beats[0]) &
                (onsets < beats[0] + args.loop_beats * 60.0 / source_bpm))):
            return source.parent
    raise ValueError("No usable reference with sufficient beats/onsets in " + str(root))


def prepare_reference(args, bpm, output):
    source_bpm = args.reference_bpm if args.reference_bpm is not None else bpm
    folder = select_reference_folder(args, source_bpm)
    source = folder / "conditioning.npz"
    audio_path = folder / "audio.wav"
    with np.load(str(source), allow_pickle=False) as data:
        stored_bpm = float(data["target_bpm"])
        onsets = times(data["onset_seconds"], "source onsets", TRAIN_SAMPLES / SR, 1)
        beats = times(data["beat_seconds"], "source beats", TRAIN_SAMPLES / SR, 3)
        old_condition = np.asarray(data["condition"], dtype=np.float32)
    if stored_bpm != source_bpm or old_condition.shape != (2, 4096):
        raise ValueError("Unexpected source BPM/shape: " + str(source))
    period = 60.0 / bpm
    source_period = 60.0 / source_bpm
    if not np.allclose(np.diff(beats), source_period, atol=2.0 / SR, rtol=0):
        raise ValueError("Source annotated beat grid is not constant at its label.")
    if not np.allclose(old_condition, make_condition(onsets, beats, source_bpm, 4096)[:2], atol=2e-6, rtol=0):
        raise ValueError("Source condition differs from the expected 10-ms Gaussian convention.")
    info = sf.info(str(audio_path))
    if (info.samplerate, info.channels, info.frames) != (SR, 2, TRAIN_SAMPLES):
        raise ValueError("Source audio must be stereo, 8 kHz, 65536 samples.")

    ref_dir = output / "references" / ("bpm_%03d" % source_bpm)
    if ref_dir.exists():
        for name in ("audio.wav", "conditioning.npz"):
            if sha256(ref_dir / name) != sha256(folder / name):
                raise ValueError("Conflicting source reference: " + str(ref_dir))
    else:
        ref_dir.mkdir(parents=True)
        for name in ("audio.wav", "conditioning.npz"):
            shutil.copy2(str(folder / name), str(ref_dir / name))
        if (folder / "performance.mid").is_file():
            shutil.copy2(str(folder / "performance.mid"), str(ref_dir / "performance.mid"))

    if args.events_json:
        events = json.loads(args.events_json.read_text())
        if float(events["target_bpm"]) != bpm or float(events["duration_seconds"]) < GEN_SECONDS:
            raise ValueError("Events must match BPM and cover at least {} seconds.".format(GEN_SECONDS))
        supplied_duration = float(events["duration_seconds"])
        extended_onsets = times(events["onset_seconds"], "long onsets", supplied_duration, 1)
        extended_beats = times(events["beat_seconds"], "long beats", supplied_duration, 3)
        extended_onsets = extended_onsets[extended_onsets < GEN_SECONDS]
        extended_beats = extended_beats[extended_beats < GEN_SECONDS]
        if len(extended_onsets) < 1 or len(extended_beats) < 3:
            raise ValueError("Long events do not cover the generation window.")
        if (extended_beats[0] > period or GEN_SECONDS - extended_beats[-1] > period + 1e-6
                or not np.allclose(np.diff(extended_beats), period, atol=2.0 / SR, rtol=0)):
            raise ValueError("Long beat grid must cover the whole timeline at the target BPM.")
        basis = "user_supplied_long_timing_events"
        loop_info = {"events_json_sha256": sha256(args.events_json)}
    else:
        if len(beats) <= args.loop_beats:
            raise ValueError("Not enough annotated beats for the requested loop length.")
        anchor = float(beats[0])
        loop_seconds = args.loop_beats * period
        source_loop_seconds = args.loop_beats * source_period
        event_time_scale = source_bpm / float(bpm)
        # Scale onset offsets in beat units; retain the first-beat phase.
        motif = (onsets[(onsets >= anchor) &
                        (onsets < anchor + source_loop_seconds)] - anchor) * event_time_scale
        if len(motif) == 0:
            raise ValueError("The selected beat interval has no MIDI note onsets.")
        pickup = anchor + (onsets[onsets < anchor] - anchor) * event_time_scale
        repeated = [pickup[pickup >= 0]]
        for k in range(int(math.ceil(GEN_SECONDS / loop_seconds)) + 1):
            repeated.append(anchor + k * loop_seconds + motif)
        extended_onsets = np.unique(np.concatenate(repeated))
        extended_onsets = extended_onsets[extended_onsets < GEN_SECONDS]
        extended_beats = anchor + np.arange(int(math.ceil(GEN_SECONDS / period)) + 1) * period
        extended_beats = extended_beats[extended_beats < GEN_SECONDS]
        basis = "ASAP_constant_beat_grid_with_repeated_MIDI_onset_excerpt"
        if args.reference_bpm is not None:
            basis = "shared_MIDI_onset_motif_scaled_to_requested_BPM"
        loop_info = dict(loop_beats=args.loop_beats, loop_seconds=loop_seconds,
                         source_loop_seconds=source_loop_seconds, event_time_scale=event_time_scale,
                         source_start_seconds=anchor, motif_onset_count=len(motif))

    condition = make_condition(extended_onsets, extended_beats, bpm, LATENT_LENGTH)
    condition_path = output / ("conditioning_%03d.npz" % bpm)
    np.savez_compressed(str(condition_path), condition=condition,
                        onset_seconds=extended_onsets, beat_seconds=extended_beats,
                        target_bpm=np.float64(bpm), sample_rate=np.int64(SR),
                        generated_samples=np.int64(GENERATE_SAMPLES), saved_samples=np.int64(SAVE_SAMPLES))
    return dict(target_bpm=bpm, conditioning_file=condition_path.name,
                conditioning_sha256=sha256(condition_path), label_basis=basis,
                timing_reference_role="requested timing, not measured generated tempo",
                source_audio=str((ref_dir / "audio.wav").relative_to(output)),
                source_conditioning=str((ref_dir / "conditioning.npz").relative_to(output)),
                source_audio_sha256=sha256(audio_path), source_conditioning_sha256=sha256(source),
                source_original_path=str(audio_path), source_reference_bpm=source_bpm,
                source_passage=folder.name,
                conditioning_scope="joint scalar BPM, MIDI onset schedule and beat grid; not scalar-only",
                **loop_info), condition


def load_checkpoint(path):
    import torch
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(str(path), **kwargs)


def load_models(args):
    import torch
    from model_e_conditioned import ConditionedModelE, ARCHITECTURE
    from audio_diffusion.vae1d import AudioVAE1D
    ckpt = load_checkpoint(args.diffusion_ckpt)
    state = ckpt["state_dict"]
    latent_scale = validate_checkpoint(ckpt)
    metadata = ckpt["model_e_expanded_run"]
    require_guidance_training(metadata, args.guidance_scale)
    ema = {k[len("diffusion_ema."):]: v for k, v in state.items() if k.startswith("diffusion_ema.")}
    vae_state = {k[len("vae."):]: v for k, v in state.items() if k.startswith("vae.")}
    if not ema or not vae_state:
        raise ValueError("Checkpoint must contain diffusion_ema.* and its frozen vae.* weights.")
    if args.vae_ckpt:
        external = load_checkpoint(args.vae_ckpt)["state_dict"]
        external = {k[len("vae."):]: v for k, v in external.items() if k.startswith("vae.")}
        if set(external) != set(vae_state) or any(not torch.equal(external[k], vae_state[k]) for k in vae_state):
            raise ValueError("External VAE weights differ from the diffusion checkpoint's VAE.")
    model = ConditionedModelE()
    vae = AudioVAE1D(in_channels=2, latent_channels=8, base_channels=64)
    model.load_state_dict(ema, strict=True)
    vae.load_state_dict(vae_state, strict=True)
    info = dict(diffusion_checkpoint=str(args.diffusion_ckpt.resolve()),
                checkpoint_sha256=sha256(args.diffusion_ckpt),
                checkpoint_step=ckpt.get("global_step"), checkpoint_epoch=ckpt.get("epoch"),
                weights="EMA", vae_weights="embedded frozen VAE",
                latent_convention=CONVENTION, latent_scale=latent_scale,
                architecture=ARCHITECTURE, condition_schema=CONDITION_SCHEMA,
                cond_drop_prob=metadata["cond_drop_prob"], guidance_scale=args.guidance_scale)
    del ckpt, state, ema, vae_state
    return model.eval().to(args.device), vae.eval().to(args.device), info


def configure_duration(seconds):
    global GENERATE_SAMPLES, LATENT_LENGTH, SAVE_SAMPLES, GEN_SECONDS
    if seconds == 8.192:
        GENERATE_SAMPLES = TRAIN_SAMPLES
        SAVE_SAMPLES = TRAIN_SAMPLES
    elif seconds == 30.0:
        GENERATE_SAMPLES = TRAIN_SAMPLES * 4
        SAVE_SAMPLES = 30 * SR
    else:
        raise ValueError("Supported durations are 8.192 and 30 seconds.")
    LATENT_LENGTH = GENERATE_SAMPLES // 16
    GEN_SECONDS = GENERATE_SAMPLES / SR


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diffusion-ckpt", type=Path, required=True)
    p.add_argument("--vae-ckpt", type=Path, help="Optional: verify this VAE equals the embedded VAE.")
    p.add_argument("--dataset-root", type=Path, default=DATA)
    p.add_argument("--passage-group-id", default="auto",
                   help="Passage identifier, or auto to select the first usable passage.")
    p.add_argument("--reference-bpm", type=int, default=120,
                   help="Use one source tempo/motif for all targets; use 120 for a nine-tempo sweep.")
    p.add_argument("--bpms", nargs="+", type=int, default=[60, 75, 90, 105, 120, 135, 150, 165, 180])
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--seconds", type=float, choices=[8.192, 30.0], default=8.192,
                   help="8.192 for the training-length test; 30 for a separate long-audio test")
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--loop-beats", type=int, default=8)
    p.add_argument("--events-json", type=Path, help="Optional continuous long events; one BPM only. See docstring/output run config.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--trusted-checkpoint", action="store_true", help="Allow loading your own trusted Lightning checkpoint.")
    p.add_argument("--prepare-only", action="store_true", help="Validate/copy references and build conditions without loading a checkpoint.")
    args = p.parse_args()
    configure_duration(args.seconds)
    if not math.isfinite(args.guidance_scale) or args.guidance_scale < 0:
        p.error("guidance-scale must be finite and nonnegative.")
    if args.steps < 1 or args.loop_beats < 1 or len(set(args.bpms)) != len(args.bpms) or len(set(args.seeds)) != len(args.seeds):
        p.error("Use positive steps/loop length and unique BPMs/seeds.")
    if any(not 60 <= b <= 180 for b in args.bpms) or any(s < 0 for s in args.seeds):
        p.error("BPMs must be between 60 and 180; seeds must be nonnegative.")
    if args.reference_bpm is not None and not 60 <= args.reference_bpm <= 180:
        p.error("reference-bpm must be between 60 and 180.")
    if args.events_json and len(args.bpms) != 1:
        p.error("--events-json requires exactly one --bpms value.")
    if args.output.exists():
        p.error("Output already exists; choose a new directory.")
    if not args.prepare_only and not args.trusted_checkpoint:
        p.error("Pass --trusted-checkpoint for your own checkpoint.")
    if not args.prepare_only and not args.diffusion_ckpt.is_file():
        p.error("Checkpoint does not exist: " + str(args.diffusion_ckpt))
    if not args.prepare_only:
        import torch
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            p.error("CUDA is unavailable. Run in your training environment on a GPU node.")
    args.output.mkdir(parents=True)
    shutil.copy2(__file__, str(args.output / "generator_snapshot.py"))
    dump(args.output / "run_config.json", dict(version=VERSION, sample_rate=SR,
         generated_samples=GENERATE_SAMPLES, generated_seconds=GEN_SECONDS, saved_seconds=SAVE_SAMPLES / SR,
         latent_shape=[1, 8, LATENT_LENGTH], condition_shape=[1, 3, LATENT_LENGTH],
         generation="single diffusion pass; no audio stitching or time stretching", steps=args.steps,
         guidance_scale=args.guidance_scale, eta=0.0, arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}))
    prepared = [prepare_reference(args, bpm, args.output) for bpm in args.bpms]
    if args.prepare_only:
        dump(args.output / "prepared_references.json", [item[0] for item in prepared])
        print("Prepared conditions:", args.output)
        return
    model, vae, checkpoint_info = load_models(args)
    dump(args.output / "checkpoint_info.json", checkpoint_info)
    import torch
    for reference, condition in prepared:
        bpm = reference["target_bpm"]
        c = torch.from_numpy(condition).unsqueeze(0).to(args.device)
        for seed in args.seeds:
            print("Generating %d BPM, seed %d: one %.3f-s pass" % (bpm, seed, GEN_SECONDS), flush=True)
            generator = torch.Generator(device=args.device).manual_seed(seed)
            noise = torch.randn((1, 8, LATENT_LENGTH), generator=generator, device=args.device)
            with torch.no_grad():
                latent = sample(model, noise, args.steps, 0.0, cond=c, guidance_scale=args.guidance_scale)
                raw = vae.decode(latent.float() / checkpoint_info["latent_scale"]).float()
            if tuple(raw.shape) != (1, 2, GENERATE_SAMPLES) or not torch.isfinite(raw).all():
                raise RuntimeError("Decoder produced incorrect shape or non-finite values.")
            raw = raw[0, :, :SAVE_SAMPLES].cpu().numpy().T.copy()
            corrected = raw - raw.mean(axis=0, keepdims=True)
            stem = "seed_%d_%dbpm" % (seed, bpm)
            raw_path = args.output / (stem + "_raw.wav")
            wav_path = args.output / (stem + "_dc_removed.wav")
            sf.write(str(raw_path), raw, SR, subtype="FLOAT")
            sf.write(str(wav_path), corrected, SR, subtype="FLOAT")
            dump(args.output / (stem + "_metadata.json"), dict(
                schema_version=VERSION, status="generated", **reference, **checkpoint_info,
                seed=seed, sampler_steps=args.steps, raw_audio=raw_path.name, generated_audio=wav_path.name,
                generated_audio_sha256=sha256(wav_path), duration_seconds=SAVE_SAMPLES / SR, sample_rate=SR,
                generated_samples=GENERATE_SAMPLES, saved_samples=SAVE_SAMPLES,
                dc_removed_per_channel=raw.mean(axis=0).astype(float).tolist(),
                postprocessing="save first {} seconds; DC removal only; no clipping, gain matching or tempo correction".format(SAVE_SAMPLES / SR)))
            del latent, raw, corrected, noise
            print("Saved:", wav_path, flush=True)
    dump(args.output / "completion.json", dict(completed=True,
         clip_count=len(args.bpms) * len(args.seeds), saved_seconds=SAVE_SAMPLES / SR,
         bpms=args.bpms, seeds=args.seeds))
    print("Finished. Evaluate this directory:", args.output)


if __name__ == "__main__":
    main()
