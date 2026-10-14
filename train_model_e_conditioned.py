#!/usr/bin/env python3
"""Train Model E with BPM FiLM, multi-resolution onset/beat conditioning, and scaling.

Python 3.8 / Torch 1.13 / Lightning 1.9 compatible API. Fresh diffusion weights,
frozen existing VAE; separate architecture/checkpoints. No defaults.ini reading.
The default cond-drop-prob is zero for an initial matched conditioning experiment.
Use --cond-drop-prob 0.1 for a separate CFG training experiment.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import inspect
import json
import math
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl

from audio_diffusion.vae1d import AudioVAE1D
from model_e_conditioning import ConditionedASAPDataset, CONDITION_SCHEMA
from model_e_conditioned import ConditionedModelE, ARCHITECTURE
from model_e_conditioned_runtime import get_alphas_sigmas, get_crash_schedule, sample, validate_checkpoint
from model_e_conditioned_scaling import CONVENTION, estimate_training_scale


VERSION = "conditioned-1.0.0"
BPMS = (60, 75, 90, 105, 120, 135, 150, 165, 180)
VALIDATION_TIMES = (0.15, 0.40, 0.65, 0.90)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_trusted(path):
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(str(path), **kwargs)


def stop_step(start, additional, absolute):
    stop = int(absolute) if absolute is not None else start + int(additional)
    if stop <= start:
        raise ValueError("Stopping step must exceed checkpoint global_step={}.".format(start))
    return stop


def scan_dataset(dataset):
    counts, selected, paths = Counter(), {}, set()
    for index in range(len(dataset)):
        audio, condition, name = dataset[index]
        path = Path(name).resolve()
        if path in paths:
            raise ValueError("Duplicate audio in loader: " + str(path))
        paths.add(path)
        if tuple(audio.shape) != (2, 65536) or tuple(condition.shape) != (3, 4096):
            raise ValueError("Unexpected audio/conditioning shape: " + name)
        if not torch.isfinite(audio).all() or not torch.isfinite(condition).all():
            raise ValueError("Non-finite input: " + name)
        value = 120.0 + 60.0 * float(condition[0, 0])
        bpm = int(round(value))
        if bpm not in BPMS or abs(value - bpm) > 1e-3:
            raise ValueError("Unexpected BPM {} in {}".format(value, name))
        expected = torch.full_like(condition[0], (bpm - 120.0) / 60.0)
        if not torch.allclose(condition[0], expected, atol=1e-6, rtol=0):
            raise ValueError("Nonconstant or incorrectly normalized BPM channel: " + name)
        if (condition[1:] < 0).any() or (condition[1:] > 1).any():
            raise ValueError("Timing envelope outside [0, 1]: " + name)
        counts[bpm] += 1
        selected.setdefault(bpm, (condition.detach().cpu().clone(), str(path)))
        if (index + 1) % 100 == 0:
            print("  Checked {}/{} examples".format(index + 1, len(dataset)), flush=True)
    if set(counts) != set(BPMS):
        raise ValueError("Missing requested BPMs: {}".format(sorted(set(BPMS) - set(counts))))
    return counts, selected, paths


def check_split_manifest(root, train_counts, val_counts):
    """Check recorded source separation; this is not a new composition audit."""
    manifest = root / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(str(manifest))
    counts, owners, groups, clip_ids = Counter(), {}, {}, set()
    with manifest.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            split = row["split"]
            key = (split, row["clip_id"])
            if key in clip_ids:
                raise ValueError("Duplicate manifest clip: " + str(key))
            clip_ids.add(key)
            if row.get("scan_only", False):
                raise ValueError("Manifest contains unprocessed scan-only candidates.")
            counts[(split, int(round(row["target_bpm"])))] += 1
            for field, mapping in (("source_audio", owners), ("passage_group_id", groups)):
                identity = row[field]
                previous = mapping.setdefault(identity, split)
                if previous != split:
                    raise ValueError("{} crosses splits: {}".format(field, identity))
    for split, observed in (("train", train_counts), ("validation", val_counts)):
        recorded = Counter({bpm: n for (s, bpm), n in counts.items() if s == split})
        if recorded != observed:
            raise ValueError("{} loader counts differ from manifest: {} vs {}".format(
                split, observed, recorded))
    return {split: sum(n for (s, _), n in counts.items() if s == split)
            for split in ("train", "validation", "test")}


def validation_noise(latents, filenames, level_index):
    """Fixed per-clip noise, independent of batching and training RNG."""
    values = []
    for name in filenames:
        identity = Path(name).parent.name + ":validation:" + str(level_index)
        seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little") % (2 ** 63 - 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        values.append(torch.randn(tuple(latents.shape[1:]), generator=generator))
    return torch.stack(values).to(device=latents.device, dtype=latents.dtype)


class ExpandedModelE(pl.LightningModule):
    def __init__(self, global_args):
        super().__init__()
        self.global_args = global_args
        self.ema_decay = global_args.ema_decay
        self.cond_drop_prob = global_args.cond_drop_prob
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True, seed=global_args.seed)
        self.vae = AudioVAE1D(in_channels=2, latent_channels=8, base_channels=64)
        state = load_trusted(global_args.vae_ckpt_path)["state_dict"]
        vae_state = {k[len("vae."):]: v for k, v in state.items() if k.startswith("vae.")}
        if not vae_state:
            raise ValueError("Expected a VAE checkpoint containing vae.* weights.")
        self.vae.load_state_dict(vae_state, strict=True)
        self.vae.requires_grad_(False).eval()
        self.diffusion = ConditionedModelE()
        self.diffusion_ema = deepcopy(self.diffusion).requires_grad_(False).eval()
        # Buffer is saved in checkpoints and moved with the model, never optimized.
        # main() replaces NaN with training statistics or a verified checkpoint value.
        self.register_buffer("latent_scale", torch.tensor(float("nan"), dtype=torch.float32))

    def configure_optimizers(self):
        return torch.optim.Adam(self.diffusion.parameters(), lr=4e-5)

    @torch.no_grad()
    def on_before_zero_grad(self, *args, **kwargs):
        decay = .95 if self.current_epoch < 25 else self.ema_decay
        for source, target in zip(self.diffusion.parameters(), self.diffusion_ema.parameters()):
            target.mul_(decay).add_(source, alpha=1-decay)
        for source, target in zip(self.diffusion.buffers(), self.diffusion_ema.buffers()):
            target.copy_(source)

    def on_train_start(self):
        if not torch.isfinite(self.latent_scale).item() or self.latent_scale.item() <= 0:
            raise RuntimeError("A valid training-set latent scale has not been initialized.")

    def training_step(self, batch, batch_idx):
        reals, condition, filenames = batch

        condition = condition.to(
            device=reals.device,
            dtype=reals.dtype,
        )

        # The VAE remains frozen.
        self.vae.eval()

        with torch.no_grad():
            mu, _ = self.vae.encode(reals)
            latents = mu * self.latent_scale.to(dtype=mu.dtype)

        expected_shape = (
            latents.shape[0],
            3,
            latents.shape[-1],
        )

        if tuple(condition.shape) != expected_shape:
            raise RuntimeError(
                "Expected condition {}, got {}".format(
                    expected_shape, tuple(condition.shape)
                )
            )

        t = self.rng.draw(latents.shape[0])[:, 0].to(self.device)
        t = get_crash_schedule(t)

        alphas, sigmas = get_alphas_sigmas(t)
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]

        noise = torch.randn_like(latents)

        noised_latents = latents * alphas + noise * sigmas
        targets = noise * alphas - latents * sigmas

        present = torch.ones(latents.shape[0], device=latents.device, dtype=latents.dtype)
        if self.cond_drop_prob > 0:
            present = (torch.rand_like(present) >= self.cond_drop_prob).to(latents.dtype)
        with torch.cuda.amp.autocast(enabled=latents.is_cuda):
            prediction = self.diffusion(
                noised_latents,
                t,
                cond=condition,
                condition_present=present,
            )

            loss = F.mse_loss(
                prediction.float(),
                targets.float(),
            )

        if not torch.isfinite(loss):
            raise RuntimeError("Training loss became non-finite.")

        self.log_dict(
            {
                "train/loss": loss.detach(),
                "train/mse_loss": loss.detach(),
            },
            prog_bar=True,
            on_step=True,
        )

        return loss

    def on_validation_epoch_start(self):
        self._validation_losses = defaultdict(list)
        self.vae.eval()
        self.diffusion_ema.eval()

    def validation_step(self, batch, batch_idx):
        audio, condition, filenames = batch
        self.vae.eval()
        self.diffusion_ema.eval()
        with torch.no_grad():
            mu, _ = self.vae.encode(audio)
            latents = mu * self.latent_scale.to(dtype=mu.dtype)
        condition = condition.to(device=latents.device, dtype=latents.dtype)
        per_level = []
        for level_index, value in enumerate(VALIDATION_TIMES):
            t = get_crash_schedule(torch.full((latents.shape[0],), value, device=latents.device))
            alpha, sigma = get_alphas_sigmas(t)
            alpha, sigma = alpha[:, None, None], sigma[:, None, None]
            noise = validation_noise(latents, filenames, level_index)
            noised = alpha * latents + sigma * noise
            target = alpha * noise - sigma * latents
            prediction = self.diffusion_ema(noised, t, cond=condition)
            loss = F.mse_loss(prediction.float(), target.float(), reduction="none")
            per_level.append(loss.mean(dim=(1, 2)))
        losses = torch.stack(per_level).mean(dim=0)
        if not torch.isfinite(losses).all():
            raise RuntimeError("Validation loss became non-finite.")
        bpms = torch.round(120.0 + 60.0 * condition[:, 0, 0]).long().cpu().tolist()
        for bpm, loss in zip(bpms, losses.detach().cpu().tolist()):
            self._validation_losses[bpm].append(loss)

    def on_validation_epoch_end(self):
        if not self._validation_losses:
            return
        means = {bpm: sum(values) / len(values) for bpm, values in self._validation_losses.items()}
        all_losses = [v for values in self._validation_losses.values() for v in values]
        metrics = {"val/ema_v_mse": sum(all_losses) / len(all_losses),
                   "val/ema_v_mse_macro": sum(means.values()) / len(means)}
        metrics.update({"val/bpm_{:03d}_mse".format(bpm): value for bpm, value in means.items()})
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False)
        print("\nValidation global_step={}: EMA MSE={:.6f}; equal-BPM mean={:.6f}".format(
            self.global_step, metrics["val/ema_v_mse"], metrics["val/ema_v_mse_macro"]), flush=True)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["model_e_expanded_run"] = self.run_metadata
        checkpoint["model_e_sobol_num_generated"] = int(self.rng.num_generated)

    def on_load_checkpoint(self, checkpoint):
        validate_checkpoint(checkpoint)
        count = checkpoint.get("model_e_sobol_num_generated")
        if count is not None:
            self.rng.reset()
            if int(count) > 0:
                self.rng.fast_forward(int(count))


class ValidationDemos(pl.Callback):
    def __init__(self, selected, output, every, steps, seed):
        super().__init__()
        self.selected, self.output = selected, Path(output)
        self.every, self.steps, self.seed = every, steps, seed
        self.last_step = -1

    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if not self.every or not step or step % self.every or step == self.last_step:
            return
        self.last_step = step
        folder = self.output / "demos" / "step_{:08d}".format(step)
        folder.mkdir(parents=True, exist_ok=True)
        previous_ema_mode = module.diffusion_ema.training
        previous_vae_mode = module.vae.training
        module.diffusion_ema.eval()
        module.vae.eval()
        records = []
        try:
            # Sequential generation: one clip in GPU memory, nine target BPMs.
            for bpm, (condition, reference) in sorted(self.selected.items()):
                generator = torch.Generator(device=module.device).manual_seed(self.seed)
                noise = torch.randn((1, 8, 4096), generator=generator, device=module.device)
                cond = condition.unsqueeze(0).to(module.device)
                latent = sample(module.diffusion_ema, noise, self.steps, 0.0,
                                cond=cond, guidance_scale=1.0)
                raw = module.vae.decode(latent / module.latent_scale).float()[0].cpu()
                if not torch.isfinite(raw).all():
                    raise RuntimeError("Non-finite demo at BPM {}".format(bpm))
                corrected = raw - raw.mean(dim=-1, keepdim=True)
                stem = "seed_{}_{}bpm".format(self.seed, bpm)
                sf.write(str(folder / (stem + "_raw.wav")), raw.T.numpy(), 8000, subtype="FLOAT")
                sf.write(str(folder / (stem + "_dc_removed.wav")), corrected.T.numpy(), 8000, subtype="FLOAT")
                records.append({"requested_bpm": bpm, "reference": reference, "seed": self.seed})
        finally:
            module.diffusion_ema.train(previous_ema_mode)
            module.vae.train(previous_vae_mode)
        write_json(folder / "demo_metadata.json", {"global_step": step, "duration_seconds": 8.192,
                   "conditions_from": module.run_metadata["demo_conditions_from"], "clips": records,
                   "latent_convention": CONVENTION, "latent_scale": float(module.latent_scale.cpu()),
                   "note": "Requested BPM and onset timing are not measured output tempo."})
        print("Saved {} conditioned demos:".format(len(records)), folder, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="Parent containing train/, validation/, manifest.jsonl")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--from-scratch", action="store_true",
                      help="New diffusion weights/EMA/optimizer at step 0; keep pretrained VAE")
    mode.add_argument("--resume", type=Path, help="Full CONDITIONED Lightning checkpoint from this launcher")
    parser.add_argument("--vae-ckpt", type=Path, required=True)
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument("--additional-steps", type=int, help="Updates to add to checkpoint global_step")
    budget.add_argument("--stop-at-step", type=int, help="Absolute global step to stop at")
    parser.add_argument("--output", type=Path, required=True, help="New directory")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--validate-every-epochs", type=int, default=5)
    parser.add_argument("--demo-every", type=int, default=3000, help="Global-step interval; 0 disables")
    parser.add_argument("--demo-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cond-drop-prob", type=float, default=None,
                        help="Fresh default 0; test 0.1 for CFG. Resume restores the saved value.")
    parser.add_argument("--debug-clips-per-bpm", type=int, default=0,
                        help="0=all training clips; 1=one training clip per BPM for a nine-clip overfit check")
    parser.add_argument("--trusted-checkpoint", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Check all data, estimate or restore scale, validate checkpoint; no training/output")
    args = parser.parse_args()
    if not args.trusted_checkpoint:
        parser.error("Use --trusted-checkpoint only for checkpoints you trust.")
    if min(args.batch_size, args.checkpoint_every, args.validate_every_epochs, args.demo_steps) < 1:
        parser.error("Batch size and training intervals must be positive.")
    if args.cond_drop_prob is not None and not 0 <= args.cond_drop_prob < 1:
        parser.error("cond-drop-prob must be in [0,1).")
    if args.debug_clips_per_bpm < 0:
        parser.error("debug-clips-per-bpm cannot be negative.")
    if args.demo_every < 0 or args.num_workers < 0:
        parser.error("demo-every and num-workers cannot be negative.")
    for name in ("dataset_root", "resume", "vae_ckpt", "output"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    for path in ([args.resume] if args.resume else []) + [args.vae_ckpt]:
        if not path.is_file():
            parser.error("File does not exist: " + str(path))
    if not args.check_only and args.output.exists():
        parser.error("Output already exists; use a new directory, including for continuation.")
    return args


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    print("Checking every training and validation example...", flush=True)
    full_train_set = ConditionedASAPDataset(str(args.dataset_root / "train"))
    train_set = full_train_set
    val_set = ConditionedASAPDataset(str(args.dataset_root / "validation"))
    train_counts, _, train_paths = scan_dataset(train_set)
    val_counts, demo_conditions, val_paths = scan_dataset(val_set)
    if train_paths & val_paths:
        raise ValueError("Training and validation share audio files.")
    split_counts = check_split_manifest(args.dataset_root, train_counts, val_counts)
    print("Clips:", split_counts, flush=True)
    if args.debug_clips_per_bpm:
        indices, picked = [], Counter()
        for index, path in enumerate(full_train_set.filenames):
            bpm = int(path.parent.parent.name.split("_")[-1])
            if picked[bpm] < args.debug_clips_per_bpm:
                indices.append(index)
                picked[bpm] += 1
        if any(picked[b] != args.debug_clips_per_bpm for b in BPMS):
            raise ValueError("Not enough clips for the requested balanced debug subset.")
        train_set = Subset(full_train_set, indices)
        train_counts, demo_conditions, train_paths = scan_dataset(train_set)
    training_selection = sorted(str(p) for p in train_paths)
    selection_hash = hashlib.sha256("\n".join(training_selection).encode()).hexdigest()
    print("Selected training clips per BPM:", dict(sorted(train_counts.items())), flush=True)

    checkpoint = None if args.from_scratch else load_trusted(args.resume)
    start = 0 if checkpoint is None else int(checkpoint["global_step"])
    stop = stop_step(start, args.additional_steps, args.stop_at_step)
    if checkpoint is not None:
        if not checkpoint.get("optimizer_states"):
            raise ValueError("Resume checkpoint has no optimizer state; full continuation is unavailable.")
        state = checkpoint["state_dict"]
        saved_scale = validate_checkpoint(checkpoint)
        previous_metadata = checkpoint["model_e_expanded_run"]
        saved_dropout = float(previous_metadata["cond_drop_prob"])
        if args.cond_drop_prob is not None and args.cond_drop_prob != saved_dropout:
            raise ValueError("Resume must keep cond-drop-prob; use a fresh run for the CFG ablation.")
        args.cond_drop_prob = saved_dropout
        if previous_metadata.get("training_selection_sha256") != selection_hash:
            raise ValueError("Resume requires the same selected training examples.")
        if previous_metadata.get("dataset_manifest_sha256") != file_hash(args.dataset_root / "manifest.jsonl"):
            raise ValueError("Dataset manifest changed. Resume requires the same scaling experiment.")
        if int(previous_metadata.get("seed", -1)) != args.seed or int(previous_metadata.get("batch_size", -1)) != args.batch_size:
            raise ValueError("Resume with the original --seed and --batch-size.")

    if args.cond_drop_prob is None:
        args.cond_drop_prob = 0.0
    model_args = SimpleNamespace(
        seed=args.seed, ema_decay=0.995, latent_channels=8, vae_base_channels=64,
        vae_ckpt_path=str(args.vae_ckpt), latent_dim=3, sample_rate=8000,
        sample_size=65536, cond_drop_prob=args.cond_drop_prob, min_bpm=60, max_bpm=180,
        use_tempo_conditioning=True, training_dir=str(args.dataset_root / "train"),
        save_path=str(args.output), batch_size=args.batch_size, accum_batches=1)
    model = ExpandedModelE(model_args)
    if checkpoint is not None:
        embedded_vae = {key[len("vae."):]: value for key, value in state.items() if key.startswith("vae.")}
        current_vae = model.vae.state_dict()
        if set(embedded_vae) != set(current_vae) or any(
                not torch.equal(embedded_vae[key], value) for key, value in current_vae.items()):
            raise ValueError("Supplied VAE differs from the frozen VAE in the diffusion checkpoint.")
        # Fail now on incompatible model/code, before starting a GPU run.
        model.load_state_dict(state, strict=True)
        latent_statistics = previous_metadata["latent_statistics"]
        if not math.isclose(float(model.latent_scale), saved_scale, rel_tol=1e-6):
            raise ValueError("Latent scale changed while loading.")
        lr = float(checkpoint["optimizer_states"][0]["param_groups"][0]["lr"])
        del state, embedded_vae, checkpoint
    else:
        # The constructor creates the diffusion and EMA from random initialization.
        # Read the existing model's configured learning rate without changing it.
        lr = float(model.configure_optimizers().param_groups[0]["lr"])
    if args.from_scratch:
        print("Estimating one latent scale from every TRAINING clip...", flush=True)
        stats_device = "cuda" if torch.cuda.is_available() else "cpu"
        latent_statistics = estimate_training_scale(model.vae, full_train_set,
                                                    batch_size=args.batch_size, device=stats_device)
        latent_statistics["dataset_manifest_sha256"] = file_hash(args.dataset_root / "manifest.jsonl")
        latent_statistics["vae_checkpoint_sha256"] = file_hash(args.vae_ckpt)
        model.latent_scale.fill_(latent_statistics["scale"])
    print("Training latent std: {:.6f}; fixed scale: {:.8f}; scaled std: {:.6f}".format(
        latent_statistics["population_std"], float(model.latent_scale),
        latent_statistics["scaled_population_std"]), flush=True)
    print("Latent mean before scaling: {:.6f}; centering=False".format(latent_statistics["mean"]), flush=True)
    batches = int(math.ceil(len(train_set) / args.batch_size))
    updates = stop - start
    print("Initialization:", "fresh diffusion + pretrained frozen VAE" if args.from_scratch else "full checkpoint resume", flush=True)
    print("Starting global step: {}".format(start), flush=True)
    print("Additional optimizer steps: {}".format(updates), flush=True)
    print("Stopping global step: {}".format(stop), flush=True)
    print("{} steps/epoch; about {:.2f} additional dataset passes".format(batches, updates / batches), flush=True)
    print("Optimizer learning rate:", lr, flush=True)
    if args.check_only:
        print("CHECK PASSED. No training started and no output directory created.", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training run; --check-only works on CPU.")

    args.output.mkdir(parents=True)
    metadata = dict(version=VERSION, dataset_root=str(args.dataset_root),
                    initialization="fresh_diffusion_pretrained_frozen_vae" if args.from_scratch else "full_resume",
                    resume_checkpoint=str(args.resume) if args.resume else None,
                    resume_checkpoint_sha256=file_hash(args.resume) if args.resume else None,
                    vae_checkpoint=str(args.vae_ckpt), starting_global_step=start,
                    stopping_global_step=stop, additional_steps=updates, batch_size=args.batch_size,
                    gradient_accumulation=1, steps_per_epoch=batches, split_counts=split_counts,
                    training_bpm_counts=dict(sorted(train_counts.items())),
                    validation_bpm_counts=dict(sorted(val_counts.items())),
                    training_sampling="selected debug clips, shuffled, drop_last=False" if args.debug_clips_per_bpm else "all clips, shuffled, drop_last=False",
                    latent_convention=CONVENTION, latent_scale=float(model.latent_scale),
                    latent_statistics=latent_statistics,
                    dataset_manifest_sha256=file_hash(args.dataset_root / "manifest.jsonl"),
                    condition_channels=3, architecture=ARCHITECTURE, condition_schema=CONDITION_SCHEMA,
                    cond_drop_prob=args.cond_drop_prob, debug_clips_per_bpm=args.debug_clips_per_bpm,
                    training_selection_sha256=selection_hash,
                    demo_conditions_from="train_debug" if args.debug_clips_per_bpm else "validation",
                    bpm_encoding="(bpm - 120) / 60", vae_frozen=True,
                    seed=args.seed, optimizer_lr=lr, validate_every_epochs=args.validate_every_epochs,
                    checkpoint_every_global_steps=args.checkpoint_every, demo_every_global_steps=args.demo_every,
                    validation_noise_times_before_crash_schedule=VALIDATION_TIMES,
                    validation_metric="EMA v-prediction MSE; macro mean weights each BPM equally",
                    validation_metric_is_measured_bpm=False,
                    torch_version=torch.__version__, lightning_version=pl.__version__)
    model.run_metadata = metadata
    write_json(args.output / "run_config.json", metadata)
    write_json(args.output / "latent_statistics.json", latent_statistics)
    shutil.copy2(inspect.getfile(estimate_training_scale), str(args.output / "model_e_conditioned_scaling.py"))
    for name in ("summary.json", "run_config.json", "manifest.jsonl", "split_conflicts.json"):
        source = args.dataset_root / name
        if source.is_file():
            shutil.copy2(str(source), str(args.output / ("dataset_" + name)))
    shutil.copy2(__file__, str(args.output / "launcher_snapshot.py"))
    for source_class in (ConditionedModelE, ConditionedASAPDataset, AudioVAE1D):
        source_path = Path(inspect.getfile(source_class))
        shutil.copy2(str(source_path), str(args.output / source_path.name))
    shutil.copy2(inspect.getfile(sample), str(args.output / "model_e_conditioned_runtime.py"))
    import audio_diffusion.blocks as backbone_blocks
    shutil.copy2(inspect.getfile(backbone_blocks), str(args.output / "backbone_blocks_snapshot.py"))
    write_json(args.output / "selected_training_files.json", training_selection)
    reference_dir = args.output / "demo_references"
    for bpm, (_, audio_path) in sorted(demo_conditions.items()):
        dest = reference_dir / ("bpm_%03d" % bpm)
        dest.mkdir(parents=True)
        for name in ("audio.wav", "conditioning.npz", "performance.mid"):
            source = Path(audio_path).with_name(name)
            if source.is_file():
                shutil.copy2(str(source), str(dest / name))

    loader_options = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                          persistent_workers=args.num_workers > 0, pin_memory=True, drop_last=False)
    train_dl = DataLoader(train_set, shuffle=True, **loader_options)
    val_dl = DataLoader(val_set, shuffle=False, **loader_options)
    regular = pl.callbacks.ModelCheckpoint(
        dirpath=str(args.output), filename="model_e-step{step:08d}", auto_insert_metric_name=False,
        every_n_train_steps=args.checkpoint_every, save_top_k=-1, save_last=True,
        save_on_train_epoch_end=False)
    best = pl.callbacks.ModelCheckpoint(
        dirpath=str(args.output / "best_validation"), filename="best-step{step:08d}",
        auto_insert_metric_name=False, monitor="val/ema_v_mse_macro", mode="min",
        save_top_k=1, save_last=False, save_on_train_epoch_end=False)
    callbacks = [regular, best]
    if args.demo_every:
        callbacks.append(ValidationDemos(demo_conditions, args.output, args.demo_every, args.demo_steps, args.seed))
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, precision=16,
        max_epochs=-1, max_steps=stop, accumulate_grad_batches=1,
        callbacks=callbacks, logger=pl.loggers.CSVLogger(str(args.output), name="logs"),
        check_val_every_n_epoch=args.validate_every_epochs, val_check_interval=1.0,
        num_sanity_val_steps=0, log_every_n_steps=10)
    trainer.fit(model, train_dl, val_dl, ckpt_path=str(args.resume) if args.resume else None)
    # Explicitly save the exact final step, even if not divisible by checkpoint_every.
    trainer.save_checkpoint(str(args.output / "last.ckpt"))
    write_json(args.output / "completion.json", {"global_step": int(trainer.global_step),
               "requested_stop": stop, "completed": int(trainer.global_step) == stop})
    print("Saved final checkpoint:", args.output / "last.ckpt", flush=True)


if __name__ == "__main__":
    main()
