"""Fixed training-set latent scale for Model E; Python 3.8 / Torch 1.13.

No mean subtraction, per-clip normalization, posterior sampling, or VAE updates.
"""
import math

CONVENTION = "mu_times_fixed_training_global_std_inverse_v1"


class PopulationMoments:
    """Merge population moments without retaining the whole dataset."""
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def merge(self, count, mean, variance):
        count = int(count)
        mean, variance = float(mean), float(variance)
        if count < 1 or not math.isfinite(mean) or not math.isfinite(variance) or variance < 0:
            raise ValueError("Invalid latent moments.")
        total = self.count + count
        delta = mean - self.mean
        self.m2 += count * variance + delta * delta * self.count * count / total
        self.mean += delta * count / total
        self.count = total

    def result(self):
        if self.count < 2:
            raise ValueError("Too few latent values for a scale estimate.")
        std = math.sqrt(self.m2 / self.count)
        if not math.isfinite(std) or std <= 1e-8:
            raise ValueError("Training latent standard deviation is invalid or almost zero.")
        return {"latent_value_count": self.count, "mean": self.mean,
                "population_std": std, "scale": 1.0 / std,
                "rms": math.sqrt(std * std + self.mean * self.mean)}


def estimate_training_scale(vae, dataset, batch_size=3, device="cuda"):
    """Use every training clip exactly once, frozen FP32 encoder, float64 moments."""
    import torch
    from torch.utils.data import DataLoader

    previous_device = next(vae.parameters()).device
    previous_mode = vae.training
    moments = PopulationMoments()
    clips = 0
    first_mu = None
    # A private generator avoids consuming the training DataLoader's RNG.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, drop_last=False,
                        generator=torch.Generator().manual_seed(1729))
    try:
        vae.to(device).eval()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            for audio, _condition, _filenames in loader:
                mu, _ = vae.encode(audio.to(device=device, dtype=torch.float32))
                if mu.ndim != 3 or tuple(mu.shape[1:]) != (8, 4096):
                    raise ValueError("Unexpected encoder shape: " + str(tuple(mu.shape)))
                if not torch.isfinite(mu).all():
                    raise ValueError("Non-finite training latents.")
                if first_mu is None:
                    first_mu = mu[:1].float().cpu().clone()
                values = mu.detach().to(device="cpu", dtype=torch.float64)
                variance, mean = torch.var_mean(values, unbiased=False)
                moments.merge(values.numel(), float(mean), float(variance))
                clips += int(mu.shape[0])
                if clips % 60 == 0 or clips == len(dataset):
                    print("Latent statistics: {}/{} training clips".format(clips, len(dataset)), flush=True)
        if clips != len(dataset):
            raise ValueError("Latent statistics did not cover every training clip.")
        report = moments.result()
        scale = torch.tensor(report["scale"], dtype=torch.float32)
        recovered = (first_mu * scale) / scale
        if not torch.allclose(recovered, first_mu, atol=1e-6, rtol=1e-6):
            raise RuntimeError("Scale/inverse-scale round trip failed.")
        report.update({"clip_count": clips, "split": "train", "encoder": "posterior_mean_mu",
                       "centering": False, "normalization": "one scalar for all clips/channels/times",
                       "encoder_precision": "float32", "moments_precision": "float64",
                       "convention": CONVENTION,
                       "round_trip_max_abs_error": float((recovered - first_mu).abs().max()),
                       "scaled_population_std": report["population_std"] * float(scale),
                       "scaled_mean": report["mean"] * float(scale)})
        return report
    finally:
        vae.to(previous_device).train(previous_mode)


def checkpoint_scale(checkpoint):
    """Reject old/unlabelled checkpoints instead of silently applying a new scale."""
    state = checkpoint.get("state_dict", {})
    metadata = checkpoint.get("model_e_expanded_run", {})
    if metadata.get("latent_convention") != CONVENTION or "latent_scale" not in state:
        raise ValueError("Expected a checkpoint from train_model_e_conditioned.py. "
                         "Use the old generator for unscaled checkpoints; start this experiment from scratch.")
    value = state["latent_scale"]
    if not hasattr(value, "numel") or value.numel() != 1 or tuple(value.shape) != ():
        raise ValueError("latent_scale must be a scalar checkpoint buffer.")
    scale = float(value.detach().cpu())
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid checkpoint latent_scale.")
    recorded = float(metadata.get("latent_scale", float("nan")))
    stats = metadata.get("latent_statistics", {})
    expected = float(stats.get("scale", float("nan")))
    if not math.isclose(scale, recorded, rel_tol=1e-6, abs_tol=0.0) or not math.isclose(
            scale, expected, rel_tol=1e-6, abs_tol=0.0):
        raise ValueError("Checkpoint buffer and recorded latent statistics disagree.")
    if stats.get("split") != "train" or stats.get("convention") != CONVENTION:
        raise ValueError("Missing training-only scaling provenance.")
    return scale
