"""Checkpoint and sampler conventions shared by training and generation."""
import inspect
import math
import torch
from model_e_conditioned import ARCHITECTURE
from model_e_conditioning import CONDITION_SCHEMA
from model_e_conditioned_scaling import checkpoint_scale


def load_trusted(path):
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(str(path), **kwargs)


def validate_checkpoint(checkpoint):
    meta = checkpoint.get("model_e_expanded_run", {})
    if meta.get("architecture") != ARCHITECTURE or meta.get("condition_schema") != CONDITION_SCHEMA:
        raise ValueError("This requires a checkpoint from train_model_e_conditioned.py, "
                         "not the old two-channel or scaling-only Model E.")
    if meta.get("condition_channels") != 3:
        raise ValueError("Wrong checkpoint condition dimensions.")
    probability = float(meta.get("cond_drop_prob", float("nan")))
    if not math.isfinite(probability) or not 0 <= probability < 1:
        raise ValueError("Missing or invalid conditioning-dropout metadata.")
    return checkpoint_scale(checkpoint)


def get_alphas_sigmas(t):
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


def get_crash_schedule(t):
    sigma = torch.sin(t * math.pi / 2) ** 2
    alpha = (1 - sigma ** 2).sqrt()
    return torch.atan2(sigma, alpha) / math.pi * 2


def require_guidance_training(metadata, guidance_scale):
    if not math.isfinite(guidance_scale) or guidance_scale < 0:
        raise ValueError("Guidance scale must be finite and nonnegative.")
    if guidance_scale != 1.0 and float(metadata.get("cond_drop_prob", 0)) <= 0:
        raise ValueError("CFG requires a checkpoint trained with --cond-drop-prob > 0. Use guidance 1.0.")


@torch.no_grad()
def sample(model, noise, steps, eta=0.0, cond=None, guidance_scale=1.0):
    """V-prediction, crash schedule, deterministic DDIM; explicit null condition."""
    if eta != 0 or steps < 1 or cond is None:
        raise ValueError("This experiment uses eta=0, positive steps and an explicit condition.")
    t = get_crash_schedule(torch.linspace(1, 0, steps + 1, device=noise.device)[:-1])
    alphas, sigmas = get_alphas_sigmas(t)
    present = noise.new_ones((noise.shape[0],))
    absent = torch.zeros_like(present)
    x = noise
    for i in range(steps):
        with torch.cuda.amp.autocast(enabled=noise.is_cuda):
            conditional = model(x, t[i].expand(x.shape[0]), cond=cond, condition_present=present).float()
            if guidance_scale == 1.0:
                velocity = conditional
            else:
                unconditional = model(x, t[i].expand(x.shape[0]), cond=cond, condition_present=absent).float()
                velocity = unconditional + guidance_scale * (conditional - unconditional)
        pred = x * alphas[i] - velocity * sigmas[i]
        eps = x * sigmas[i] + velocity * alphas[i]
        if i + 1 < steps:
            x = pred * alphas[i + 1] + eps * sigmas[i + 1]
    return pred
