#!/usr/bin/env python3
"""Fast implementation checks; optional real-width 8-s/30-s shape checks.

These test code behavior, not whether trained generations have correct tempo.
"""
import argparse
import math
from pathlib import Path
import numpy as np
import torch
from model_e_conditioning import make_condition, load_condition, CONDITION_SCHEMA
from model_e_conditioned import ConditionedModelE, ARCHITECTURE
from model_e_conditioned_runtime import validate_checkpoint, require_guidance_training, sample
from model_e_conditioned_scaling import CONVENTION, PopulationMoments


def rejected(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError("Invalid configuration was accepted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--full-shape", action="store_true", help="Also run real-width U-Net at latent length 4096")
    parser.add_argument("--long-shape", action="store_true", help="Also run real-width U-Net at latent length 16384")
    parser.add_argument("--reference-dir", type=Path, help="Optionally check every conditioning.npz below this directory")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(123)
    beats = .1 + np.arange(17) * .5
    beats = beats[beats < 8.192]
    c = make_condition([.12, .65, 1.25], beats, 120)
    assert c.shape == (3,4096) and np.all(c[0] == 0)
    assert np.argmax(c[2,:200]) == 50  # preserve the .1 s phase; no beat at time zero
    assert c[1,60] > .999 and c[2,50] > .999
    rejected(lambda: make_condition([.12], beats, 90))
    if args.reference_dir:
        files = sorted(args.reference_dir.rglob("conditioning.npz"))
        if not files:
            raise ValueError("No conditioning.npz files below reference-dir")
        for path in files:
            value, _ = load_condition(path)
            with np.load(str(path), allow_pickle=False) as stored:
                assert np.array_equal(value[:2].numpy(), stored["condition"])
        print("PASS: {} reference conditions; original BPM/onset channels preserved".format(len(files)))
    print("PASS: event phase, three-channel encoding, and inconsistent BPM/grid rejection")

    # Small-width, six-level model makes backwards/CFG tests practical on CPU.
    model = ConditionedModelE(channels=(32,32,64,64,64,64)).to(args.device).eval()
    x = torch.randn(2,8,256,device=args.device)
    condition = torch.from_numpy(c[:,:256]).unsqueeze(0).repeat(2,1,1).to(args.device)
    altered = condition.clone()
    altered[:,0] = -.5
    altered[:,1:] = torch.roll(altered[:,1:], 25, dims=-1)
    t = torch.full((2,), .5, device=args.device)
    zeros = torch.zeros(2,device=args.device)
    with torch.no_grad():
        y = model(x,t,condition)
        null_a = model(x,t,condition,zeros)
        null_b = model(x,t,altered,zeros)
    assert y.shape == x.shape and torch.isfinite(y).all()
    assert torch.equal(null_a,null_b), "Dropped conditioning leaked into the null branch"
    assert not torch.equal(y,null_a), "120 BPM/rest was confused with missing conditioning"
    print("PASS: forward shape, finite output, and explicit null-condition isolation")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    target = torch.randn_like(x)
    loss = (model(x,t,condition)-target).square().mean()
    loss.backward()
    block = model.net.pre[0]
    assert block.film.weight.grad.abs().sum() > 0
    assert block.timing_projection.weight.grad.abs().sum() > 0
    opt.step()
    opt.zero_grad()
    (model(x,t,condition)-target).square().mean().backward()
    assert model.bpm_embedding[0].weight.grad is not None
    # At 120 BPM the scalar is zero, so test the learned embedding's bias gradient.
    assert model.bpm_embedding[0].bias.grad.abs().sum() > 0
    assert model.timing_encoder[0].weight.grad.abs().sum() > 0
    print("PASS: gradients reach FiLM, timing projections, BPM MLP, and timing encoder")
    model.eval()
    with torch.no_grad():
        z = sample(model,x,2,cond=condition,guidance_scale=1)
        guided = sample(model,x,2,cond=condition,guidance_scale=1.5)
    assert z.shape == x.shape and torch.isfinite(z).all() and torch.isfinite(guided).all()
    rejected(lambda: require_guidance_training({"cond_drop_prob":0}, 1.5))
    require_guidance_training({"cond_drop_prob":.1}, 1.5)
    print("PASS: ordinary/CFG sampler and guidance-training guard")

    values = np.random.RandomState(1).normal(1.2,3.7,size=10001)
    moments = PopulationMoments()
    for block_values in np.array_split(values,13):
        moments.merge(len(block_values),block_values.mean(),block_values.var())
    stats = moments.result()
    assert math.isclose(stats["population_std"],values.std(),rel_tol=1e-12)
    scale = stats["scale"]
    assert np.allclose((values*scale)/scale,values)
    checkpoint = {"state_dict":{"latent_scale":torch.tensor(scale)}, "model_e_expanded_run":{
        "architecture":ARCHITECTURE, "condition_schema":CONDITION_SCHEMA, "condition_channels":3,
        "cond_drop_prob":0, "latent_convention":CONVENTION, "latent_scale":scale,
        "latent_statistics":{"scale":scale,"split":"train","convention":CONVENTION}}}
    validate_checkpoint(checkpoint)
    checkpoint["model_e_expanded_run"]["architecture"] = "old_model_e"
    rejected(lambda: validate_checkpoint(checkpoint))
    print("PASS: pooled scale, inverse transform, and incompatible checkpoint rejection")
    del model, opt, y, null_a, null_b, z, guided, loss

    if args.full_shape or args.long_shape:
        full = ConditionedModelE().to(args.device).eval()
        print("Real-width parameter count:", sum(p.numel() for p in full.parameters()), flush=True)
        for length in ([4096] if args.full_shape else []) + ([16384] if args.long_shape else []):
            control = torch.zeros(1,3,length,device=args.device)
            control[:,1,50::125]=1
            control[:,2,50::250]=1
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.device.startswith("cuda")):
                output = full(torch.randn(1,8,length,device=args.device),
                              torch.tensor([.5],device=args.device),control)
            assert output.shape == (1,8,length) and torch.isfinite(output).all()
            print("PASS: real-width forward at latent length",length,flush=True)
    print("Implementation checks passed; generated-audio quality and tempo remain unverified.")


if __name__ == "__main__":
    main()
