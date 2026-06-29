"""Convert a trained MJX policy (flax params) into the torch checkpoint format
that pytorch/infer.py and pytorch/goto.py already load.

    uv run python mjx/export_policy.py [src.pkl] [dst.pt]
"""
import sys
import pickle
import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/mjx_policy.pkl"
    dst = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/checkpoint_mjx.pt"
    with open(ROOT / src, "rb") as f:
        ckpt = pickle.load(f)

    p = ckpt["actor_params"]["params"]
    state = {"log_std": torch.tensor(np.asarray(p["log_std"]))}
    # flax Dense_i -> torch arch.(2i); kernel (in,out) -> weight (out,in)
    for i in range(5):
        d = p[f"Dense_{i}"]
        state[f"arch.{2 * i}.weight"] = torch.tensor(np.asarray(d["kernel"]).T)
        state[f"arch.{2 * i}.bias"] = torch.tensor(np.asarray(d["bias"]))

    out = {
        "actor": state,
        "obs_mean": np.asarray(ckpt["obs_mean"]),
        "obs_var": np.asarray(ckpt["obs_var"]),
        "obs_count": float(ckpt["obs_count"]),
        "rewards": list(ckpt.get("rewards", [])),   # training curve, if recorded
    }
    dstp = ROOT / dst
    torch.save(out, dstp)
    print("wrote", dstp)


if __name__ == "__main__":
    main()
