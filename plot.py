"""Plot the training reward curve stored in a checkpoint.

    uv run python plot.py [checkpoints/checkpoint_mjx_ft5.pt]
"""
import sys
import torch
import matplotlib.pyplot as plt

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/checkpoint_mjx_ft5.pt'
checkpoint = torch.load(ckpt_path, weights_only=False)

if 'rewards' not in checkpoint:
    raise SystemExit(f"{ckpt_path} has no 'rewards' history (keys: {list(checkpoint)})")
rewards = checkpoint['rewards']

plt.figure(figsize=(8, 4))
plt.plot(rewards)
plt.xlabel('Iteration')
plt.ylabel('Mean Reward')
plt.title(f'Training Rewards ({ckpt_path.split("/")[-1]}, {len(rewards)} iters)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('rewards.png', dpi=120)
