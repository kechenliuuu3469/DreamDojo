"""
Modified main.py for LAM fine-tuning.

Usage:
  # Fine-tune from DreamDojo checkpoint (loads weights only, fresh optimizer, epoch 0):
  python main.py fit --config config/lam_bridge_test.yaml --pretrained_ckpt /path/to/LAM_400k.ckpt

  # Train from scratch (no checkpoint):
  python main.py fit --config config/lam_bridge_test.yaml

  # Test:
  python main.py test --config config/lam_bridge_test.yaml --ckpt_path /path/to/LAM_400k.ckpt
"""
import sys
import torch
from lightning.pytorch.cli import LightningCLI

from lam.dataset import LightningVideoDataset
from lam.model import LAM


# Step 1: Extract --pretrained_ckpt from sys.argv before LightningCLI sees it.
pretrained_ckpt = None
filtered_args = []
i = 0
while i < len(sys.argv):
    if sys.argv[i] == "--pretrained_ckpt":
        pretrained_ckpt = sys.argv[i + 1]
        i += 2
    else:
        filtered_args.append(sys.argv[i])
        i += 1
sys.argv = filtered_args

if pretrained_ckpt is not None:
    # Fine-tuning mode
    # Remove 'fit' from sys.argv since we'll call trainer.fit() manually
    sys.argv = [a for a in sys.argv if a != "fit"]

    # Build model and datamodule without running
    cli = LightningCLI(
        LAM,
        LightningVideoDataset,
        seed_everything_default=32,
        run=False,
        subclass_mode_model=False,
        subclass_mode_data=False,
    )

    # Load only the model weights (no optimizer state, no epoch counter)
    print("=" * 60)
    print(f"FINE-TUNING MODE")
    print(f"Loading pretrained weights from: {pretrained_ckpt}")
    print(f"  - Model weights: LOADED from checkpoint")
    print(f"  - Optimizer state: FRESH (not from checkpoint)")
    print(f"  - Epoch counter: starts from 0 (not from checkpoint)")
    print("=" * 60)

    ckpt = torch.load(pretrained_ckpt, map_location="cpu")
    state_dict = ckpt["state_dict"]
    cli.model.load_state_dict(state_dict, strict=True)
    print("Pretrained weights loaded successfully!\n")

    # Setup datamodule and run fit
    cli.datamodule.setup("fit")
    cli.trainer.fit(
        model=cli.model,
        datamodule=cli.datamodule,
    )
else:
    # Normal mode: train from scratch, resume with --ckpt_path, or test
    cli = LightningCLI(
        LAM,
        LightningVideoDataset,
        seed_everything_default=32
    )