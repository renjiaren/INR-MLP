"""
Standalone DeepSDF training script.

This script trains only the DeepSDF decoder and the training-set latent codes.
The trained decoder checkpoint can then be loaded by inr_mlp_deepsdf_pipeline.py
through --pretrained-deepsdf-model.


from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch


def load_pipeline_module():
    """Loads INR-MLP.py even though the filename contains a hyphen."""

    module_path = Path(__file__).with_name("INR-MLP.py")
    if not module_path.exists():
        raise FileNotFoundError(f"Cannot find pipeline module: {module_path}")

    spec = importlib.util.spec_from_file_location("inr_mlp_pipeline", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import pipeline module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PIPELINE = load_pipeline_module()


def run(args: argparse.Namespace) -> None:
    PIPELINE.set_seed(args.seed)
    device = PIPELINE.get_device()

    if args.split_json:
        data_root = Path(args.data_root) if args.data_root else Path(args.split_json).parent
        records = PIPELINE.read_split_json(Path(args.split_json), data_root=data_root)
    elif args.split_csv:
        records = PIPELINE.read_split_csv(Path(args.split_csv))
    else:
        raise ValueError("Provide either --split-json or --split-csv.")

    train_records, val_records, test_records = PIPELINE.split_records(records)
    if not train_records:
        raise ValueError("The split file must contain training samples.")

    print(f"Device: {device}")
    print(
        "Split sizes: "
        f"train={len(train_records)}, val={len(val_records)}, test={len(test_records)}"
    )

    decoder, train_latents = PIPELINE.train_deepsdf(
        train_records=train_records,
        latent_dim=args.latent_dim,
        points_per_sample=args.sdf_points_per_sample,
        epochs=args.deepsdf_epochs,
        batch_size=args.sdf_batch_size,
        lr=args.deepsdf_lr,
        lr_decay_gamma=args.deepsdf_lr_decay_gamma,
        lr_decay_step=args.deepsdf_lr_decay_step,
        min_lr=args.min_lr,
        latent_l2=args.latent_l2,
        sdf_loss=args.sdf_loss,
        clamp_dist=args.clamp_dist,
        device=device,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": decoder.state_dict(),
        "train_latents": train_latents,
        "train_sample_ids": [record.sample_id for record in train_records],
        "args": vars(args),
    }
    checkpoint_path = out_dir / "deepsdf_decoder.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved DeepSDF checkpoint to {checkpoint_path}")

    latents_path = out_dir / "deepsdf_train_latents.pt"
    torch.save(
        {
            "train_latents": train_latents,
            "train_sample_ids": [record.sample_id for record in train_records],
            "args": vars(args),
        },
        latents_path,
    )
    print(f"Saved training latent codes to {latents_path}")

    if args.export_train_meshes > 0:
        PIPELINE.export_latent_code_meshes(
            decoder=decoder,
            records=train_records,
            latent_codes=train_latents,
            out_dir=out_dir,
            split_name="train",
            count=args.export_train_meshes,
            resolution=args.mesh_export_resolution,
            max_batch=args.mesh_export_max_batch,
            smooth_iterations=args.mesh_smooth_iterations,
            show_mesh=args.show_exported_mesh,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone DeepSDF training")
    parser.add_argument("--split-json", default=None, help="JSON with train/validation/test sample directory names")
    parser.add_argument("--data-root", default=None, help="Root directory containing sample directories")
    parser.add_argument("--split-csv", default=None, help="CSV with sample_id, split, path columns")
    parser.add_argument("--out-dir", default="outputs/deepsdf_runs")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--latent-l2", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument(
        "--sdf-loss",
        choices=["l1", "mse"],
        default="l1",
        help="Loss used for DeepSDF reconstruction.",
    )
    parser.add_argument(
        "--clamp-dist",
        type=float,
        default=None,
        help="Optional SDF clamping distance. Example: 0.1 clamps SDF to [-0.1, 0.1].",
    )

    parser.add_argument("--deepsdf-epochs", type=int, default=500)
    parser.add_argument("--deepsdf-lr", type=float, default=1e-3)
    parser.add_argument("--deepsdf-lr-decay-gamma", type=float, default=0.95)
    parser.add_argument("--deepsdf-lr-decay-step", type=int, default=100)
    parser.add_argument("--sdf-batch-size", type=int, default=16)
    parser.add_argument("--sdf-points-per-sample", type=int, default=4096)

    parser.add_argument(
        "--export-train-meshes",
        type=int,
        default=0,
        help="Export this many reconstructed training meshes after DeepSDF training.",
    )
    parser.add_argument("--mesh-export-resolution", type=int, default=256)
    parser.add_argument("--mesh-export-max-batch", type=int, default=2 ** 18)
    parser.add_argument("--mesh-smooth-iterations", type=int, default=10)
    parser.add_argument("--show-exported-mesh", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
