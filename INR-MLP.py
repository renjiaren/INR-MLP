"""
Minimal PyTorch pipeline for an SDF latent-code + INR-MLP surrogate.

Expected data format
--------------------
The script supports the directory layout:
    data_root/
        split.json
        sample_id_001/
            sample_id_001.msh
            sample_id_001.npz
            burningarea.csv

The split JSON is expected to contain:
    {
        "train": ["sample_id_001", ...],
        "validation": ["sample_id_101", ...],
        "test": ["sample_id_201", ...]
    }

The .npz file contain:
    points: float array, shape [num_sdf_points, 3]
    sdf:    float array, shape [num_sdf_points, 1] or [num_sdf_points]

or DeepSDF-style signed samples:
    pos: float array, shape [num_positive_points, 4], columns [x, y, z, sdf]
    neg: float array, shape [num_negative_points, 4], columns [x, y, z, sdf]

The target curve is read from burningarea.csv by default, using the "area" column.
"""


from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_exponential_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    initial_lr: float,
    gamma: float,
    step_size: int,
    min_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Creates lr_t = max(min_lr, initial_lr * gamma ** floor(t / step_size)).
    """

    if step_size <= 0:
        raise ValueError("step_size must be positive.")

    min_factor = min_lr / initial_lr if initial_lr > 0 else 0.0

    def lr_lambda(step: int) -> float:
        return max(min_factor, gamma ** (step // step_size))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


@dataclass
class SampleRecord:
    sample_id: str
    split: str
    path: Path
    curve_path: Path
    mesh_path: Path | None = None


def read_split_csv(split_csv: Path) -> List[SampleRecord]:
    records: List[SampleRecord] = []
    with split_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(
                SampleRecord(
                    sample_id=row["sample_id"],
                    split=row["split"].lower(),
                    path=Path(row["path"]),
                    curve_path=Path(row.get("curve_path", Path(row["path"]).with_name("burningarea.csv"))),
                )
            )
    return records


def read_split_json(split_json: Path, data_root: Path) -> List[SampleRecord]:
    with split_json.open("r", encoding="utf-8") as f:
        split_data = json.load(f)

    records: List[SampleRecord] = []
    split_aliases = {
        "train": "train",
        "val": "validation",
        "valid": "validation",
        "validation": "validation",
        "test": "test",
    }
    for raw_split, sample_ids in split_data.items():
        split = split_aliases.get(raw_split.lower(), raw_split.lower())
        for sample_id in sample_ids:
            sample_dir = data_root / sample_id
            npz_path = sample_dir / f"{sample_id}.npz"
            curve_path = sample_dir / "burningarea.csv"
            mesh_path = sample_dir / f"{sample_id}.msh"
            records.append(
                SampleRecord(
                    sample_id=sample_id,
                    split=split,
                    path=npz_path,
                    curve_path=curve_path,
                    mesh_path=mesh_path if mesh_path.exists() else None,
                )
            )
    return records


def load_sdf_samples(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    if "points" in data and "sdf" in data:
        points = data["points"].astype(np.float32)
        sdf = data["sdf"].astype(np.float32)
        if sdf.ndim == 1:
            sdf = sdf[:, None]
        return points, sdf

    if "pos" in data and "neg" in data:
        samples = np.concatenate([data["pos"], data["neg"]], axis=0).astype(np.float32)
        points = samples[:, :3]
        sdf = samples[:, 3:4]
        return points, sdf

    raise KeyError(
        f"{npz_path} must contain either ('points', 'sdf') or ('pos', 'neg') arrays."
    )


class SDFPointDataset(Dataset):
    """
    Samples random SDF query points from each geometry.

    Returns:
        sample_idx: integer index of the geometry in the training set
        points: [points_per_sample, 3]
        sdf: [points_per_sample, 1]
    """

    def __init__(
        self,
        records: List[SampleRecord],
        points_per_sample: int,
    ) -> None:
        self.records = records
        self.points_per_sample = points_per_sample
        self.cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _load(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if idx not in self.cache:
            points, sdf = load_sdf_samples(self.records[idx].path)
            self.cache[idx] = (points, sdf)
        return self.cache[idx]

    def __getitem__(self, idx: int):
        points, sdf = self._load(idx)
        n = points.shape[0]
        choice = np.random.choice(n, size=self.points_per_sample, replace=n < self.points_per_sample)
        return (
            torch.tensor(idx, dtype=torch.long),
            torch.from_numpy(points[choice]),
            torch.from_numpy(sdf[choice]),
        )


def load_target_curve(record: SampleRecord, curve_column: str = "area") -> np.ndarray:
    if record.curve_path.exists():
        with record.curve_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            values = [float(row[curve_column]) for row in reader if row.get(curve_column, "") != ""]
        return np.asarray(values, dtype=np.float32)

    data = np.load(record.path)
    if "web_thickness_curve" in data:
        return data["web_thickness_curve"].astype(np.float32)
    if "area_curve" in data:
        return data["area_curve"].astype(np.float32)
    raise KeyError(
        f"{record.sample_id} must contain {record.curve_path} or a curve array in {record.path}."
    )


class LatentCurveDataset(Dataset):
    """Dataset for training the latent-code-to-curve GRU decoder."""

    def __init__(self, records: List[SampleRecord], latent_codes: torch.Tensor) -> None:
        self.records = records
        self.latent_codes = latent_codes.detach().cpu().float()
        self.target_curves: List[torch.Tensor] = []
        for record in records:
            self.target_curves.append(torch.from_numpy(load_target_curve(record)))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        return self.latent_codes[idx], self.target_curves[idx]


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        act: nn.Module
        if activation.lower() == "sine":
            act = nn.SiLU()
        elif activation.lower() == "tanh":
            act = nn.Tanh()
        else:
            act = nn.ReLU()

        layers: List[nn.Module] = []
        last_dim = in_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(act.__class__())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepSDFDecoder(nn.Module):
    """Decoder f_theta([x, z]) -> sdf."""

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        num_hidden_layers: int = 6,
    ) -> None:
        super().__init__()
        self.mlp = MLP(
            in_dim=3 + latent_dim,
            out_dim=1,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            activation="relu",
        )

    def forward(self, points: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim == 2:
            latent = latent[:, None, :].expand(-1, points.shape[1], -1)
        x = torch.cat([points, latent], dim=-1)
        return self.mlp(x)


def strip_data_parallel_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Removes the 'module.' prefix saved by torch.nn.DataParallel."""

    return {
        key.replace("module.", "", 1) if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def build_official_deepsdf_decoder(
    latent_dim: int,
    hidden_dim: int = 512,
    num_layers: int = 8,
) -> nn.Module:
    """
    Builds the DeepSDF decoder architecture used in outputs/main.py.

    This keeps the latent-code inference path compatible with checkpoints saved
    by the original DeepSDF training script.
    """

    from networks import deep_sdf_decoder as network

    layer_ids = list(range(num_layers))
    return network.Decoder(
        latent_size=latent_dim,
        dims=[hidden_dim] * num_layers,
        dropout=layer_ids,
        dropout_prob=0.2,
        norm_layers=layer_ids,
        latent_in=[4],
        weight_norm=True,
        xyz_in_all=False,
        use_tanh=False,
        latent_dropout=True,
    )


def load_pretrained_deepsdf_decoder(
    model_path: Path,
    latent_dim: int,
    decoder_type: str,
    official_hidden_dim: int,
    official_num_layers: int,
    device: torch.device,
) -> nn.Module:
    """Loads either the internal decoder or the original DeepSDF decoder."""

    if decoder_type == "official":
        decoder = build_official_deepsdf_decoder(
            latent_dim=latent_dim,
            hidden_dim=official_hidden_dim,
            num_layers=official_num_layers,
        )
    elif decoder_type == "internal":
        decoder = DeepSDFDecoder(latent_dim=latent_dim)
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    decoder.load_state_dict(strip_data_parallel_prefix(state_dict))
    return decoder.to(device)


def decode_sdf_values(
    decoder: nn.Module,
    points: torch.Tensor,
    latent: torch.Tensor,
) -> torch.Tensor:
    """
    Decodes SDF values for both decoder APIs used in this workspace.

    Internal decoder:
        decoder(points=[B, P, 3], latent=[B, D]) -> [B, P, 1]

    Original DeepSDF decoder from outputs/main.py:
        decoder(input=[B*P, D+3]) -> [B*P, 1]
    """

    if isinstance(decoder, DeepSDFDecoder):
        return decoder(points, latent)

    if latent.ndim == 2:
        latent = latent[:, None, :].expand(-1, points.shape[1], -1)
    flat_points = points.reshape(-1, points.shape[-1])
    flat_latent = latent.reshape(-1, latent.shape[-1])
    flat_input = torch.cat([flat_latent, flat_points], dim=-1)
    flat_pred = decoder(flat_input)
    return flat_pred.reshape(points.shape[0], points.shape[1], -1)


class GRUCurveDecoder(nn.Module):
    """
    GRU decoder g_phi(z, t) -> web-thickness curve.

    The latent geometry code initializes the recurrent hidden state. A normalized
    progression coordinate t in [0, 1] is then decoded step by step into the
    target curve.
    """

    def __init__(
        self,
        latent_dim: int,
        curve_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.curve_dim = curve_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.init_hidden = nn.Linear(latent_dim, num_layers * hidden_dim)
        self.gru = nn.GRU(
            input_size=1 + latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        batch_size = latent.shape[0]
        t = torch.linspace(0.0, 1.0, self.curve_dim, device=latent.device)
        t = t[None, :, None].expand(batch_size, -1, -1)
        z_seq = latent[:, None, :].expand(-1, self.curve_dim, -1)
        decoder_input = torch.cat([t, z_seq], dim=-1)

        h0 = self.init_hidden(latent)
        h0 = h0.view(batch_size, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        output, _ = self.gru(decoder_input, h0)
        return self.out(output).squeeze(-1)


def train_deepsdf(
    train_records: List[SampleRecord],
    latent_dim: int,
    points_per_sample: int,
    epochs: int,
    batch_size: int,
    lr: float,
    lr_decay_gamma: float,
    lr_decay_step: int,
    min_lr: float,
    latent_l2: float,
    sdf_loss: str,
    clamp_dist: float | None,
    device: torch.device,
) -> Tuple[nn.Module, torch.Tensor]:
    dataset = SDFPointDataset(train_records, points_per_sample=points_per_sample)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    decoder = DeepSDFDecoder(latent_dim=latent_dim).to(device)
    latent_table = nn.Embedding(len(train_records), latent_dim).to(device)
    nn.init.normal_(latent_table.weight, mean=0.0, std=0.01)

    optimizer = torch.optim.Adam(
        list(decoder.parameters()) + list(latent_table.parameters()),
        lr=lr,
        weight_decay=0.0,
    )
    scheduler = make_exponential_lr_scheduler(
        optimizer=optimizer,
        initial_lr=lr,
        gamma=lr_decay_gamma,
        step_size=lr_decay_step,
        min_lr=min_lr,
    )
    criterion: nn.Module
    if sdf_loss.lower() == "l1":
        criterion = nn.L1Loss()
    elif sdf_loss.lower() == "mse":
        criterion = nn.MSELoss()
    else:
        raise ValueError("--sdf-loss must be either 'l1' or 'mse'.")

    for epoch in range(1, epochs + 1):
        decoder.train()
        total_loss = 0.0
        total_count = 0
        for sample_idx, points, sdf in loader:
            sample_idx = sample_idx.to(device)
            points = points.to(device)
            sdf = sdf.to(device)
            latent = latent_table(sample_idx)

            if clamp_dist is not None:
                sdf = torch.clamp(sdf, -clamp_dist, clamp_dist)

            pred = decode_sdf_values(decoder, points, latent)
            if clamp_dist is not None:
                pred = torch.clamp(pred, -clamp_dist, clamp_dist)

            recon_loss = criterion(pred, sdf)
            reg_loss = latent_l2 * latent.pow(2).mean()
            loss = recon_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * points.shape[0]
            total_count += points.shape[0]

        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(
                f"[DeepSDF] epoch={epoch:04d} "
                f"loss={total_loss / max(total_count, 1):.6e} lr={current_lr(optimizer):.3e}"
            )
        scheduler.step()

    return decoder, latent_table.weight.detach().clone()


@torch.no_grad()
def load_area_curve_dim(record: SampleRecord) -> int:
    return int(load_target_curve(record).shape[0])


def infer_latent_code(
    decoder: nn.Module,
    record: SampleRecord,
    latent_dim: int,
    points_per_iter: int,
    max_iters: int,
    lr: float,
    lr_decay_gamma: float,
    lr_decay_step: int,
    min_lr: float,
    latent_l2: float,
    latent_init_std: float,
    sdf_loss: str,
    clamp_dist: float | None,
    tolerance: float,
    patience: int,
    device: torch.device,
) -> Tuple[torch.Tensor, bool, float]:
    """
    Infers the latent code for one unseen geometry with decoder weights frozen.

    Returns:
        latent_code, failed, final_loss
    """
    points_all, sdf_all = load_sdf_samples(record.path)

    for p in decoder.parameters():
        p.requires_grad_(False)
    decoder.eval()

    z = torch.empty(1, latent_dim, device=device)
    nn.init.normal_(z, mean=0.0, std=latent_init_std)
    z.requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=lr)
    scheduler = make_exponential_lr_scheduler(
        optimizer=optimizer,
        initial_lr=lr,
        gamma=lr_decay_gamma,
        step_size=lr_decay_step,
        min_lr=min_lr,
    )
    if sdf_loss.lower() == "l1":
        criterion = nn.L1Loss()
    elif sdf_loss.lower() == "mse":
        criterion = nn.MSELoss()
    else:
        raise ValueError("--sdf-loss must be either 'l1' or 'mse'.")
    best_loss = math.inf
    stale = 0
    failed = False

    n = points_all.shape[0]
    for _ in range(max_iters):
        choice = np.random.choice(n, size=points_per_iter, replace=n < points_per_iter)
        points = torch.from_numpy(points_all[choice]).to(device)[None, :, :]
        sdf = torch.from_numpy(sdf_all[choice]).to(device)[None, :, :]

        if clamp_dist is not None:
            sdf = torch.clamp(sdf, -clamp_dist, clamp_dist)

        pred = decode_sdf_values(decoder, points, z)
        if clamp_dist is not None:
            pred = torch.clamp(pred, -clamp_dist, clamp_dist)

        loss = criterion(pred, sdf) + latent_l2 * z.pow(2).mean()

        if torch.isnan(loss) or torch.isinf(loss):
            failed = True
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        current = float(loss.item())
        if math.isinf(best_loss):
            relative_improvement = math.inf
        else:
            relative_improvement = (best_loss - current) / max(abs(best_loss), 1e-12)
        if current < best_loss:
            best_loss = current
        if relative_improvement < tolerance:
            stale += 1
        else:
            stale = 0
        if stale >= patience:
            break

    return z.detach().cpu().squeeze(0), failed, float(best_loss)


def infer_latent_codes_for_records(
    decoder: nn.Module,
    records: List[SampleRecord],
    latent_dim: int,
    points_per_iter: int,
    max_iters: int,
    lr: float,
    lr_decay_gamma: float,
    lr_decay_step: int,
    min_lr: float,
    latent_l2: float,
    latent_init_std: float,
    sdf_loss: str,
    clamp_dist: float | None,
    tolerance: float,
    patience: int,
    device: torch.device,
) -> torch.Tensor:
    latents: List[torch.Tensor] = []
    failed_count = 0
    for record in records:
        z, failed, final_loss = infer_latent_code(
            decoder=decoder,
            record=record,
            latent_dim=latent_dim,
            points_per_iter=points_per_iter,
            max_iters=max_iters,
            lr=lr,
            lr_decay_gamma=lr_decay_gamma,
            lr_decay_step=lr_decay_step,
            min_lr=min_lr,
            latent_l2=latent_l2,
            latent_init_std=latent_init_std,
            sdf_loss=sdf_loss,
            clamp_dist=clamp_dist,
            tolerance=tolerance,
            patience=patience,
            device=device,
        )
        failed_count += int(failed)
        latents.append(z)
        print(f"[Latent inference] sample={record.sample_id} failed={failed} loss={final_loss:.6e}")
    print(f"[Latent inference] failed cases: {failed_count}/{len(records)}")
    return torch.stack(latents, dim=0)


def export_latent_code_mesh(
    decoder: nn.Module,
    latent_code: torch.Tensor,
    output_prefix: Path,
    resolution: int,
    max_batch: int,
    smooth_iterations: int,
    show_mesh: bool,
) -> None:
    """
    Reconstructs a mesh from one latent code using the DeepSDF mesh routine.

    This is the functionality originally implemented in generateLatentCode.py:
    given a trained decoder and an inferred latent code, sample the SDF field,
    extract the zero level set, and optionally smooth/export the mesh.
    """

    if not torch.cuda.is_available():
        raise RuntimeError(
            "outputs/deep_sdf/mesh.py uses CUDA tensors internally; mesh export requires CUDA."
        )

    import deep_sdf.mesh

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    latent = latent_code.detach().reshape(1, -1).cuda()
    decoder = decoder.cuda().eval()

    with torch.no_grad():
        deep_sdf.mesh.create_mesh(
            decoder,
            latent,
            str(output_prefix),
            N=resolution,
            max_batch=max_batch,
        )

    ply_path = output_prefix.with_suffix(".ply")
    if smooth_iterations <= 0 and not show_mesh:
        return

    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "open3d is required for mesh smoothing or display. Install open3d or set "
            "--mesh-smooth-iterations 0 and omit --show-exported-mesh."
        ) from exc

    mesh = o3d.io.read_triangle_mesh(str(ply_path))
    mesh.compute_vertex_normals()
    if smooth_iterations > 0:
        mesh = mesh.filter_smooth_simple(number_of_iterations=smooth_iterations)
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(output_prefix.with_suffix(".obj")), mesh)
    if show_mesh:
        o3d.visualization.draw_geometries(
            [mesh],
            window_name="Latent-code reconstruction",
            width=1600,
            height=1200,
        )


def export_latent_code_meshes(
    decoder: nn.Module,
    records: List[SampleRecord],
    latent_codes: torch.Tensor,
    out_dir: Path,
    split_name: str,
    count: int,
    resolution: int,
    max_batch: int,
    smooth_iterations: int,
    show_mesh: bool,
) -> None:
    if count <= 0:
        return

    export_count = min(count, len(records))
    for idx in range(export_count):
        output_prefix = out_dir / "latent_meshes" / split_name / records[idx].sample_id
        export_latent_code_mesh(
            decoder=decoder,
            latent_code=latent_codes[idx],
            output_prefix=output_prefix,
            resolution=resolution,
            max_batch=max_batch,
            smooth_iterations=smooth_iterations,
            show_mesh=show_mesh,
        )
        print(f"[Latent mesh] exported {output_prefix.with_suffix('.ply')}")


def train_area_predictor(
    train_records: List[SampleRecord],
    val_records: List[SampleRecord],
    train_latents: torch.Tensor,
    val_latents: torch.Tensor,
    latent_dim: int,
    curve_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    lr_decay_gamma: float,
    lr_decay_step: int,
    min_lr: float,
    weight_decay: float,
    patience: int,
    device: torch.device,
) -> GRUCurveDecoder:
    train_ds = LatentCurveDataset(train_records, train_latents)
    val_ds = LatentCurveDataset(val_records, val_latents)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = GRUCurveDecoder(latent_dim=latent_dim, curve_dim=curve_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = make_exponential_lr_scheduler(
        optimizer=optimizer,
        initial_lr=lr,
        gamma=lr_decay_gamma,
        step_size=lr_decay_step,
        min_lr=min_lr,
    )
    mse = nn.MSELoss()

    best_state = None
    best_val = math.inf
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for z, area in train_loader:
            z = z.to(device)
            area = area.to(device)
            pred = model(z)
            loss = mse(pred, area)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_mse = evaluate_area_predictor(model, val_loader, device)
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if epoch == 1 or epoch % 50 == 0:
            print(
                f"[GRU curve decoder] epoch={epoch:04d} "
                f"val_mse={val_mse:.6e} lr={current_lr(optimizer):.3e}"
            )
        if stale >= patience:
            print(f"[GRU curve decoder] early stopping at epoch {epoch}")
            break
        scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_area_predictor(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    mse = nn.MSELoss(reduction="sum")
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for z, area in loader:
            z = z.to(device)
            area = area.to(device)
            pred = model(z)
            total_loss += float(mse(pred, area).item())
            total_count += int(area.numel())
    return total_loss / max(total_count, 1)


def split_records(records: Iterable[SampleRecord]) -> Tuple[List[SampleRecord], List[SampleRecord], List[SampleRecord]]:
    train = [r for r in records if r.split == "train"]
    val = [r for r in records if r.split in {"val", "validation"}]
    test = [r for r in records if r.split == "test"]
    return train, val, test


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device()
    if args.split_json:
        data_root = Path(args.data_root) if args.data_root else Path(args.split_json).parent
        records = read_split_json(Path(args.split_json), data_root=data_root)
    elif args.split_csv:
        records = read_split_csv(Path(args.split_csv))
    else:
        raise ValueError("Provide either --split-json or --split-csv.")
    train_records, val_records, test_records = split_records(records)
    if not train_records or not val_records or not test_records:
        raise ValueError("The split file must contain train, validation, and test samples.")

    print(f"Device: {device}")
    print(f"Split sizes: train={len(train_records)}, val={len(val_records)}, test={len(test_records)}")

    if args.pretrained_deepsdf_model:
        decoder = load_pretrained_deepsdf_decoder(
            model_path=Path(args.pretrained_deepsdf_model),
            latent_dim=args.latent_dim,
            decoder_type=args.decoder_type,
            official_hidden_dim=args.official_deepsdf_hidden_dim,
            official_num_layers=args.official_deepsdf_num_layers,
            device=device,
        )
        print(f"Loaded pretrained DeepSDF decoder from {args.pretrained_deepsdf_model}")
        train_latents = infer_latent_codes_for_records(
            decoder=decoder,
            records=train_records,
            latent_dim=args.latent_dim,
            points_per_iter=args.infer_points_per_iter,
            max_iters=args.infer_iters,
            lr=args.infer_lr,
            lr_decay_gamma=args.infer_lr_decay_gamma,
            lr_decay_step=args.infer_lr_decay_step,
            min_lr=args.min_lr,
            latent_l2=args.latent_l2,
            latent_init_std=args.infer_latent_init_std,
            sdf_loss=args.sdf_loss,
            clamp_dist=args.clamp_dist,
            tolerance=args.infer_tolerance,
            patience=args.infer_patience,
            device=device,
        )
    else:
        decoder, train_latents = train_deepsdf(
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

    val_latents = infer_latent_codes_for_records(
        decoder=decoder,
        records=val_records,
        latent_dim=args.latent_dim,
        points_per_iter=args.infer_points_per_iter,
        max_iters=args.infer_iters,
        lr=args.infer_lr,
        lr_decay_gamma=args.infer_lr_decay_gamma,
        lr_decay_step=args.infer_lr_decay_step,
        min_lr=args.min_lr,
        latent_l2=args.latent_l2,
        latent_init_std=args.infer_latent_init_std,
        sdf_loss=args.sdf_loss,
        clamp_dist=args.clamp_dist,
        tolerance=args.infer_tolerance,
        patience=args.infer_patience,
        device=device,
    )
    test_latents = infer_latent_codes_for_records(
        decoder=decoder,
        records=test_records,
        latent_dim=args.latent_dim,
        points_per_iter=args.infer_points_per_iter,
        max_iters=args.infer_iters,
        lr=args.infer_lr,
        lr_decay_gamma=args.infer_lr_decay_gamma,
        lr_decay_step=args.infer_lr_decay_step,
        min_lr=args.min_lr,
        latent_l2=args.latent_l2,
        latent_init_std=args.infer_latent_init_std,
        sdf_loss=args.sdf_loss,
        clamp_dist=args.clamp_dist,
        tolerance=args.infer_tolerance,
        patience=args.infer_patience,
        device=device,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "train_latents": train_latents,
            "val_latents": val_latents,
            "test_latents": test_latents,
            "args": vars(args),
        },
        out_dir / "inferred_latent_codes.pt",
    )
    print(f"Saved inferred latent codes to {out_dir / 'inferred_latent_codes.pt'}")

    if args.export_latent_meshes > 0:
        if args.export_mesh_split in {"train", "all"}:
            export_latent_code_meshes(
                decoder=decoder,
                records=train_records,
                latent_codes=train_latents,
                out_dir=out_dir,
                split_name="train",
                count=args.export_latent_meshes,
                resolution=args.mesh_export_resolution,
                max_batch=args.mesh_export_max_batch,
                smooth_iterations=args.mesh_smooth_iterations,
                show_mesh=args.show_exported_mesh,
            )
        if args.export_mesh_split in {"validation", "all"}:
            export_latent_code_meshes(
                decoder=decoder,
                records=val_records,
                latent_codes=val_latents,
                out_dir=out_dir,
                split_name="validation",
                count=args.export_latent_meshes,
                resolution=args.mesh_export_resolution,
                max_batch=args.mesh_export_max_batch,
                smooth_iterations=args.mesh_smooth_iterations,
                show_mesh=args.show_exported_mesh,
            )
        if args.export_mesh_split in {"test", "all"}:
            export_latent_code_meshes(
                decoder=decoder,
                records=test_records,
                latent_codes=test_latents,
                out_dir=out_dir,
                split_name="test",
                count=args.export_latent_meshes,
                resolution=args.mesh_export_resolution,
                max_batch=args.mesh_export_max_batch,
                smooth_iterations=args.mesh_smooth_iterations,
                show_mesh=args.show_exported_mesh,
            )

    if args.infer_latents_only:
        return

    curve_dim = load_area_curve_dim(train_records[0])
    predictor = train_area_predictor(
        train_records=train_records,
        val_records=val_records,
        train_latents=train_latents,
        val_latents=val_latents,
        latent_dim=args.latent_dim,
        curve_dim=curve_dim,
        epochs=args.predictor_epochs,
        batch_size=args.predictor_batch_size,
        lr=args.predictor_lr,
        lr_decay_gamma=args.predictor_lr_decay_gamma,
        lr_decay_step=args.predictor_lr_decay_step,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        patience=args.predictor_patience,
        device=device,
    )

    test_ds = LatentCurveDataset(test_records, test_latents)
    test_loader = DataLoader(test_ds, batch_size=args.predictor_batch_size, shuffle=False)
    test_mse = evaluate_area_predictor(predictor, test_loader, device)
    print(f"[Final] test_mse={test_mse:.6e}, test_rmse={math.sqrt(test_mse):.6e}")

    torch.save(
        {
            "decoder_state_dict": decoder.state_dict(),
            "predictor_state_dict": predictor.state_dict(),
            "train_latents": train_latents,
            "val_latents": val_latents,
            "test_latents": test_latents,
            "args": vars(args),
        },
        out_dir / "inr_mlp_deepsdf_checkpoint.pt",
    )
    print(f"Saved checkpoint to {out_dir / 'inr_mlp_deepsdf_checkpoint.pt'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeepSDF + INR-MLP burning-area surrogate")
    parser.add_argument("--split-json", default=None, help="JSON with train/validation/test sample directory names")
    parser.add_argument("--data-root", default=None, help="Root directory containing sample directories")
    parser.add_argument("--split-csv", default=None, help="CSV with sample_id, split, path columns")
    parser.add_argument("--out-dir", default="outputs/inr_mlp_runs")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--latent-l2", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument(
        "--sdf-loss",
        choices=["l1", "mse"],
        default="l1",
        help="Loss used for DeepSDF training and latent-code inference.",
    )
    parser.add_argument(
        "--clamp-dist",
        type=float,
        default=None,
        help="Optional SDF clamping distance. Example: 0.1 clamps SDF to [-0.1, 0.1].",
    )

    parser.add_argument(
        "--pretrained-deepsdf-model",
        default=None,
        help="Optional pretrained DeepSDF model checkpoint, e.g. examples/ninegrain/ModelParameters/latest.pth.",
    )
    parser.add_argument(
        "--decoder-type",
        choices=["internal", "official"],
        default="internal",
        help="Use 'official' for checkpoints produced by outputs/main.py.",
    )
    parser.add_argument("--official-deepsdf-hidden-dim", type=int, default=512)
    parser.add_argument("--official-deepsdf-num-layers", type=int, default=8)
    parser.add_argument(
        "--infer-latents-only",
        action="store_true",
        help="Only infer and save latent codes; skip curve-decoder training.",
    )
    parser.add_argument(
        "--export-latent-meshes",
        type=int,
        default=0,
        help="Export this many reconstructed meshes per selected split after latent-code inference.",
    )
    parser.add_argument(
        "--export-mesh-split",
        choices=["train", "validation", "test", "all"],
        default="test",
        help="Dataset split used for optional latent-code mesh export.",
    )
    parser.add_argument("--mesh-export-resolution", type=int, default=256)
    parser.add_argument("--mesh-export-max-batch", type=int, default=2 ** 18)
    parser.add_argument("--mesh-smooth-iterations", type=int, default=10)
    parser.add_argument("--show-exported-mesh", action="store_true")

    parser.add_argument("--deepsdf-epochs", type=int, default=500)
    parser.add_argument("--deepsdf-lr", type=float, default=1e-3)
    parser.add_argument("--deepsdf-lr-decay-gamma", type=float, default=0.95)
    parser.add_argument("--deepsdf-lr-decay-step", type=int, default=100)
    parser.add_argument("--sdf-batch-size", type=int, default=16)
    parser.add_argument("--sdf-points-per-sample", type=int, default=4096)

    parser.add_argument("--infer-iters", type=int, default=1000)
    parser.add_argument("--infer-lr", type=float, default=1e-3)
    parser.add_argument("--infer-lr-decay-gamma", type=float, default=0.95)
    parser.add_argument("--infer-lr-decay-step", type=int, default=100)
    parser.add_argument("--infer-points-per-iter", type=int, default=4096)
    parser.add_argument("--infer-latent-init-std", type=float, default=0.01)
    parser.add_argument("--infer-tolerance", type=float, default=1e-5)
    parser.add_argument("--infer-patience", type=int, default=200)

    parser.add_argument("--predictor-epochs", type=int, default=1000)
    parser.add_argument("--predictor-lr", type=float, default=1e-3)
    parser.add_argument("--predictor-lr-decay-gamma", type=float, default=0.95)
    parser.add_argument("--predictor-lr-decay-step", type=int, default=100)
    parser.add_argument("--predictor-batch-size", type=int, default=32)
    parser.add_argument("--predictor-patience", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
