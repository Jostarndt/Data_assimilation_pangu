# uv run prio_knowledge_training_single_gpu.py
"""
Fine-tune PanguWeather (ONNX → PyTorch) for precipitation prediction via:
  - LoRA adapters on all large nn.Linear layers (frozen base weights)
  - A lightweight CNN precipitation head on top of the surface output
  - Only LoRA params + head params are trained

Data convention (norm_scheme=False → raw physical units):
  surface variables: [mslp, u10, v10, t2m,  total_precipitation_24hr]
                      ← PANGU_N_SURFACE=4 → │← PRECIP_IDX=4 →
  PanguWeather input/output: first 4 surface channels only
  Precipitation target: batch['next_state']['surface'][:, PRECIP_IDX:PRECIP_IDX+1]
"""

import os
import sys
import math
import argparse
import itertools
import time
import yaml
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import numpy as np
try:
    import wandb
except ImportError:
    print("Warning: wandb failed to import. Using a dummy wandb module.")
    class wandb:
        @staticmethod
        def init(*args, **kwargs): pass
        @staticmethod
        def log(*args, **kwargs): pass
        @staticmethod
        def save(*args, **kwargs): pass
        @staticmethod
        def define_metric(*args, **kwargs): pass

from tensordict.tensordict import TensorDict
from onnx2torch import convert

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

# ── Variable layout ───────────────────────────────────────────────────────────

PANGU_SURFACE_VARS = [
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
]
PANGU_LEVEL_VARS = [
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
]
PANGU_N_SURFACE = len(PANGU_SURFACE_VARS)   # 4 – channels fed to PanguWeather
PRECIP_IDX      = PANGU_N_SURFACE           # 4 – index of precip in full surface tensor

ALL_SURFACE_VARS = PANGU_SURFACE_VARS + ["total_precipitation_24hr"]


# ── LoRA ──────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Low-rank adaptation wrapper for any module with a 2-D 'weight' parameter.

    Works with nn.Linear AND the OnnxGemm/OnnxMatMul modules that onnx2torch
    produces from ONNX Gemm / MatMul nodes.

    Output = base_module(x)  +  scale · (x @ A.T @ B.T)
    B is initialised to zero → LoRA delta starts at zero.
    """

    def __init__(self, module: nn.Module, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.module = module
        self.scale  = alpha / rank
        w = module.weight
        out_f, in_f = w.shape[0], w.shape[1]
        self.lora_A = nn.Parameter(torch.randn(rank, in_f) * (rank ** -0.5))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        for p in self.module.parameters():
            p.requires_grad = False

    def forward(self, *args, **kwargs) -> torch.Tensor:
        base = self.module(*args, **kwargs)
        # LoRA delta applied to the first positional tensor input
        x = args[0]
        if x.ndim > 2:
            shape = x.shape
            x_2d  = x.reshape(-1, shape[-1])
            delta  = (x_2d @ self.lora_A.T @ self.lora_B.T) * self.scale
            delta  = delta.reshape(*shape[:-1], -1)
        else:
            delta = (x @ self.lora_A.T @ self.lora_B.T) * self.scale
        return base + delta


def apply_lora(model: nn.Module, rank: int = 8, alpha: float = 16.0,
               min_params: int = 256) -> int:
    """Wrap any module that has a 2-D 'weight' nn.Parameter with LoRALinear.

    Targets nn.Linear and onnx2torch Gemm/MatMul modules alike.
    Skips Conv layers (weight.ndim > 2) and tiny projections.
    Returns the number of adapted layers.
    """
    count = 0
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            w = getattr(child, 'weight', None)
            if (w is not None
                    and isinstance(w, nn.Parameter)
                    and w.ndim == 2
                    and w.numel() >= min_params):
                setattr(parent, name, LoRALinear(child, rank=rank, alpha=alpha))
                count += 1
    return count


# ── Precipitation head ────────────────────────────────────────────────────────

class PrecipitationHead(nn.Module):
    """Lightweight CNN: PanguWeather surface output → 24-h total precipitation.

    Input:  [B, 4, H, W]  – raw surface fields (mslp, u10, v10, t2m)
    Output: [B, 1, H, W]  – predicted precip (non-negative via Softplus)
    """

    def __init__(self, in_channels: int = 4, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden,      3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden,      hidden // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden // 2, 1,           1),
            nn.Softplus(),
        )

    def forward(self, surface: torch.Tensor) -> torch.Tensor:
        return self.net(surface)


# ── Loss ──────────────────────────────────────────────────────────────────────

class PrecipLoss:
    """Precipitation loss: latitude-weighted MSE + optional SEEPS.

    SEEPS requires threshold/contingency-table files at the *same* spatial
    resolution as PanguWeather (0.25°, 721×1440).  The arches files
    (seeps_thresholds_240.pt) are at 1.5° and cannot be used here directly;
    pass --seeps_thresholds / --seeps_contingency to point to 0.25° versions.
    """

    def __init__(self, device,
                 seeps_thresholds_path: str = None,
                 seeps_contingency_path: str = None):
        self.device = device
        self.area_weights = self._lat_weights(721).to(device)   # 0.25° → 721 lat pts

        self.use_seeps = False
        if (seeps_thresholds_path and os.path.exists(seeps_thresholds_path)
                and seeps_contingency_path and os.path.exists(seeps_contingency_path)):
            thr = torch.load(seeps_thresholds_path, map_location=device, weights_only=False)
            # imerg_seeps_stats_0p25deg.pt is a dict; pull the [366, lon, lat, 2] tensor.
            # Layout stored by generate_imerg_seeps_0p25deg.py:
            #   thresholds_per_doy: [366, 721, 1440, 2]  (lat-major)
            # Permute to [366, 1440, 721, 2] to match contingency table (lon-major).
            if isinstance(thr, dict):
                thr = thr['thresholds_per_doy']          # [366, lat, lon, 2]
                thr = thr.permute(0, 2, 1, 3)            # [366, lon, lat, 2]
            self.thresholds = thr.to(device)
            self.con_table  = torch.load(seeps_contingency_path, map_location=device,
                                         weights_only=False)
            self.use_seeps  = True
            print(f"✓ SEEPS enabled (thresholds: {self.thresholds.shape}, "
                  f"con_table: {self.con_table.shape})")
        else:
            print("⚠  SEEPS disabled – provide 0.25° threshold files via "
                  "--seeps_thresholds / --seeps_contingency to enable")

    @staticmethod
    def _lat_weights(nlat: int) -> torch.Tensor:
        lats   = torch.linspace(-90, 90, nlat)
        pts    = torch.deg2rad(lats)
        half   = torch.tensor([math.pi / 2])
        bounds = torch.cat([-half, (pts[:-1] + pts[1:]) / 2, half])
        w = torch.sin(bounds[1:]) - torch.sin(bounds[:-1])
        return (w / w.mean()).unsqueeze(1)   # [nlat, 1]

    def rmse_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Latitude-weighted MSE.  pred / target: [B, 1, H, W] raw units."""
        return ((pred - target).pow(2) * self.area_weights).mean()

    def seeps_loss(self, pred_denorm: torch.Tensor, gt_denorm: torch.Tensor,
                   day_of_years) -> torch.Tensor:
        # a1, a2: [N, lon, lat, 3]   con_table: [366, lon, lat, 3, 3]
        a1 = self._soft_onehot(pred_denorm, day_of_years)
        a2 = self._soft_onehot(gt_denorm,   day_of_years)
        ct = torch.nan_to_num(self.con_table) if torch.is_grad_enabled() else self.con_table
        loss = torch.einsum('...ij,...j->...i', ct[day_of_years, ...], a2)
        loss = torch.einsum('...i,...i->...', a1, loss)
        if torch.is_grad_enabled():
            loss = torch.nan_to_num(loss)
        return loss.nanmean()

    def _soft_onehot(self, vals: torch.Tensor, doy, temperature: float = 1e-4):
        # thresholds: [366, lon=1440, lat=721, 2]  → slice → [N, 1440, 721, 2]
        lo = self.thresholds[doy, :, :, 0]   # [N, 1440, 721]
        hi = self.thresholds[doy, :, :, 1]
        # vals from precip_pred/target squeezed: [B, H=721, W=1440]
        # reshape to [N, lat, lon] then permute to [N, lon, lat] to match thresholds
        vals = vals.reshape(lo.shape[0], lo.shape[2], lo.shape[1])  # [N, lat, lon] → [N, lon, lat] after transpose
        # At this point both lo/hi and vals are [N, lon, lat] — no further transpose needed
        if torch.is_grad_enabled():
            vals, lo, hi = map(torch.nan_to_num, (vals, lo, hi))
        pb  = torch.sigmoid((lo - vals) / temperature)
        pa  = torch.sigmoid((vals - hi) / temperature)
        out = torch.stack([pb, 1 - pb - pa, pa], dim=-1)  # [N, lon, lat, 3]
        return torch.nan_to_num(out) if torch.is_grad_enabled() else out

    def __call__(self, pred: torch.Tensor, target: torch.Tensor,
                 timestamp=None, denorm_fn=None,
                 seeps_weight: float = 1.0):
        """
        Args:
            pred / target: [B, 1, H, W] in raw physical units
            timestamp: int64 UNIX timestamps (for SEEPS day-of-year lookup)
            denorm_fn: dataset.denormalize – only needed if norm_scheme != False
        Returns:
            (loss_rmse, loss_seeps)  – both scalar tensors
        """
        loss_rmse = self.rmse_loss(pred, target)
        loss_seeps = torch.zeros(1, device=self.device).squeeze()

        if self.use_seeps and timestamp is not None:
            doy = [(datetime.fromtimestamp(v).timetuple().tm_yday + 1) % 366
                   for v in timestamp.cpu().numpy()]
            loss_seeps = self.seeps_loss(pred.squeeze(1), target.squeeze(1), doy)

        return loss_rmse, loss_seeps


# ── Utilities ─────────────────────────────────────────────────────────────────

def collate_fn(lst):
    return {k: torch.stack([x[k] for x in lst]) for k in lst[0]}


def send_to_device(batch, device, precision):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if key == "timestamp":
                batch[key] = value.to(device)
            else:
                batch[key] = value.to(precision).to(device)
        elif isinstance(value, TensorDict):
            batch[key] = value.to(precision).to(device)
    return batch


def save_checkpoint(model, precip_head, optimizer, epoch, step, config):
    """Save only trainable (LoRA + head) parameters."""
    os.makedirs(config['output']['model_save_path'], exist_ok=True)
    path = os.path.join(config['output']['model_save_path'],
                        f'ckpt_epoch{epoch}_step{step}.pth')
    lora_state = {k: v for k, v in model.state_dict().items() if 'lora_' in k}
    torch.save({
        'lora_state': lora_state,
        'head_state': precip_head.state_dict(),
        'optimizer':  optimizer.state_dict(),
        'epoch': epoch,
        'step':  step,
    }, path)
    print(f"Checkpoint saved → {path}")
    for fn in os.listdir(config['output']['model_save_path']):
        if (fn.startswith('ckpt_epoch') and fn.endswith('.pth')
                and fn != os.path.basename(path)):
            os.remove(os.path.join(config['output']['model_save_path'], fn))


def load_checkpoint(model, precip_head, optimizer, path):
    """Restore LoRA weights + head from a checkpoint."""
    ckpt = torch.load(path, map_location='cpu')
    state = model.state_dict()
    state.update(ckpt['lora_state'])
    model.load_state_dict(state)
    precip_head.load_state_dict(ckpt['head_state'])
    optimizer.load_state_dict(ckpt['optimizer'])
    print(f"Resumed from {path}  (epoch {ckpt['epoch']}, step {ckpt['step']})")
    return ckpt['epoch'], ckpt['step']


# ── Training & validation ─────────────────────────────────────────────────────

def train_fkt(model, precip_head, train_loader, criterion, optimizer, scheduler,
              scaler, device, precision, config, epoch, start_step=0):
    # Keep PanguWeather in eval mode (BN/dropout frozen), but LoRA params
    # still receive gradients because requires_grad=True on lora_A / lora_B.
    model.eval()
    precip_head.train()

    start_time = time.time()
    for i, batch in enumerate(itertools.islice(train_loader, start_step, None)):
        actual_step = start_step + i
        batch.pop('prev_state', None)
        batch = send_to_device(batch, device, precision)

        # ERA5Forecast surface tensors have shape [B, n_vars, 1, H, W] (extra level dim).
        # PanguWeather expects [B, 4, H, W] for surface and [B, 5, 13, H, W] for level.
        surface_input = batch['state']['surface'][:, :PANGU_N_SURFACE, 0, :, :]  # [B, 4, H, W]
        level_input   = batch['state']['level']  # [B, 5, 13, H, W]

        optimizer.zero_grad()

        with autocast('cuda', dtype=precision):
            outputs_level, outputs_surface = model(
                input_1=level_input, input_2=surface_input
            )
            precip_pred   = precip_head(outputs_surface)                     # [B, 1, H, W]
            precip_target = batch['next_state']['surface'][:, PRECIP_IDX:PRECIP_IDX + 1, 0, :, :]  # [B, 1, H, W]

            loss_rmse, loss_seeps = criterion(
                precip_pred, precip_target, timestamp=batch['timestamp']
            )
            loss = loss_rmse + loss_seeps

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        trainable = [p for p in list(model.parameters()) + list(precip_head.parameters())
                     if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            wandb.log({
                'step':          actual_step,
                'loss_rmse':     loss_rmse.item(),
                'loss_seeps':    loss_seeps.item() if torch.is_tensor(loss_seeps) else float(loss_seeps),
                'lr':            optimizer.param_groups[0]['lr'],
                'time/100steps': elapsed,
            })
            print(f"[epoch {epoch}  step {actual_step}]  "
                  f"rmse={loss_rmse.item():.4f}  seeps={loss_seeps:.4f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}  t={elapsed:.1f}s")
            start_time = time.time()

        if (i + 1) % 1000 == 0:
            save_checkpoint(model, precip_head, optimizer, epoch, actual_step, config)


def validate(model, precip_head, val_loader, criterion, device, precision,
             max_batches: int = 200):
    model.eval()
    precip_head.eval()
    total_rmse = 0.0
    with torch.no_grad():
        for j, batch in enumerate(val_loader):
            if j >= max_batches:
                break
            batch.pop('prev_state', None)
            batch = send_to_device(batch, device, precision)
            surface_input = batch['state']['surface'][:, :PANGU_N_SURFACE, 0, :, :]  # [B, 4, H, W]
            level_input   = batch['state']['level']
            _, outputs_surface = model(input_1=level_input, input_2=surface_input)
            precip_pred   = precip_head(outputs_surface)
            precip_target = batch['next_state']['surface'][:, PRECIP_IDX:PRECIP_IDX + 1, 0, :, :]  # [B, 1, H, W]
            rmse, _ = criterion(precip_pred, precip_target)
            total_rmse += rmse.item()
    model.train()
    return total_rmse / max(j + 1, 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='PanguWeather LoRA + precipitation head fine-tuning')
    parser.add_argument('--config',            type=str,   default='config.yaml')
    parser.add_argument('--name',              type=str,   default=None)
    parser.add_argument('--cluster',           action='store_true')
    parser.add_argument('--checkpoint',        type=str,   default=None,
                        help='Path to ckpt_epoch*.pth to resume from')
    parser.add_argument('--slurm_id',          type=str,   default=None)
    parser.add_argument('--git_commit',        type=str,   default=None)
    parser.add_argument('--lora_rank',         type=int,   default=8)
    parser.add_argument('--lora_alpha',        type=float, default=16.0)
    parser.add_argument('--batch_size',        type=int,   default=1)
    parser.add_argument('--pangu_onnx',        type=str,
                        default='/home/fe/isil/pangu_weather_24.onnx')
    parser.add_argument('--data_path',         type=str,
                        default='/srv/data/era_high_res_jost/weatherbench2_complete_2022.nc')
    parser.add_argument('--device',            type=str,   default=None,
                        help='CUDA device, e.g. cuda:0, cuda:1 (default: first available)')
    parser.add_argument('--seeps_thresholds',  type=str,   default=None,
                        help='0.25° SEEPS threshold file (721×1440); omit to disable SEEPS')
    parser.add_argument('--seeps_contingency', type=str,   default=None,
                        help='0.25° SEEPS contingency-table file')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config['git_commit'] = args.git_commit

    if args.cluster:
        wandb.init(project=config['wandb']['project'], config=config,
                   name=args.name, tags=[args.slurm_id], dir='/mnt/output')
        config['output']['model_save_path'] = '/mnt/output/'
        wandb.save('prio_knowledge_training_single_gpu.py')
        print(f"Cluster run – output → {config['output']['model_save_path']}")
    else:
        wandb.init(project=config['wandb']['project'], config=config,
                   name=args.name, mode='online')

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    precision = torch.float32
    print(f'Device: {device}')

    # ── data ─────────────────────────────────────────────────────────────────
    current_file_dir = Path(__file__).resolve().parent
    sys.path.append(str(current_file_dir.parent.parent / 'data/era_from_arches'))
    from era5 import Era5Forecast

    variables = {
        'surface': ALL_SURFACE_VARS,   # 5 channels; first 4 go to PanguWeather
        'level':   PANGU_LEVEL_VARS,   # 5 channels
    }
    lead_time = config['data']['lead_time_hours']

    train_dataset = Era5Forecast(domain='train', lead_time_hours=lead_time,
                                 multistep=1, variables=variables, norm_scheme=False,
                                 path=args.data_path)

    # With a single-year file (2022) the standard 'val' domain (2019) would be empty.
    # Fall back to holding out the last 10% of train samples as validation.
    try:
        val_dataset = Era5Forecast(domain='val', lead_time_hours=lead_time,
                                   multistep=1, variables=variables, norm_scheme=False,
                                   path=args.data_path)
        if len(val_dataset) == 0:
            raise ValueError("val domain empty")
    except Exception:
        n_val = max(1, len(train_dataset) // 10)
        val_dataset, train_dataset = torch.utils.data.random_split(
            train_dataset, [n_val, len(train_dataset) - n_val],
            generator=torch.Generator().manual_seed(42))
        print(f"⚠  val domain empty; using last {n_val} samples as val set")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True,  num_workers=8, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, collate_fn=collate_fn)

    # ── model ─────────────────────────────────────────────────────────────────
    print(f'Loading PanguWeather from {args.pangu_onnx} …')
    pangu = convert(args.pangu_onnx).to(device).to(precision)
    for p in pangu.parameters():
        p.requires_grad = False

    n_lora = apply_lora(pangu, rank=args.lora_rank, alpha=args.lora_alpha)
    print(f'LoRA applied to {n_lora} linear layers  '
          f'(rank={args.lora_rank}, α={args.lora_alpha})')

    precip_head = PrecipitationHead(in_channels=PANGU_N_SURFACE, hidden=64).to(device).to(precision)

    trainable = [p for p in list(pangu.parameters()) + list(precip_head.parameters())
                 if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = (sum(p.numel() for p in pangu.parameters())
               + sum(p.numel() for p in precip_head.parameters()))
    print(f'Trainable params: {n_train:,} / {n_total:,}  '
          f'({100 * n_train / n_total:.2f}%)')

    # ── loss & optimiser ──────────────────────────────────────────────────────
    criterion = PrecipLoss(device,
                           seeps_thresholds_path=args.seeps_thresholds,
                           seeps_contingency_path=args.seeps_contingency)

    optimizer = optim.AdamW(trainable, lr=config['training']['learning_rate'],
                            betas=(0.9, 0.98), weight_decay=0.05)
    scaler    = GradScaler('cuda')

    num_epochs  = config['training']['epochs']
    total_steps = num_epochs * len(train_loader)
    sched_lin = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-8, end_factor=1.0, total_iters=500)
    sched_cos = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps - 500, 1))
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, [sched_lin, sched_cos], milestones=[500])

    start_epoch, start_step = 1, 0
    if args.checkpoint:
        start_epoch, start_step = load_checkpoint(
            pangu, precip_head, optimizer, args.checkpoint)

    # ── training loop ─────────────────────────────────────────────────────────
    print('Start training')
    for epoch in range(start_epoch, num_epochs + 1):
        train_fkt(pangu, precip_head, train_loader, criterion, optimizer,
                  scheduler, scaler, device, precision, config, epoch,
                  start_step=start_step if epoch == start_epoch else 0)

        val_rmse = validate(pangu, precip_head, val_loader, criterion, device, precision)
        wandb.log({'epoch': epoch, 'val_rmse': val_rmse})
        print(f'Epoch {epoch} – val RMSE: {val_rmse:.5f}')


if __name__ == '__main__':
    import pkg_resources
    print("=== Environment ===")
    print(f"Python:  {sys.version}")
    print(f"PyTorch: {torch.__version__}  CUDA: {torch.version.cuda}")
    print(f"GPU:     {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    print(f"onnx2torch: {pkg_resources.get_distribution('onnx2torch').version}")
    main()
