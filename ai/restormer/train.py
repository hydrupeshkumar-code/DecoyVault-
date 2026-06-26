"""
Three-phase Restormer training pipeline.

Phase 1 – Warmup          (low LR, build stable gradient flow)
Phase 2 – Main Training   (full LR schedule, EMA)
Phase 3 – LISS-IV Fine-Tuning (domain-adapted data, optional adversarial)

Usage:
    python -m ai.restormer.train --config ai/restormer/train_config.yaml
    python -m ai.restormer.train --config ai/restormer/train_config.yaml --resume ai/restormer/checkpoints/last.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset, random_split

import yaml

# Allow running as a module from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ai.datasets.sen12ms_cr import SEN12MSCRDataset
from ai.datasets.liss4 import LISS4Dataset
# Legacy SEN2-CR dataset (kept for backward compat)
from ai.restormer.dataset import SEN2CRDataset
from ai.restormer.losses import CloudRemovalLoss, PatchDiscriminator
from ai.restormer.metrics import aggregate_metrics, compute_metrics, save_report
from ai.restormer.model import Restormer
from ai.restormer.residual_head import CloudAdaptiveResidualHead

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------

class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module, backup: Dict[str, torch.Tensor]) -> None:
        for name, param in model.named_parameters():
            param.data.copy_(backup[name])


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    epoch: int,
    phase: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    ema: EMA,
    metrics: Dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "phase": phase,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "ema_shadow": ema.shadow,
            "metrics": metrics,
        },
        path,
    )
    log.info("Checkpoint saved → %s", path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    ema: EMA,
    device: torch.device,
) -> tuple[int, int, Dict[str, float]]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
    ema.shadow = ckpt["ema_shadow"]
    log.info("Resumed from %s (epoch %d, phase %d)", path, ckpt["epoch"], ckpt["phase"])
    return ckpt["epoch"], ckpt["phase"], ckpt.get("metrics", {})


def _prune_checkpoints(ckpt_dir: Path, phase_id: int, keep_last_n: int) -> None:
    """Delete all but the newest keep_last_n milestone checkpoints for a phase.

    Only touches phase{N}_epoch*.pt files; last.pt and phase{N}_final.pt are
    never pruned. Keeps disk bounded over a long multi-epoch run.
    """
    if keep_last_n <= 0:
        return
    milestones = sorted(ckpt_dir.glob(f"phase{phase_id}_epoch*.pt"))
    for old in milestones[:-keep_last_n]:
        try:
            old.unlink()
            log.info("Pruned old checkpoint → %s", old)
        except OSError as e:
            log.warning("Could not prune %s: %s", old, e)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

class Trainer:
    """End-to-end Restormer trainer."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        resume_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        smoke: bool = False,
    ) -> None:
        self.cfg = cfg
        self.max_samples = max_samples
        self.smoke = smoke
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Using device: %s", self.device)
        if smoke:
            log.info("SMOKE MODE: tiny subset, few epochs, Phase 1 only — sanity check, not a real run.")

        seed = cfg.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        log.info("Random seed: %d", seed)

        self._build_model()
        self._build_dataloaders()

        # Resume metadata: peek at the checkpoint header so run() can skip
        # already-finished phases and restart the current phase at the right
        # epoch. Full state (model/optim/scheduler/scaler/ema) is restored later
        # inside _run_phase, once those objects exist.
        self.resume_path = resume_path
        self.resume_phase = 0
        self.resume_epoch = 0
        if resume_path:
            header = torch.load(resume_path, map_location="cpu")
            self.resume_phase = int(header.get("phase", 0))
            self.resume_epoch = int(header.get("epoch", 0))
            log.info(
                "Resume requested: %s → phase %d, %d epoch(s) completed",
                resume_path, self.resume_phase, self.resume_epoch,
            )

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(self) -> None:
        mcfg = self.cfg["model"]
        self.model = Restormer(
            in_channels=mcfg["in_channels"],
            out_channels=mcfg["out_channels"],
            dim=mcfg["dim"],
            num_blocks=mcfg["num_blocks"],
            num_refinement_blocks=mcfg["num_refinement_blocks"],
            heads=mcfg["heads"],
            ffn_expansion_factor=mcfg["ffn_expansion_factor"],
            bias=mcfg["bias"],
        ).to(self.device)

        rh = self.cfg.get("residual_head", {})
        self.res_head = CloudAdaptiveResidualHead(
            in_channels=mcfg["dim"],
            out_channels=mcfg["out_channels"],
            mid_channels=rh.get("mid_channels", 32),
        ).to(self.device) if rh.get("enabled", False) else None

        log.info(
            "Model parameters: %d M",
            sum(p.numel() for p in self.model.parameters()) // 1_000_000,
        )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _build_dataloaders(self) -> None:
        dcfg = self.cfg["data"]
        dataset_type = dcfg.get("dataset_type", "sen2cr")

        if dataset_type == "sen12ms_cr":
            full_ds = SEN12MSCRDataset(
                root_dir=dcfg["root_dir"],
                layout=dcfg.get("layout", "flat"),
                split="train",
                patch_size=dcfg["patch_size"],
                augment=True,
                min_cloud_fraction=dcfg.get("min_cloud_fraction", 0.02),
                max_cloud_fraction=dcfg.get("max_cloud_fraction", 0.95),
            )
            if dcfg.get("pair_quality_path"):
                full_ds.load_quality_scores(dcfg["pair_quality_path"])
        else:
            full_ds = SEN2CRDataset(
                root_dir=dcfg["root_dir"],
                split="train",
                patch_size=dcfg["patch_size"],
                augment=True,
            )

        # Optional cap (smoke runs / quick experiments).
        if self.max_samples and len(full_ds) > self.max_samples:
            full_ds = Subset(full_ds, list(range(self.max_samples)))
            log.info("Capped dataset to %d samples (--max-samples).", self.max_samples)

        if len(full_ds) < 2:
            raise RuntimeError(
                f"Need at least 2 samples to train, found {len(full_ds)}. "
                "Check root_dir / run ai.dataset_tools.prepare_dataset first."
            )

        val_size = max(1, int(len(full_ds) * 0.1))
        train_size = len(full_ds) - val_size
        train_ds, val_ds = random_split(full_ds, [train_size, val_size])

        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg["phase1"]["batch_size"],
            shuffle=True,
            num_workers=dcfg["num_workers"],
            pin_memory=dcfg["pin_memory"],
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=4,
            shuffle=False,
            num_workers=dcfg["num_workers"],
            pin_memory=dcfg["pin_memory"],
        )

    # ------------------------------------------------------------------
    # One training phase
    # ------------------------------------------------------------------

    def _run_phase(
        self,
        phase_id: int,
        phase_cfg: Dict[str, Any],
        start_epoch: int = 0,
    ) -> None:
        log.info("=== Phase %d ===", phase_id)
        lcfg = self.cfg["loss"]
        use_adv = phase_cfg.get("use_adversarial", lcfg.get("use_adversarial", False))

        criterion = CloudRemovalLoss(use_adversarial=use_adv).to(self.device)
        discriminator: Optional[PatchDiscriminator] = None
        disc_optim: Optional[torch.optim.Optimizer] = None

        if use_adv:
            discriminator = PatchDiscriminator().to(self.device)
            disc_optim = AdamW(discriminator.parameters(), lr=phase_cfg["lr"] * 0.5)

        params = list(self.model.parameters())
        if self.res_head is not None:
            params += list(self.res_head.parameters())

        optimizer = AdamW(
            params,
            lr=phase_cfg["lr"],
            betas=tuple(self.cfg["optimizer"]["betas"]),  # type: ignore[arg-type]
            weight_decay=self.cfg["optimizer"]["weight_decay"],
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=phase_cfg["epochs"],
            eta_min=self.cfg["scheduler"]["eta_min"],
        )
        amp_enabled = self.cfg["amp"]["enabled"]
        amp_dtype_str = self.cfg["amp"].get("dtype", "float16")
        amp_dtype = torch.bfloat16 if amp_dtype_str == "bfloat16" else torch.float16
        # GradScaler is only meaningful for float16 (which can overflow to Inf).
        # bfloat16 has float32's exponent range — no overflow, no scaling needed.
        use_scaler = amp_enabled and amp_dtype == torch.float16
        scaler = GradScaler(self.device.type, enabled=use_scaler)
        ema = EMA(self.model, decay=phase_cfg["ema_decay"])

        # Restore full training state only when resuming INTO the same phase we
        # checkpointed in (mid-phase continue). For a cross-phase resume
        # (e.g. phase1_final → phase 2) the model weights already persist in
        # self.model from the prior _run_phase call, and we deliberately start
        # the new phase with a FRESH optimizer + scheduler at its own LR.
        if self.resume_path and self.resume_phase == phase_id:
            load_checkpoint(
                self.resume_path,
                self.model,
                optimizer,
                scheduler,
                scaler,
                ema,
                self.device,
            )

        ckpt_dir = Path(self.cfg["checkpointing"]["checkpoint_dir"])
        log_every = self.cfg["logging"]["log_every_n_steps"]
        val_every = self.cfg["logging"]["val_every_n_epochs"]
        save_every = self.cfg["checkpointing"]["save_every_n_epochs"]
        keep_last_n = self.cfg["checkpointing"].get("keep_last_n", 5)

        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer: Optional[SummaryWriter] = SummaryWriter(
                log_dir=str(Path(self.cfg["logging"]["log_dir"]) / f"phase{phase_id}")
            )
        except ImportError:
            tb_writer = None

        global_step = 0

        for epoch in range(start_epoch, phase_cfg["epochs"]):
            self.model.train()
            if self.res_head:
                self.res_head.train()

            epoch_losses: list[float] = []

            for step, batch in enumerate(self.train_loader):
                cloudy = batch["cloudy"].to(self.device, non_blocking=True)
                clear = batch["clear"].to(self.device, non_blocking=True)
                mask = batch["mask"].to(self.device, non_blocking=True)

                with autocast(self.device.type, enabled=amp_enabled, dtype=amp_dtype):
                    if self.res_head is not None:
                        # Residual-head path: feed the *decoder feature map*
                        # (dim channels), NOT the 3-channel reconstruction, to
                        # the head. NOTE: when this path is enabled the head
                        # weights must also be persisted in checkpoints and
                        # applied at inference time, otherwise training and
                        # serving diverge. It is disabled by default in the
                        # config for exactly that reason.
                        feats = self.model.forward_features(cloudy)
                        pred = self.res_head(feats, cloudy_input=cloudy, cloud_mask=mask)
                    else:
                        pred = self.model(cloudy)
                    losses = criterion(pred, clear, discriminator)

                optimizer.zero_grad()
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                # clip_grad_norm_ returns the total norm BEFORE clipping; if any
                # gradient is NaN/Inf the returned norm is non-finite. Under AMP
                # GradScaler skips such steps automatically, but smoke/CPU runs
                # have AMP off — so guard explicitly. A single bad batch must not
                # corrupt the weights and turn every subsequent forward to NaN.
                grad_norm = torch.nn.utils.clip_grad_norm_(params, phase_cfg["grad_clip"])
                if torch.isfinite(grad_norm):
                    scaler.step(optimizer)
                    scaler.update()
                    ema.update(self.model)
                else:
                    scaler.update()  # keep AMP scale state consistent
                    optimizer.zero_grad(set_to_none=True)
                    log.warning(
                        "Phase %d | Epoch %d | Step %d | non-finite gradient — step skipped",
                        phase_id, epoch + 1, step,
                    )
                    continue

                # Discriminator update
                if use_adv and discriminator is not None and disc_optim is not None:
                    real = torch.cat([cloudy, clear], dim=1)
                    fake = torch.cat([cloudy, pred.detach()], dim=1)
                    d_real = discriminator(real)
                    d_fake = discriminator(fake)
                    d_loss = 0.5 * (
                        torch.mean((d_real - 1.0) ** 2) + torch.mean(d_fake ** 2)
                    )
                    disc_optim.zero_grad()
                    d_loss.backward()
                    disc_optim.step()

                epoch_losses.append(losses["total"].item())
                global_step += 1

                if step % log_every == 0:
                    log.info(
                        "Phase %d | Epoch %d/%d | Step %d | Loss %.4f",
                        phase_id,
                        epoch + 1,
                        phase_cfg["epochs"],
                        step,
                        losses["total"].item(),
                    )
                    if tb_writer is not None:
                        tb_writer.add_scalar("train/loss_total", losses["total"].item(), global_step)
                        for k, v in losses.items():
                            if k != "total":
                                tb_writer.add_scalar(f"train/loss_{k}", v.item(), global_step)

            scheduler.step()
            mean_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            current_lr = optimizer.param_groups[0]["lr"]
            log.info(
                "Phase %d | Epoch %d | Mean Loss %.4f | LR %.2e",
                phase_id,
                epoch + 1,
                mean_loss,
                current_lr,
            )
            if tb_writer is not None:
                tb_writer.add_scalar(f"train/epoch_loss", mean_loss, epoch)
                tb_writer.add_scalar(f"train/lr", current_lr, epoch)

            # Validation
            if (epoch + 1) % val_every == 0:
                val_metrics = self._validate(ema)
                log.info("Val metrics: %s", val_metrics)
                save_report(
                    val_metrics,
                    Path(self.cfg["logging"]["metrics_report"]),
                )
                if tb_writer is not None:
                    for k, v in val_metrics.items():
                        tb_writer.add_scalar(f"val/{k}", v, epoch)

            # Crash-safe resume: overwrite a single rolling checkpoint EVERY
            # epoch. Stopping the run (Ctrl+C, reboot, power loss) costs at most
            # one epoch; resume with --resume <ckpt_dir>/last.pt.
            save_checkpoint(
                ckpt_dir / "last.pt",
                epoch=epoch + 1,
                phase=phase_id,
                model=self.model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                ema=ema,
                metrics={},
            )

            # Milestone checkpoints at the configured interval, pruned to the
            # last keep_last_n so disk stays bounded over a long run.
            if (epoch + 1) % save_every == 0:
                save_checkpoint(
                    ckpt_dir / f"phase{phase_id}_epoch{epoch + 1:04d}.pt",
                    epoch=epoch + 1,
                    phase=phase_id,
                    model=self.model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    ema=ema,
                    metrics={},
                )
                _prune_checkpoints(ckpt_dir, phase_id, keep_last_n)

        # Save final checkpoint for this phase
        save_checkpoint(
            ckpt_dir / f"phase{phase_id}_final.pt",
            epoch=phase_cfg["epochs"],
            phase=phase_id,
            model=self.model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
            metrics={},
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate(self, ema: EMA) -> Dict[str, float]:
        # Swap in EMA weights
        backup = {n: p.data.clone() for n, p in self.model.named_parameters()}
        ema.apply_shadow(self.model)
        self.model.eval()

        batch_metrics: list[Dict[str, float]] = []
        for batch in self.val_loader:
            cloudy = batch["cloudy"].to(self.device)
            clear = batch["clear"].to(self.device)
            mask = batch["mask"].to(self.device)

            pred = self.model(cloudy)
            metrics = compute_metrics(pred, clear, cloudy, mask)
            batch_metrics.append(metrics)

        result = aggregate_metrics(batch_metrics)

        # Restore original weights
        ema.restore(self.model, backup)
        return result

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        rp = self.resume_phase  # 0 when not resuming

        # Phase 1 — skipped outright if the checkpoint is already past it.
        if rp <= 1:
            self._run_phase(1, self.cfg["phase1"],
                            start_epoch=self.resume_epoch if rp == 1 else 0)

        if self.smoke:
            log.info("SMOKE MODE complete — Phase 1 ran end-to-end. Skipping Phases 2/3.")
            return

        # Phase 2
        if rp <= 2:
            self._run_phase(2, self.cfg["phase2"],
                            start_epoch=self.resume_epoch if rp == 2 else 0)

        # Phase 3: fine-tune on real LISS-IV data
        p3 = self.cfg.get("phase3", {})
        if p3.get("liss4_root") and Path(p3["liss4_root"]).exists():
            liss4_ds = LISS4Dataset(
                root_dir=p3["liss4_root"],
                split="train",
                patch_size=self.cfg["data"]["patch_size"],
                augment=True,
                min_cloud_fraction=p3.get("min_cloud_fraction", 0.02),
                max_cloud_fraction=p3.get("max_cloud_fraction", 0.95),
            )
            self.train_loader = DataLoader(
                liss4_ds,
                batch_size=p3["batch_size"],
                shuffle=True,
                num_workers=self.cfg["data"]["num_workers"],
                pin_memory=True,
                drop_last=True,
            )
            self._run_phase(3, p3, start_epoch=self.resume_epoch if rp == 3 else 0)
        else:
            log.info("Phase 3 skipped (no LISS-IV data found at %s)", p3.get("liss4_root"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Restormer for cloud removal")
    parser.add_argument("--config", default="ai/restormer/train_config.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap the dataset to the first N samples (quick experiments).")
    parser.add_argument("--smoke", action="store_true",
                        help="Sanity run: tiny subset, few epochs, Phase 1 only, AMP off. "
                             "Confirms the loss descends and nothing NaNs before a full run.")
    return parser.parse_args()


def _apply_smoke_overrides(cfg: Dict[str, Any]) -> None:
    """Shrink a config in-place for a fast CPU/GPU sanity run."""
    cfg.setdefault("phase1", {})
    cfg["phase1"]["epochs"] = 2
    cfg["phase1"]["batch_size"] = 2
    cfg.setdefault("amp", {})["enabled"] = False          # AMP/GradScaler is CUDA-only
    cfg.setdefault("data", {})["num_workers"] = 0          # deterministic, no worker overhead
    cfg.setdefault("logging", {})["val_every_n_epochs"] = 1
    cfg.setdefault("logging", {})["log_every_n_steps"] = 1
    cfg.setdefault("checkpointing", {})["save_every_n_epochs"] = 1


def main() -> None:
    args = _parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    max_samples = args.max_samples
    if args.smoke:
        _apply_smoke_overrides(cfg)
        if max_samples is None:
            max_samples = 64

    trainer = Trainer(cfg, resume_path=args.resume, max_samples=max_samples, smoke=args.smoke)
    trainer.run()


if __name__ == "__main__":
    main()
