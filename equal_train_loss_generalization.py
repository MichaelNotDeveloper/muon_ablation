#!/usr/bin/env python3
"""
Equal-train-loss generalization experiment for nanoGPT.

The experiment compares AdamW and hybrid MuonNS/AdamW at fixed train-loss
targets, with all hyperparameters fixed before the run:

1. Train both optimizers from the same seeds and save dense checkpoints.
2. Re-evaluate every checkpoint on fixed train/validation batches.
3. For each target, take the first checkpoint whose train loss is <= target.
4. Optionally run layer-wise Hessian-top diagnostics at the matched iterations.

The Hessian diagnostic remains same-state within each checkpoint. The main
generalization comparison is validation loss at equal train loss. Validation is
used only for reporting, never for checkpoint or hyperparameter selection.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from nanogpt_layerwise_hessian_top import (
    ModelArgs,
    estimate_loss,
    force_manual_attention,
    get_batch,
    load_memmap,
    load_nanogpt_model_module,
    make_model,
)


LAYER_SUFFIXES = [
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
]

OPTIMIZER_COLORS = {
    "adamw": "#1f77b4",
    "muon_ns": "#d62728",
}


@dataclass(frozen=True)
class RunConfig:
    optimizer: str
    learning_rate: float
    weight_decay: float
    beta2: float
    aux_learning_rate: Optional[float] = None
    muon_momentum: float = 0.95
    ns_steps: int = 5


def fmt_float(x: float) -> str:
    return f"{float(x):g}".replace(".", "p").replace("-", "m")


def config_id(cfg: RunConfig) -> str:
    parts = [
        cfg.optimizer,
        f"lr{fmt_float(cfg.learning_rate)}",
        f"wd{fmt_float(cfg.weight_decay)}",
        f"b2{fmt_float(cfg.beta2)}",
    ]
    if cfg.optimizer == "muon_ns":
        aux = cfg.learning_rate if cfg.aux_learning_rate is None else cfg.aux_learning_rate
        parts.extend([f"aux{fmt_float(aux)}", f"mom{fmt_float(cfg.muon_momentum)}", f"ns{cfg.ns_steps}"])
    return "_".join(parts)


def csv_write(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def csv_read(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    xs = [float(v) for v in values if math.isfinite(float(v))]
    if not xs:
        return float("nan"), float("nan")
    mean = sum(xs) / len(xs)
    if len(xs) == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return mean, math.sqrt(max(var, 0.0))


def checkpoint_iters(max_iters: int, interval: int) -> List[int]:
    values = set(range(0, int(max_iters) + 1, int(interval)))
    values.add(int(max_iters))
    return sorted(values)


def layers_for_blocks(blocks: Sequence[int]) -> List[str]:
    out: List[str] = []
    for block in blocks:
        for suffix in LAYER_SUFFIXES:
            out.append(f"transformer.h.{int(block)}.{suffix}")
    return out


def default_configs(args: argparse.Namespace) -> List[RunConfig]:
    configs: List[RunConfig] = []
    if "adamw" in args.optimizers:
        configs.append(RunConfig("adamw", args.adamw_lr, args.adamw_weight_decay, args.adamw_beta2))
    if "muon_ns" in args.optimizers:
        configs.append(
            RunConfig(
                "muon_ns",
                args.muon_lr,
                args.muon_weight_decay,
                args.muon_beta2,
                aux_learning_rate=args.muon_aux_lr,
                muon_momentum=args.muon_momentum,
                ns_steps=args.ns_steps,
            )
        )
    return configs


def run_dir(args: argparse.Namespace, cfg: RunConfig) -> Path:
    return Path(args.base_out_dir).expanduser() / config_id(cfg)


def run_cmd(cmd: Sequence[str], *, dry_run: bool) -> None:
    print(" ".join(str(x) for x in cmd), flush=True)
    if not dry_run:
        subprocess.run([str(x) for x in cmd], check=True)


def train_configs(args: argparse.Namespace, configs: Sequence[RunConfig]) -> None:
    iters = checkpoint_iters(args.max_iters, args.checkpoint_interval)
    for cfg in configs:
        cmd: List[Any] = [
            sys.executable,
            args.train_script,
            "--nanogpt_dir",
            args.nanogpt_dir,
            "--mode",
            "train",
            "--out_dir",
            run_dir(args, cfg),
            "--train_optimizer",
            cfg.optimizer,
            "--device",
            args.device,
            "--seeds",
            *args.seeds,
            "--diag_iters",
            *iters,
            "--n_layer",
            args.n_layer,
            "--n_head",
            args.n_head,
            "--n_embd",
            args.n_embd,
            "--block_size",
            args.block_size,
            "--batch_size",
            args.batch_size,
            "--dropout",
            args.dropout,
            "--max_iters",
            args.max_iters,
            "--learning_rate",
            cfg.learning_rate,
            "--min_lr",
            cfg.learning_rate * args.min_lr_ratio,
            "--lr_decay_iters",
            args.max_iters,
            "--warmup_iters",
            args.warmup_iters,
            "--weight_decay",
            cfg.weight_decay,
            "--beta1",
            args.beta1,
            "--beta2",
            cfg.beta2,
            "--adam_eps",
            args.adam_eps,
            "--grad_clip",
            args.grad_clip,
            "--eval_interval",
            args.eval_interval,
            "--eval_iters",
            args.train_eval_iters,
            "--log_interval",
            args.log_interval,
            "--ns_steps",
            cfg.ns_steps,
        ]
        if args.early_stop_train_loss is not None:
            cmd.extend(
                [
                    "--early_stop_train_loss",
                    args.early_stop_train_loss,
                    "--early_stop_patience",
                    args.early_stop_patience,
                ]
            )
        if args.bias:
            cmd.append("--bias")
        if args.no_decay_lr:
            cmd.append("--no_decay_lr")
        if cfg.optimizer == "muon_ns":
            cmd.extend(
                [
                    "--aux_learning_rate",
                    cfg.aux_learning_rate if cfg.aux_learning_rate is not None else cfg.learning_rate,
                    "--muon_momentum",
                    cfg.muon_momentum,
                    "--adamw_state_mode",
                    "warmup",
                    "--adamw_warmup_steps",
                    args.adamw_warmup_steps,
                ]
            )
        run_cmd(cmd, dry_run=args.dry_run)


def load_checkpoint_model(
    *,
    model_module: Any,
    ckpt_path: Path,
    device: str,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model_args = ModelArgs(**checkpoint["model_args"])
    model = make_model(model_module, model_args, device)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    if any(k.startswith(unwanted_prefix) for k in state_dict):
        state_dict = {k[len(unwanted_prefix) :] if k.startswith(unwanted_prefix) else k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    force_manual_attention(model, model_args.block_size, device)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def estimate_loss_accuracy(
    model: torch.nn.Module,
    train_data: Any,
    val_data: Any,
    *,
    batch_size: int,
    block_size: int,
    device: str,
    eval_iters: int,
    seed: int,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    out: Dict[str, float] = {}
    for split, data in (("train", train_data), ("val", val_data)):
        losses: List[float] = []
        correct = 0
        total = 0
        gen = torch.Generator()
        gen.manual_seed(seed + (0 if split == "train" else 10_000))
        for _ in range(eval_iters):
            x, y = get_batch(
                data,
                batch_size=batch_size,
                block_size=block_size,
                device=device,
                generator=gen,
            )
            logits, loss = model(x, y)
            losses.append(float(loss.detach().cpu()))
            pred = torch.argmax(logits.detach(), dim=-1)
            correct += int((pred == y).sum().detach().cpu())
            total += int(y.numel())
        out[f"{split}_loss"] = sum(losses) / max(1, len(losses))
        out[f"{split}_acc"] = float(correct) / float(max(1, total))
    if was_training:
        model.train()
    return out


def evaluate_checkpoints(args: argparse.Namespace, configs: Sequence[RunConfig]) -> None:
    model_module = load_nanogpt_model_module(Path(args.nanogpt_dir))
    data_dir = Path(args.nanogpt_dir) / "data" / args.dataset
    train_data = load_memmap(data_dir, "train")
    val_data = load_memmap(data_dir, "val")
    rows: List[Dict[str, Any]] = []
    for cfg in configs:
        cid = config_id(cfg)
        for seed in args.seeds:
            seed_dir = run_dir(args, cfg) / f"seed{seed}"
            for iter_num in checkpoint_iters(args.max_iters, args.checkpoint_interval):
                ckpt = seed_dir / f"ckpt_iter_{iter_num}.pt"
                if not ckpt.exists():
                    continue
                model, checkpoint = load_checkpoint_model(
                    model_module=model_module,
                    ckpt_path=ckpt,
                    device=args.device,
                )
                metrics = estimate_loss_accuracy(
                    model,
                    train_data,
                    val_data,
                    batch_size=args.eval_batch_size,
                    block_size=checkpoint["model_args"]["block_size"],
                    device=args.device,
                    eval_iters=args.eval_iters,
                    seed=args.eval_seed + int(seed) * 10_000 + int(iter_num),
                )
                row = {
                    "config_id": cid,
                    "optimizer": cfg.optimizer,
                    "seed": int(seed),
                    "iter": int(iter_num),
                    "elapsed_sec": float(checkpoint.get("elapsed_sec", float("nan"))),
                    "wall_time_min": float(checkpoint.get("elapsed_sec", float("nan"))) / 60.0,
                    "train_loss": metrics["train_loss"],
                    "val_loss": metrics["val_loss"],
                    "gap": metrics["val_loss"] - metrics["train_loss"],
                    "train_acc": metrics["train_acc"],
                    "val_acc": metrics["val_acc"],
                    "checkpoint": str(ckpt),
                    "learning_rate": cfg.learning_rate,
                    "aux_learning_rate": "" if cfg.aux_learning_rate is None else cfg.aux_learning_rate,
                    "weight_decay": cfg.weight_decay,
                    "beta2": cfg.beta2,
                }
                rows.append(row)
                print(
                    f"eval {cid} seed {seed} iter {iter_num}: "
                    f"train={metrics['train_loss']:.4f} val={metrics['val_loss']:.4f} "
                    f"val_acc={metrics['val_acc']:.4f}",
                    flush=True,
                )
                del model
                if str(args.device).startswith("cuda"):
                    torch.cuda.empty_cache()

    fields = [
        "config_id",
        "optimizer",
        "seed",
        "iter",
        "elapsed_sec",
        "wall_time_min",
        "train_loss",
        "val_loss",
        "gap",
        "train_acc",
        "val_acc",
        "checkpoint",
        "learning_rate",
        "aux_learning_rate",
        "weight_decay",
        "beta2",
    ]
    csv_write(Path(args.base_out_dir) / "eval_history.csv", rows, fields)


def first_reached_row(rows: Sequence[Dict[str, Any]], target: float) -> Optional[Dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: int(row["iter"]))
    for row in ordered:
        if float(row["train_loss"]) <= float(target):
            return row
    return None


def match_checkpoints(args: argparse.Namespace) -> None:
    rows = csv_read(Path(args.base_out_dir) / "eval_history.csv")
    by_seed_opt: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed_opt[(int(row["seed"]), row["optimizer"])].append(row)

    matched: List[Dict[str, Any]] = []
    for seed in args.seeds:
        for target in args.match_train_losses:
            for optimizer in args.optimizers:
                candidates = by_seed_opt.get((int(seed), optimizer), [])
                ordered = sorted(candidates, key=lambda candidate: int(candidate["iter"]))
                row = first_reached_row(ordered, float(target))
                reached = row is not None
                if row is None and ordered:
                    row = ordered[-1]
                if row is None:
                    continue
                out = dict(row)
                out["target_train_loss"] = float(target)
                out["target_reached"] = bool(reached)
                out["target_minus_train_loss"] = float(target) - float(row["train_loss"])
                out["match_ok"] = bool(reached)
                matched.append(out)

    fields = [
        "target_train_loss",
        "match_ok",
        "target_reached",
        "target_minus_train_loss",
        "config_id",
        "optimizer",
        "seed",
        "iter",
        "elapsed_sec",
        "wall_time_min",
        "train_loss",
        "val_loss",
        "gap",
        "train_acc",
        "val_acc",
        "checkpoint",
        "learning_rate",
        "aux_learning_rate",
        "weight_decay",
        "beta2",
    ]
    csv_write(Path(args.base_out_dir) / "matched_checkpoints.csv", matched, fields)
    summarize_matched(Path(args.base_out_dir) / "matched_checkpoints.csv", Path(args.base_out_dir) / "matched_summary.csv")


def summarize_matched(in_path: Path, out_path: Path) -> None:
    rows = csv_read(in_path)
    groups: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if str(row.get("match_ok", "True")) not in {"True", "true", "1"}:
            continue
        key = (str(row["target_train_loss"]), row["optimizer"])
        for metric in [
            "iter",
            "wall_time_min",
            "train_loss",
            "val_loss",
            "gap",
            "train_acc",
            "val_acc",
            "target_minus_train_loss",
        ]:
            groups[key][metric].append(float(row[metric]))

    out: List[Dict[str, Any]] = []
    for (target, optimizer), rec in sorted(groups.items(), key=lambda item: (float(item[0][0]), item[0][1])):
        row: Dict[str, Any] = {
            "target_train_loss": float(target),
            "optimizer": optimizer,
            "n": len(rec["train_loss"]),
        }
        for metric, values in rec.items():
            mean, std = mean_std(values)
            row[f"mean_{metric}"] = mean
            row[f"std_{metric}"] = std
        out.append(row)
    fields = [
        "target_train_loss",
        "optimizer",
        "n",
        "mean_iter",
        "std_iter",
        "mean_wall_time_min",
        "std_wall_time_min",
        "mean_train_loss",
        "std_train_loss",
        "mean_val_loss",
        "std_val_loss",
        "mean_gap",
        "std_gap",
        "mean_train_acc",
        "std_train_acc",
        "mean_val_acc",
        "std_val_acc",
        "mean_target_minus_train_loss",
        "std_target_minus_train_loss",
    ]
    csv_write(out_path, out, fields)


def finite_bounds(values: Sequence[float]) -> Tuple[float, float]:
    xs = [float(v) for v in values if math.isfinite(float(v))]
    if not xs:
        return 0.0, 1.0
    lo, hi = min(xs), max(xs)
    if lo == hi:
        pad = max(abs(lo) * 0.05, 1e-3)
        return lo - pad, hi + pad
    pad = 0.06 * (hi - lo)
    return lo - pad, hi + pad


def svg_polyline(points: Sequence[Tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def write_svg_plot(
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: Sequence[Dict[str, Any]],
    width: int = 960,
    height: int = 560,
) -> None:
    usable = [s for s in series if s.get("points")]
    if not usable:
        return
    xs = [float(x) for s in usable for x, _ in s["points"]]
    ys = [float(y) for s in usable for _, y in s["points"]]
    x_min, x_max = finite_bounds(xs)
    y_min, y_max = finite_bounds(ys)
    left, right, top, bottom = 82, 230, 54, 78
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(x: float) -> float:
        return left + (float(x) - x_min) / max(x_max - x_min, 1e-30) * plot_w

    def sy(y: float) -> float:
        return top + (y_max - float(y)) / max(y_max - y_min, 1e-30) * plot_h

    ticks = 5
    x_ticks = [x_min + i * (x_max - x_min) / (ticks - 1) for i in range(ticks)]
    y_ticks = [y_min + i * (y_max - y_min) / (ticks - 1) for i in range(ticks)]
    lines: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>"
        "text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}"
        ".title{font-size:20px;font-weight:700}"
        ".axis{stroke:#334155;stroke-width:1.2}"
        ".grid{stroke:#d8dee9;stroke-width:1}"
        ".tick{font-size:12px;fill:#475569}"
        ".label{font-size:14px;font-weight:600}"
        ".legend{font-size:13px}"
        "</style>",
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
        f'<text class="title" x="{left}" y="30">{html.escape(title)}</text>',
    ]
    for xt in x_ticks:
        x = sx(xt)
        lines.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}"/>')
        lines.append(f'<text class="tick" x="{x:.2f}" y="{top + plot_h + 24}" text-anchor="middle">{xt:.3g}</text>')
    for yt in y_ticks:
        y = sy(yt)
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}"/>')
        lines.append(f'<text class="tick" x="{left - 12}" y="{y + 4:.2f}" text-anchor="end">{yt:.3g}</text>')
    lines.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>',
            f'<text class="label" x="{left + plot_w / 2:.2f}" y="{height - 24}" text-anchor="middle">{html.escape(x_label)}</text>',
            f'<text class="label" transform="translate(24 {top + plot_h / 2:.2f}) rotate(-90)" text-anchor="middle">{html.escape(y_label)}</text>',
        ]
    )
    for idx, s in enumerate(usable):
        color = s.get("color", "#111827")
        dash = ' stroke-dasharray="7 5"' if s.get("dash") else ""
        screen_points = [(sx(x), sy(y)) for x, y in s["points"]]
        lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.4"{dash} '
            f'points="{svg_polyline(screen_points)}"/>'
        )
        for x, y in screen_points:
            lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"/>')
        ly = top + 24 + idx * 24
        lx = left + plot_w + 34
        lines.append(f'<line x1="{lx}" y1="{ly - 4}" x2="{lx + 28}" y2="{ly - 4}" stroke="{color}" stroke-width="2.4"{dash}/>')
        lines.append(f'<text class="legend" x="{lx + 38}" y="{ly}">{html.escape(str(s["label"]))}</text>')
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def grouped_eval_means(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, int], Dict[str, float]]:
    groups: Dict[Tuple[str, int], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["optimizer"], int(row["iter"]))
        for metric in ["train_loss", "val_loss", "gap", "train_acc", "val_acc", "wall_time_min"]:
            if metric in row and row[metric] != "":
                groups[key][metric].append(float(row[metric]))
    out: Dict[Tuple[str, int], Dict[str, float]] = {}
    for key, rec in groups.items():
        out[key] = {metric: mean_std(values)[0] for metric, values in rec.items()}
    return out


def plot_eval_history(args: argparse.Namespace) -> None:
    history_path = Path(args.base_out_dir) / "eval_history.csv"
    if not history_path.exists():
        return
    rows = csv_read(history_path)
    means = grouped_eval_means(rows)
    out_dir = Path(args.base_out_dir) / "plots"
    loss_series: List[Dict[str, Any]] = []
    wall_loss_series: List[Dict[str, Any]] = []
    val_train_series: List[Dict[str, Any]] = []
    gap_series: List[Dict[str, Any]] = []
    wall_gap_series: List[Dict[str, Any]] = []
    val_acc_iter_series: List[Dict[str, Any]] = []
    val_acc_train_series: List[Dict[str, Any]] = []
    val_acc_wall_series: List[Dict[str, Any]] = []
    for optimizer in sorted({row["optimizer"] for row in rows}):
        color = OPTIMIZER_COLORS.get(optimizer, "#111827")
        items = sorted(
            [(it, vals) for (opt, it), vals in means.items() if opt == optimizer],
            key=lambda item: item[0],
        )
        loss_series.append(
            {
                "label": f"{optimizer} train",
                "points": [(it, vals["train_loss"]) for it, vals in items],
                "color": color,
            }
        )
        loss_series.append(
            {
                "label": f"{optimizer} val",
                "points": [(it, vals["val_loss"]) for it, vals in items],
                "color": color,
                "dash": True,
            }
        )
        wall_items = [(vals.get("wall_time_min", float("nan")), vals) for _, vals in items]
        wall_items = [(t, vals) for t, vals in wall_items if math.isfinite(float(t))]
        wall_loss_series.append(
            {
                "label": f"{optimizer} train",
                "points": [(t, vals["train_loss"]) for t, vals in wall_items],
                "color": color,
            }
        )
        wall_loss_series.append(
            {
                "label": f"{optimizer} val",
                "points": [(t, vals["val_loss"]) for t, vals in wall_items],
                "color": color,
                "dash": True,
            }
        )
        val_train_series.append(
            {
                "label": optimizer,
                "points": [(vals["train_loss"], vals["val_loss"]) for _, vals in items],
                "color": color,
            }
        )
        gap_series.append(
            {
                "label": optimizer,
                "points": [(vals["train_loss"], vals["gap"]) for _, vals in items],
                "color": color,
            }
        )
        wall_gap_series.append(
            {
                "label": optimizer,
                "points": [(t, vals["gap"]) for t, vals in wall_items],
                "color": color,
            }
        )
        val_acc_iter_series.append(
            {
                "label": optimizer,
                "points": [(it, vals["val_acc"]) for it, vals in items if "val_acc" in vals],
                "color": color,
            }
        )
        val_acc_train_series.append(
            {
                "label": optimizer,
                "points": [
                    (vals["train_loss"], vals["val_acc"])
                    for _, vals in items
                    if "train_loss" in vals and "val_acc" in vals
                ],
                "color": color,
            }
        )
        val_acc_wall_series.append(
            {
                "label": optimizer,
                "points": [(t, vals["val_acc"]) for t, vals in wall_items if "val_acc" in vals],
                "color": color,
            }
        )
    write_svg_plot(
        out_dir / "train_val_loss_vs_iter.svg",
        title="Train and validation loss over training",
        x_label="iteration",
        y_label="loss",
        series=loss_series,
    )
    write_svg_plot(
        out_dir / "val_loss_vs_train_loss.svg",
        title="Validation loss at equal train loss",
        x_label="train loss",
        y_label="validation loss",
        series=val_train_series,
    )
    write_svg_plot(
        out_dir / "train_val_loss_vs_wall_time.svg",
        title="Train and validation loss over wall-clock time",
        x_label="wall-clock time (minutes)",
        y_label="loss",
        series=wall_loss_series,
    )
    write_svg_plot(
        out_dir / "gap_vs_train_loss.svg",
        title="Generalization gap at equal train loss",
        x_label="train loss",
        y_label="validation - train",
        series=gap_series,
    )
    write_svg_plot(
        out_dir / "gap_vs_wall_time.svg",
        title="Generalization gap over wall-clock time",
        x_label="wall-clock time (minutes)",
        y_label="validation - train",
        series=wall_gap_series,
    )
    write_svg_plot(
        out_dir / "val_acc_vs_iter.svg",
        title="Validation accuracy over training",
        x_label="iteration",
        y_label="validation accuracy",
        series=val_acc_iter_series,
    )
    write_svg_plot(
        out_dir / "val_acc_vs_train_loss.svg",
        title="Validation accuracy at equal train loss",
        x_label="train loss",
        y_label="validation accuracy",
        series=val_acc_train_series,
    )
    write_svg_plot(
        out_dir / "val_acc_vs_wall_time.svg",
        title="Validation accuracy over wall-clock time",
        x_label="wall-clock time (minutes)",
        y_label="validation accuracy",
        series=val_acc_wall_series,
    )


def plot_matched_summary(args: argparse.Namespace) -> None:
    summary_path = Path(args.base_out_dir) / "matched_summary.csv"
    if not summary_path.exists():
        return
    rows = csv_read(summary_path)
    out_dir = Path(args.base_out_dir) / "plots"
    val_series: List[Dict[str, Any]] = []
    gap_series: List[Dict[str, Any]] = []
    acc_series: List[Dict[str, Any]] = []
    for optimizer in sorted({row["optimizer"] for row in rows}):
        color = OPTIMIZER_COLORS.get(optimizer, "#111827")
        items = sorted([row for row in rows if row["optimizer"] == optimizer], key=lambda row: float(row["target_train_loss"]))
        val_series.append(
            {
                "label": f"{optimizer} val",
                "points": [(float(row["target_train_loss"]), float(row["mean_val_loss"])) for row in items],
                "color": color,
            }
        )
        gap_series.append(
            {
                "label": f"{optimizer} gap",
                "points": [(float(row["target_train_loss"]), float(row["mean_gap"])) for row in items],
                "color": color,
            }
        )
        acc_series.append(
            {
                "label": f"{optimizer} val acc",
                "points": [(float(row["target_train_loss"]), float(row["mean_val_acc"])) for row in items],
                "color": color,
            }
        )
    write_svg_plot(
        out_dir / "matched_val_loss_vs_target_train_loss.svg",
        title="Matched checkpoints: validation loss",
        x_label="target train loss",
        y_label="mean validation loss",
        series=val_series,
    )
    write_svg_plot(
        out_dir / "matched_gap_vs_target_train_loss.svg",
        title="Matched checkpoints: generalization gap",
        x_label="target train loss",
        y_label="mean validation - train",
        series=gap_series,
    )
    write_svg_plot(
        out_dir / "matched_val_acc_vs_target_train_loss.svg",
        title="Matched checkpoints: validation accuracy",
        x_label="target train loss",
        y_label="mean validation accuracy",
        series=acc_series,
    )


def run_hessian_diagnostics(args: argparse.Namespace, configs: Sequence[RunConfig]) -> None:
    matched = [row for row in csv_read(Path(args.base_out_dir) / "matched_checkpoints.csv") if row.get("match_ok") in {"True", "true", "1"}]
    iters_by_config: Dict[str, set[int]] = defaultdict(set)
    for row in matched:
        iters_by_config[row["config_id"]].add(int(row["iter"]))
    layers = args.hessian_layers if args.hessian_layers else layers_for_blocks(args.hessian_layer_blocks)
    for cfg in configs:
        cid = config_id(cfg)
        iters = sorted(iters_by_config.get(cid, set()))
        if not iters:
            continue
        cmd: List[Any] = [
            sys.executable,
            args.train_script,
            "--nanogpt_dir",
            args.nanogpt_dir,
            "--mode",
            "diag",
            "--out_dir",
            run_dir(args, cfg),
            "--device",
            args.device,
            "--seeds",
            *args.seeds,
            "--diag_iters",
            *iters,
            "--layers",
            *layers,
            "--k_values",
            "10",
            "--lanczos_steps",
            args.hessian_lanczos_steps,
            "--num_diag_batches",
            args.hessian_num_diag_batches,
            "--diag_batch_size",
            args.hessian_diag_batch_size,
            "--diag_eval_iters",
            args.hessian_diag_eval_iters,
            "--ns_steps",
            cfg.ns_steps,
        ]
        if cfg.optimizer == "muon_ns":
            cmd.extend(["--adamw_state_mode", "warmup", "--adamw_warmup_steps", args.adamw_warmup_steps])
        else:
            cmd.extend(["--adamw_state_mode", "checkpoint"])
        run_cmd(cmd, dry_run=args.dry_run)
    summarize_matched_hessian(args, configs)


def summarize_matched_hessian(args: argparse.Namespace, configs: Sequence[RunConfig]) -> None:
    matched = [row for row in csv_read(Path(args.base_out_dir) / "matched_checkpoints.csv") if row.get("match_ok") in {"True", "true", "1"}]
    metrics_by_key: Dict[Tuple[str, int, int, str], List[Dict[str, Any]]] = defaultdict(list)
    actual_optimizer = {"adamw": "AdamW-full", "muon_ns": f"MuonNS{args.ns_steps}"}
    for cfg in configs:
        metrics_path = run_dir(args, cfg) / "layerwise_hessian_top_metrics.csv"
        if not metrics_path.exists():
            continue
        for row in csv_read(metrics_path):
            if row.get("eigenspace", "top") != "top" or row.get("k") != "10":
                continue
            opt_name = actual_optimizer.get(cfg.optimizer)
            if row.get("optimizer") != opt_name:
                continue
            key = (config_id(cfg), int(row["seed"]), int(row["iter"]), cfg.optimizer)
            metrics_by_key[key].append(row)

    rows: List[Dict[str, Any]] = []
    for match in matched:
        key = (match["config_id"], int(match["seed"]), int(match["iter"]), match["optimizer"])
        recs = metrics_by_key.get(key, [])
        out = dict(match)
        out["hessian_n_layers"] = len(recs)
        for metric in ["C", "rho_over_gd", "q_eff_over_gd"]:
            mean, std = mean_std([float(row[metric]) for row in recs])
            out[f"mean_{metric}"] = mean
            out[f"std_{metric}"] = std
        rows.append(out)
    fields = [
        "target_train_loss",
        "match_ok",
        "target_reached",
        "target_minus_train_loss",
        "config_id",
        "optimizer",
        "seed",
        "iter",
        "elapsed_sec",
        "wall_time_min",
        "train_loss",
        "val_loss",
        "gap",
        "train_acc",
        "val_acc",
        "checkpoint",
        "hessian_n_layers",
        "mean_C",
        "std_C",
        "mean_rho_over_gd",
        "std_rho_over_gd",
        "mean_q_eff_over_gd",
        "std_q_eff_over_gd",
    ]
    csv_write(Path(args.base_out_dir) / "matched_hessian_metrics.csv", rows, fields)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nanogpt_dir", required=True)
    p.add_argument("--base_out_dir", required=True)
    p.add_argument("--train_script", default=str(Path(__file__).with_name("nanogpt_layerwise_hessian_top.py")))
    p.add_argument("--stage", choices=["train", "eval", "match", "plots", "hessian", "all"], default="all")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dataset", default="shakespeare_char")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--optimizers", nargs="+", choices=["adamw", "muon_ns"], default=["adamw", "muon_ns"])
    p.add_argument("--dry_run", action="store_true")

    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=384)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--bias", action="store_true")
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--max_iters", type=int, default=2000)
    p.add_argument("--checkpoint_interval", type=int, default=50)
    p.add_argument("--eval_interval", type=int, default=50)
    p.add_argument("--train_eval_iters", type=int, default=20)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--eval_seed", type=int, default=777_123)
    p.add_argument("--log_interval", type=int, default=10)

    p.add_argument("--adamw_lr", type=float, default=1e-3)
    p.add_argument("--adamw_weight_decay", type=float, default=0.1)
    p.add_argument("--adamw_beta2", type=float, default=0.99)
    p.add_argument("--muon_lr", type=float, default=0.01)
    p.add_argument("--muon_aux_lr", type=float, default=6e-4)
    p.add_argument("--muon_weight_decay", type=float, default=0.01)
    p.add_argument("--muon_beta2", type=float, default=0.95)
    p.add_argument("--muon_momentum", type=float, default=0.95)
    p.add_argument("--ns_steps", type=int, default=5)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--adam_eps", type=float, default=1e-8)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_iters", type=int, default=100)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--no_decay_lr", action="store_true")
    p.add_argument("--adamw_warmup_steps", type=int, default=100)
    p.add_argument("--early_stop_train_loss", type=float, default=None)
    p.add_argument("--early_stop_patience", type=int, default=1)

    p.add_argument("--match_train_losses", type=float, nargs="+", default=[1.2, 1.4, 1.6])

    p.add_argument("--hessian_layer_blocks", type=int, nargs="+", default=[0])
    p.add_argument("--hessian_layers", nargs="+", default=[])
    p.add_argument("--hessian_lanczos_steps", type=int, default=30)
    p.add_argument("--hessian_num_diag_batches", type=int, default=1)
    p.add_argument("--hessian_diag_batch_size", type=int, default=16)
    p.add_argument("--hessian_diag_eval_iters", type=int, default=10)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    args.base_out_dir = str(Path(args.base_out_dir).expanduser())
    Path(args.base_out_dir).mkdir(parents=True, exist_ok=True)
    configs = default_configs(args)
    with (Path(args.base_out_dir) / "experiment_config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    if args.stage in {"train", "all"}:
        train_configs(args, configs)
    if args.stage in {"eval", "all"}:
        evaluate_checkpoints(args, configs)
        plot_eval_history(args)
    if args.stage in {"match", "all"}:
        match_checkpoints(args)
        plot_matched_summary(args)
    if args.stage == "plots":
        plot_eval_history(args)
        plot_matched_summary(args)
    if args.stage in {"hessian", "all"}:
        run_hessian_diagnostics(args, configs)


if __name__ == "__main__":
    main()
