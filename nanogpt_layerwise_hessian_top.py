#!/usr/bin/env python3
"""
Layer-wise Hessian-top deconcentration diagnostic for tiny nanoGPT.

This script is intended to be run next to a nanoGPT checkout. It can:

1. Train a tiny Shakespeare-character nanoGPT with AdamW and save checkpoints at
   specified diagnostic iterations.
2. At each checkpoint, compute top algebraic eigenvectors of selected layer-wise
   mini-batch Hessian blocks using HVP + Lanczos.
3. Compare same-state update directions: GD, AdamW, AdamW with decoupled weight
   decay, MuonExact, and MuonNS5.

Example:

    python nanogpt_layerwise_hessian_top.py \
      --nanogpt_dir /path/to/nanoGPT \
      --mode train_diag \
      --out_dir runs_nanogpt_hessian/tiny_adamw \
      --seeds 0 1 2 \
      --device cuda
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import pickle
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


Tensor = torch.Tensor


DEFAULT_LAYERS = [
    "transformer.h.0.attn.c_attn.weight",
    "transformer.h.0.attn.c_proj.weight",
    "transformer.h.0.mlp.c_fc.weight",
    "transformer.h.0.mlp.c_proj.weight",
]


@dataclass(frozen=True)
class ModelArgs:
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    bias: bool
    vocab_size: int
    dropout: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def csv_write(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    xs = [float(v) for v in values if math.isfinite(float(v))]
    if not xs:
        return float("nan"), float("nan")
    mean = sum(xs) / len(xs)
    if len(xs) == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return mean, math.sqrt(max(var, 0.0))


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    group_fields = ["trajectory_optimizer", "eigenspace", "layer", "optimizer", "k"]
    metric_fields = [
        "rho",
        "C",
        "rho_over_gd",
        "q_eff",
        "q_eff_over_gd",
        "cos_grad_update",
        "descent_per_update_norm",
        "update_to_weight_ratio",
        "grad_norm",
        "update_norm",
    ]
    groups: Dict[Tuple[Any, ...], Dict[str, List[float]]] = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        rec = groups.setdefault(key, {metric: [] for metric in metric_fields})
        for metric in metric_fields:
            rec[metric].append(float(row[metric]))

    out: List[Dict[str, Any]] = []
    for key, rec in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        row = {field: value for field, value in zip(group_fields, key)}
        row["n"] = len(rec["rho"])
        for metric in metric_fields:
            mean, std = mean_std(rec[metric])
            row[f"mean_{metric}"] = mean
            row[f"std_{metric}"] = std
        out.append(row)
    return out


def summarize_rows_over_seeds(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    group_fields = ["trajectory_optimizer", "eigenspace", "layer", "optimizer", "k"]
    metric_fields = [
        "rho",
        "C",
        "rho_over_gd",
        "q_eff",
        "q_eff_over_gd",
        "cos_grad_update",
        "descent_per_update_norm",
        "update_to_weight_ratio",
        "grad_norm",
        "update_norm",
    ]

    per_seed: Dict[Tuple[Any, ...], Dict[str, List[float]]] = {}
    for row in rows:
        key = (
            row["seed"],
            row["trajectory_optimizer"],
            row["eigenspace"],
            row["layer"],
            row["optimizer"],
            row["k"],
        )
        rec = per_seed.setdefault(key, {metric: [] for metric in metric_fields})
        for metric in metric_fields:
            rec[metric].append(float(row[metric]))

    seed_means: Dict[Tuple[Any, ...], Dict[str, List[float]]] = {}
    for key, rec in per_seed.items():
        _, trajectory_optimizer, eigenspace, layer, optimizer, k = key
        out_key = (trajectory_optimizer, eigenspace, layer, optimizer, k)
        out_rec = seed_means.setdefault(out_key, {metric: [] for metric in metric_fields})
        for metric in metric_fields:
            mean, _ = mean_std(rec[metric])
            out_rec[metric].append(mean)

    out: List[Dict[str, Any]] = []
    for key, rec in sorted(seed_means.items(), key=lambda item: tuple(str(x) for x in item[0])):
        row = {field: value for field, value in zip(group_fields, key)}
        row["n_seeds"] = len(rec["rho"])
        for metric in metric_fields:
            mean, std = mean_std(rec[metric])
            row[f"mean_{metric}"] = mean
            row[f"std_{metric}"] = std
        out.append(row)
    return out


def parse_int_set(values: Sequence[int]) -> List[int]:
    out = sorted({int(v) for v in values})
    if not out:
        raise ValueError("expected at least one integer")
    return out


def stable_int_hash(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def load_nanogpt_model_module(nanogpt_dir: Path) -> Any:
    model_py = nanogpt_dir / "model.py"
    if not model_py.exists():
        raise FileNotFoundError(f"Could not find nanoGPT model.py at {model_py}")
    spec = importlib.util.spec_from_file_location("nanogpt_model", model_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import nanoGPT model.py from {model_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["nanogpt_model"] = module
    spec.loader.exec_module(module)
    return module


def read_vocab_size(data_dir: Path) -> int:
    meta_path = data_dir / "meta.pkl"
    if meta_path.exists():
        with meta_path.open("rb") as f:
            meta = pickle.load(f)
        if "vocab_size" in meta:
            return int(meta["vocab_size"])
    # Shakespeare char normally has meta.pkl. This fallback matches nanoGPT's
    # common GPT-2 default but should not be used for this experiment.
    return 50304


def load_memmap(data_dir: Path, split: str) -> np.memmap:
    path = data_dir / f"{split}.bin"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run nanoGPT's data/shakespeare_char/prepare.py first."
        )
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(
    data: np.memmap,
    *,
    batch_size: int,
    block_size: int,
    device: str,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor]:
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    starts = [int(i) for i in ix]
    x_np = np.stack([np.asarray(data[i : i + block_size], dtype=np.int64) for i in starts])
    y_np = np.stack(
        [np.asarray(data[i + 1 : i + 1 + block_size], dtype=np.int64) for i in starts]
    )
    x_cpu = torch.from_numpy(x_np)
    y_cpu = torch.from_numpy(y_np)
    if str(device).startswith("cuda"):
        x_cpu = x_cpu.pin_memory()
        y_cpu = y_cpu.pin_memory()
    x = x_cpu.to(device, non_blocking=str(device).startswith("cuda"))
    y = y_cpu.to(device, non_blocking=str(device).startswith("cuda"))
    return x, y


def get_lr(
    it: int,
    *,
    learning_rate: float,
    min_lr: float,
    warmup_iters: int,
    lr_decay_iters: int,
    decay_lr: bool,
) -> float:
    if not decay_lr:
        return float(learning_rate)
    if warmup_iters > 0 and it < warmup_iters:
        return float(learning_rate) * it / max(1, warmup_iters)
    if it > lr_decay_iters:
        return float(min_lr)
    decay_ratio = (it - warmup_iters) / max(1, lr_decay_iters - warmup_iters)
    decay_ratio = min(1.0, max(0.0, decay_ratio))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return float(min_lr) + coeff * (float(learning_rate) - float(min_lr))


def make_model(model_module: Any, model_args: ModelArgs, device: str) -> torch.nn.Module:
    cfg = model_module.GPTConfig(**asdict(model_args))
    model = model_module.GPT(cfg)
    model.to(device)
    return model


def force_manual_attention(model: torch.nn.Module, block_size: int, device: str) -> None:
    """
    nanoGPT uses torch scaled_dot_product_attention when available. Some PyTorch
    builds do not implement second derivatives through the efficient attention
    backward kernel, which breaks Hessian-vector products. Force nanoGPT's
    explicit attention path and add the causal mask buffer if the module was
    constructed on the flash-attention path.
    """
    for module in model.modules():
        if not hasattr(module, "flash"):
            continue
        module.flash = False
        if not hasattr(module, "bias"):
            mask = torch.tril(torch.ones(block_size, block_size, device=device)).view(
                1, 1, block_size, block_size
            )
            module.register_buffer("bias", mask)


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    train_data: np.memmap,
    val_data: np.memmap,
    *,
    batch_size: int,
    block_size: int,
    device: str,
    eval_iters: int,
    seed: int,
) -> Dict[str, Any]:
    model.eval()
    out: Dict[str, float] = {}
    for split, data in (("train", train_data), ("val", val_data)):
        losses = torch.empty(eval_iters, device=device)
        gen = torch.Generator()
        gen.manual_seed(seed + (0 if split == "train" else 10_000))
        for i in range(eval_iters):
            x, y = get_batch(
                data,
                batch_size=batch_size,
                block_size=block_size,
                device=device,
                generator=gen,
            )
            _, loss = model(x, y)
            losses[i] = loss.detach()
        out[split] = float(losses.mean().cpu())
    model.train()
    return out


class MuonMatrixOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float,
        momentum: float,
        weight_decay: float,
        ns_steps: int,
        ns_eps: float,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            ns_eps=ns_eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], Tensor]] = None) -> Optional[Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            momentum = float(group["momentum"])
            weight_decay = float(group["weight_decay"])
            ns_steps = int(group["ns_steps"])
            ns_eps = float(group["ns_eps"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError(f"MuonMatrixOptimizer expects 2D params, got {tuple(p.shape)}")
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)
                g = p.grad
                if momentum > 0.0:
                    state = self.state[p]
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = torch.zeros_like(g)
                    buf.mul_(momentum).add_(g, alpha=1.0 - momentum)
                    state["momentum_buffer"] = buf
                    g = buf
                update = newtonschulz5(g, steps=ns_steps, eps=ns_eps).to(dtype=p.dtype)
                p.add_(update, alpha=-lr)
        return loss


class HybridMuonAdamW:
    def __init__(
        self,
        *,
        adamw: Optional[torch.optim.Optimizer],
        muon: Optional[MuonMatrixOptimizer],
    ):
        self.adamw = adamw
        self.muon = muon

    @property
    def param_groups(self) -> List[Dict[str, Any]]:
        groups: List[Dict[str, Any]] = []
        if self.adamw is not None:
            groups.extend(self.adamw.param_groups)
        if self.muon is not None:
            groups.extend(self.muon.param_groups)
        return groups

    @property
    def state(self) -> Dict[torch.nn.Parameter, Dict[str, Any]]:
        state: Dict[torch.nn.Parameter, Dict[str, Any]] = {}
        if self.adamw is not None:
            state.update(self.adamw.state)
        if self.muon is not None:
            state.update(self.muon.state)
        return state

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self.adamw is not None:
            self.adamw.zero_grad(set_to_none=set_to_none)
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        if self.adamw is not None:
            self.adamw.step()
        if self.muon is not None:
            self.muon.step()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "kind": "hybrid_muon_adamw",
            "adamw": None if self.adamw is None else self.adamw.state_dict(),
            "muon": None if self.muon is None else self.muon.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if state_dict.get("kind") != "hybrid_muon_adamw":
            if self.adamw is None:
                raise ValueError("Cannot load plain optimizer state without AdamW sub-optimizer")
            self.adamw.load_state_dict(state_dict)
            return
        if self.adamw is not None and state_dict.get("adamw") is not None:
            self.adamw.load_state_dict(state_dict["adamw"])
        if self.muon is not None and state_dict.get("muon") is not None:
            self.muon.load_state_dict(state_dict["muon"])


def should_train_with_muon(name: str, param: torch.nn.Parameter, mode: str) -> bool:
    if mode == "adamw":
        return False
    if param.ndim != 2:
        return False
    if not name.startswith("transformer.h."):
        return False
    return name.endswith(".weight")


def configure_optimizer(
    model: torch.nn.Module,
    *,
    optimizer_name: str,
    weight_decay: float,
    learning_rate: float,
    beta1: float,
    beta2: float,
    device: str,
    aux_learning_rate: Optional[float] = None,
    muon_momentum: float = 0.95,
    ns_steps: int = 5,
    ns_eps: float = 1e-7,
) -> torch.optim.Optimizer:
    if optimizer_name not in {"adamw", "muon_ns"}:
        raise ValueError(f"unknown training optimizer: {optimizer_name}")
    if optimizer_name == "muon_ns":
        muon_params: List[torch.nn.Parameter] = []
        adamw_params: List[torch.nn.Parameter] = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if should_train_with_muon(name, param, optimizer_name):
                muon_params.append(param)
            else:
                adamw_params.append(param)
        adamw = None
        if adamw_params:
            adamw = torch.optim.AdamW(
                adamw_params,
                lr=float(learning_rate if aux_learning_rate is None else aux_learning_rate),
                betas=(beta1, beta2),
                weight_decay=weight_decay,
            )
            aux_lr = float(learning_rate if aux_learning_rate is None else aux_learning_rate)
            for group in adamw.param_groups:
                group["lr_scale"] = aux_lr / float(learning_rate)
                group["optimizer_role"] = "adamw_aux"
        muon = None
        if muon_params:
            muon = MuonMatrixOptimizer(
                muon_params,
                lr=learning_rate,
                momentum=muon_momentum,
                weight_decay=weight_decay,
                ns_steps=ns_steps,
                ns_eps=ns_eps,
            )
            for group in muon.param_groups:
                group["lr_scale"] = 1.0
                group["optimizer_role"] = "muon_matrix"
        print(
            f"using hybrid MuonNS/AdamW: {len(muon_params)} Muon tensors, "
            f"{len(adamw_params)} AdamW tensors",
            flush=True,
        )
        return HybridMuonAdamW(adamw=adamw, muon=muon)  # type: ignore[return-value]

    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    if hasattr(model, "configure_optimizers"):
        return model.configure_optimizers(
            weight_decay,
            learning_rate,
            (beta1, beta2),
            device_type,
        )
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
    )


def checkpoint_path(seed_dir: Path, iter_num: int) -> Path:
    return seed_dir / f"ckpt_iter_{iter_num}.pt"


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_args: ModelArgs,
    iter_num: int,
    best_val_loss: float,
    config: Dict[str, Any],
    elapsed_sec: Optional[float] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": asdict(model_args),
        "iter_num": int(iter_num),
        "best_val_loss": float(best_val_loss),
        "config": config,
    }
    if elapsed_sec is not None:
        checkpoint["elapsed_sec"] = float(elapsed_sec)
    torch.save(checkpoint, path)


def train_seed(args: argparse.Namespace, model_module: Any, seed: int) -> None:
    seed_everything(seed)
    data_dir = Path(args.nanogpt_dir) / "data" / args.dataset
    train_data = load_memmap(data_dir, "train")
    val_data = load_memmap(data_dir, "val")
    model_args = ModelArgs(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
        bias=args.bias,
        vocab_size=read_vocab_size(data_dir),
        dropout=args.dropout,
    )
    model = make_model(model_module, model_args, args.device)
    optimizer = configure_optimizer(
        model,
        optimizer_name=args.train_optimizer,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        device=args.device,
        aux_learning_rate=args.aux_learning_rate,
        muon_momentum=args.muon_momentum,
        ns_steps=args.ns_steps,
        ns_eps=args.ns_eps,
    )

    seed_dir = Path(args.out_dir) / f"seed{seed}"
    config = vars(args).copy()
    config["seed"] = seed
    config["model_args"] = asdict(model_args)
    save_json(seed_dir / "config.json", config)

    diag_iters = set(parse_int_set(args.diag_iters))
    best_val_loss = float("inf")
    early_stop_hits = 0
    train_gen = torch.Generator()
    train_gen.manual_seed(seed + 1234)
    start_time = time.time()

    if 0 in diag_iters:
        losses = estimate_loss(
            model,
            train_data,
            val_data,
            batch_size=args.batch_size,
            block_size=args.block_size,
            device=args.device,
            eval_iters=args.eval_iters,
            seed=seed + 20_000,
        )
        best_val_loss = min(best_val_loss, losses["val"])
        save_checkpoint(
            checkpoint_path(seed_dir, 0),
            model=model,
            optimizer=optimizer,
            model_args=model_args,
            iter_num=0,
            best_val_loss=best_val_loss,
            config=config,
            elapsed_sec=time.time() - start_time,
        )

    model.train()
    for iter_num in range(1, args.max_iters + 1):
        lr = get_lr(
            iter_num,
            learning_rate=args.learning_rate,
            min_lr=args.min_lr,
            warmup_iters=args.warmup_iters,
            lr_decay_iters=args.lr_decay_iters,
            decay_lr=not args.no_decay_lr,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * float(param_group.get("lr_scale", 1.0))

        x, y = get_batch(
            train_data,
            batch_size=args.batch_size,
            block_size=args.block_size,
            device=args.device,
            generator=train_gen,
        )
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if iter_num % args.log_interval == 0:
            elapsed = time.time() - start_time
            print(
                f"seed {seed} iter {iter_num}: "
                f"loss {float(loss.detach().cpu()):.4f}, lr {lr:.3e}, "
                f"elapsed {elapsed/60:.1f}m",
                flush=True,
            )

        did_eval = False
        early_stop_triggered = False
        if iter_num % args.eval_interval == 0 or iter_num in diag_iters:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                batch_size=args.batch_size,
                block_size=args.block_size,
                device=args.device,
                eval_iters=args.eval_iters,
                seed=seed + 20_000 + iter_num,
            )
            best_val_loss = min(best_val_loss, losses["val"])
            did_eval = True
            print(
                f"seed {seed} iter {iter_num}: "
                f"train {losses['train']:.4f}, val {losses['val']:.4f}",
                flush=True,
            )
            if args.early_stop_train_loss is not None:
                if losses["train"] <= float(args.early_stop_train_loss):
                    early_stop_hits += 1
                else:
                    early_stop_hits = 0
                early_stop_triggered = early_stop_hits >= int(args.early_stop_patience)

        if iter_num in diag_iters:
            save_checkpoint(
                checkpoint_path(seed_dir, iter_num),
                model=model,
                optimizer=optimizer,
                model_args=model_args,
                iter_num=iter_num,
                best_val_loss=best_val_loss,
                config=config,
                elapsed_sec=time.time() - start_time,
            )
        elif early_stop_triggered:
            save_checkpoint(
                checkpoint_path(seed_dir, iter_num),
                model=model,
                optimizer=optimizer,
                model_args=model_args,
                iter_num=iter_num,
                best_val_loss=best_val_loss,
                config=config,
                elapsed_sec=time.time() - start_time,
            )

        if did_eval and early_stop_triggered:
            print(
                f"seed {seed} early stop at iter {iter_num}: "
                f"train <= {float(args.early_stop_train_loss):.4f} "
                f"for {early_stop_hits} eval(s)",
                flush=True,
            )
            break


def polar_exact(g: Tensor) -> Tensor:
    assert g.ndim == 2
    u, _, vh = torch.linalg.svd(g.float(), full_matrices=False)
    return (u @ vh).to(device=g.device, dtype=g.dtype)


def newtonschulz5(g: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    assert g.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.bfloat16()
    x = x / (x.norm() + eps)

    transposed = False
    if g.size(0) > g.size(1):
        x = x.T
        transposed = True

    for _ in range(steps):
        aa = x @ x.T
        bb = b * aa + c * (aa @ aa)
        x = a * x + bb @ x

    if transposed:
        x = x.T
    return x.to(device=g.device, dtype=g.dtype)


def tensor_norm(x: Tensor) -> float:
    return float(torch.linalg.vector_norm(x.detach().double()).cpu())


def rho_k(d: Tensor, u_flat: Tensor) -> float:
    d_flat = d.reshape(-1).double()
    denom = torch.dot(d_flat, d_flat).clamp_min(1e-30)
    coeffs = u_flat.double().T @ d_flat
    return float((coeffs.square().sum() / denom).detach().cpu())


def concentration_c(d: Tensor, u_flat: Tensor) -> float:
    p = int(d.numel())
    k = int(u_flat.shape[1])
    return rho_k(d, u_flat) / (float(k) / float(p))


def scale_to_norm(direction: Tensor, target_norm: float) -> Tensor:
    norm = torch.linalg.vector_norm(direction).clamp_min(1e-30)
    return direction * (float(target_norm) / norm)


def unit_grad_direction(grad: Tensor) -> Tensor:
    return scale_to_norm(grad, 1.0)


def sign_grad_direction(grad: Tensor) -> Tensor:
    return torch.sign(grad)


def row_norm_direction(grad: Tensor) -> Tensor:
    return grad / torch.linalg.vector_norm(grad, dim=1, keepdim=True).clamp_min(1e-30)


def col_norm_direction(grad: Tensor) -> Tensor:
    return grad / torch.linalg.vector_norm(grad, dim=0, keepdim=True).clamp_min(1e-30)


def spectral_norm_direction(grad: Tensor) -> Tensor:
    return grad / torch.linalg.matrix_norm(grad.float(), ord=2).to(grad.dtype).clamp_min(1e-30)


def random_directions_like(
    grad: Tensor,
    *,
    seed: int,
    device: str,
) -> Dict[str, Tensor]:
    gen_device = "cuda" if str(device).startswith("cuda") else "cpu"
    gen = torch.Generator(device=gen_device)
    gen.manual_seed(seed)
    z = torch.randn(grad.shape, device=grad.device, dtype=grad.dtype, generator=gen)
    grad_norm = float(torch.linalg.vector_norm(grad).detach().cpu())
    return {
        "RandomGaussian": scale_to_norm(z, grad_norm),
        "RandomPolar": polar_exact(z),
    }


def extra_baseline_directions(
    grad: Tensor,
    *,
    seed: int,
    device: str,
) -> Dict[str, Tensor]:
    directions = {
        "UnitGrad": unit_grad_direction(grad),
        "SignGrad": sign_grad_direction(grad),
        "RowNormGrad": row_norm_direction(grad),
        "ColNormGrad": col_norm_direction(grad),
        "SpectralNormGrad": spectral_norm_direction(grad),
    }
    directions.update(random_directions_like(grad, seed=seed, device=device))
    return directions


def adamw_direction_from_state(
    *,
    param: Tensor,
    grad: Tensor,
    state: Dict[str, Any],
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
) -> Dict[str, Tensor]:
    m_prev = state.get("exp_avg")
    v_prev = state.get("exp_avg_sq")
    step_prev = state.get("step", 0)
    if torch.is_tensor(step_prev):
        step_prev_f = float(step_prev.detach().cpu().item())
    else:
        step_prev_f = float(step_prev)

    if m_prev is None or v_prev is None:
        m_prev = torch.zeros_like(grad)
        v_prev = torch.zeros_like(grad)
        step_prev_f = 0.0

    step = step_prev_f + 1.0
    m = float(beta1) * m_prev.to(dtype=grad.dtype, device=grad.device) + (
        1.0 - float(beta1)
    ) * grad
    v = float(beta2) * v_prev.to(dtype=grad.dtype, device=grad.device) + (
        1.0 - float(beta2)
    ) * grad.square()
    m_hat = m / (1.0 - float(beta1) ** step)
    v_hat = v / (1.0 - float(beta2) ** step)
    d_no_decay = m_hat / (torch.sqrt(v_hat) + float(eps))
    return {
        "AdamW-no-decay": d_no_decay,
        "AdamW-full": d_no_decay + float(weight_decay) * param.detach(),
    }


def selected_named_parameter(model: torch.nn.Module, name: str) -> torch.nn.Parameter:
    named = dict(model.named_parameters())
    if name not in named:
        available = "\n".join(sorted(named))
        raise KeyError(f"Unknown layer '{name}'. Available parameters:\n{available}")
    param = named[name]
    if param.ndim != 2:
        raise ValueError(f"Layer '{name}' is not a matrix parameter: {tuple(param.shape)}")
    return param


def gradient_for_layer(
    model: torch.nn.Module,
    param: torch.nn.Parameter,
    x: Tensor,
    y: Tensor,
) -> Tuple[Tensor, float]:
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        _, loss = model(x, y)
        grad = torch.autograd.grad(loss, param, create_graph=False, retain_graph=False)[0]
    return grad.detach(), float(loss.detach().cpu())


def build_adamw_warmup_state(
    model: torch.nn.Module,
    params: Sequence[torch.nn.Parameter],
    data: np.memmap,
    *,
    batch_size: int,
    block_size: int,
    device: str,
    steps: int,
    beta1: float,
    beta2: float,
    seed: int,
) -> Dict[torch.nn.Parameter, Dict[str, Any]]:
    states: Dict[torch.nn.Parameter, Dict[str, Any]] = {
        p: {"exp_avg": torch.zeros_like(p), "exp_avg_sq": torch.zeros_like(p), "step": torch.tensor(0.0)}
        for p in params
    }
    if steps <= 0:
        return states

    gen = torch.Generator()
    gen.manual_seed(seed)
    model.eval()
    for t in range(1, steps + 1):
        x, y = get_batch(
            data,
            batch_size=batch_size,
            block_size=block_size,
            device=device,
            generator=gen,
        )
        model.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        with torch.no_grad():
            for p in params:
                if p.grad is None:
                    continue
                st = states[p]
                g = p.grad.detach()
                st["exp_avg"].mul_(float(beta1)).add_(g, alpha=1.0 - float(beta1))
                st["exp_avg_sq"].mul_(float(beta2)).addcmul_(g, g, value=1.0 - float(beta2))
                st["step"] = torch.tensor(float(t), device=p.device)
    model.zero_grad(set_to_none=True)
    return states


def make_block_hvp(
    model: torch.nn.Module,
    param: torch.nn.Parameter,
    x: Tensor,
    y: Tensor,
) -> Callable[[Tensor], Tensor]:
    def hvp(v: Tensor) -> Tensor:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            _, loss = model(x, y)
            grad = torch.autograd.grad(
                loss,
                param,
                create_graph=True,
                retain_graph=True,
            )[0]
            gv = (grad * v.to(dtype=grad.dtype, device=grad.device)).sum()
            hv = torch.autograd.grad(gv, param, retain_graph=False)[0]
        return hv.detach()

    return hvp


def lanczos_topk(
    hvp_fn: Callable[[Tensor], Tensor],
    shape: torch.Size,
    *,
    k: int,
    steps: int,
    device: str,
    dtype: torch.dtype,
    seed: int,
    which: str = "top",
    tol: float = 1e-8,
) -> Tuple[Tensor, Tensor, int]:
    if steps < k:
        raise ValueError(f"lanczos steps ({steps}) must be >= k ({k})")
    if which not in {"top", "bottom"}:
        raise ValueError("which must be 'top' or 'bottom'")

    gen_device = "cuda" if str(device).startswith("cuda") else "cpu"
    gen = torch.Generator(device=gen_device)
    gen.manual_seed(seed)
    q = torch.randn(shape, device=device, dtype=dtype, generator=gen)
    q = q / torch.linalg.vector_norm(q).clamp_min(1e-30)
    q_prev = torch.zeros_like(q)
    beta = torch.zeros((), device=device, dtype=dtype)

    q_basis: List[Tensor] = []
    alphas: List[Tensor] = []
    betas: List[Tensor] = []

    for j in range(steps):
        z = hvp_fn(q).to(dtype=dtype)
        if j > 0:
            z = z - beta * q_prev
        alpha = (q * z).sum()
        z = z - alpha * q

        # Full reorthogonalization is cheap at these dimensions and avoids
        # duplicated Ritz vectors when Hessian eigenvalues are clustered.
        for q_old in q_basis:
            z = z - (z * q_old).sum() * q_old

        beta_next = torch.linalg.vector_norm(z)
        q_basis.append(q.detach())
        alphas.append(alpha.detach())
        if j < steps - 1:
            betas.append(beta_next.detach())
        if float(beta_next.detach().cpu()) < tol:
            break
        q_prev = q
        q = z / beta_next
        beta = beta_next

    m_eff = len(q_basis)
    k_eff = min(k, m_eff)
    q_flat = torch.stack([qq.reshape(-1) for qq in q_basis], dim=1)
    tri = torch.zeros(m_eff, m_eff, device=device, dtype=torch.float64)
    for i, alpha in enumerate(alphas):
        tri[i, i] = alpha.double()
    for i, beta_i in enumerate(betas[: max(0, m_eff - 1)]):
        tri[i, i + 1] = beta_i.double()
        tri[i + 1, i] = beta_i.double()

    evals, evecs = torch.linalg.eigh(tri)
    idx = torch.argsort(evals, descending=(which == "top"))[:k_eff]
    extreme_evals = evals[idx]
    extreme_vecs = q_flat.double() @ evecs[:, idx]
    extreme_vecs = extreme_vecs / torch.linalg.vector_norm(extreme_vecs, dim=0, keepdim=True).clamp_min(
        1e-30
    )
    return extreme_evals.detach(), extreme_vecs.detach(), m_eff


def lanczos_quality(
    hvp_fn: Callable[[Tensor], Tensor],
    evals: Tensor,
    vecs_flat: Tensor,
    shape: torch.Size,
    dtype: torch.dtype,
    m_eff: int,
) -> Dict[str, Any]:
    if vecs_flat.numel() == 0:
        return {
            "lanczos_m_eff": int(m_eff),
            "lanczos_resid_mean": float("nan"),
            "lanczos_resid_max": float("nan"),
            "lanczos_rel_resid_mean": float("nan"),
            "lanczos_rel_resid_max": float("nan"),
            "lanczos_orth_error": float("nan"),
            "lanczos_eigvals": "[]",
            "lanczos_residuals": "[]",
            "lanczos_rel_residuals": "[]",
        }

    residuals: List[float] = []
    rel_residuals: List[float] = []
    for i in range(vecs_flat.shape[1]):
        v = vecs_flat[:, i].reshape(shape).to(dtype=dtype)
        hv = hvp_fn(v).reshape(-1).double()
        v_flat = vecs_flat[:, i].double()
        lam = evals[i].double()
        resid = torch.linalg.vector_norm(hv - lam * v_flat)
        denom = torch.maximum(
            torch.linalg.vector_norm(hv),
            torch.abs(lam) * torch.linalg.vector_norm(v_flat),
        ).clamp_min(1e-30)
        residuals.append(float(resid.detach().cpu()))
        rel_residuals.append(float((resid / denom).detach().cpu()))

    gram = vecs_flat.double().T @ vecs_flat.double()
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    orth_error = torch.linalg.matrix_norm(gram - eye, ord="fro")
    resid_mean, _ = mean_std(residuals)
    rel_resid_mean, _ = mean_std(rel_residuals)
    return {
        "lanczos_m_eff": int(m_eff),
        "lanczos_resid_mean": resid_mean,
        "lanczos_resid_max": max(residuals),
        "lanczos_rel_resid_mean": rel_resid_mean,
        "lanczos_rel_resid_max": max(rel_residuals),
        "lanczos_orth_error": float(orth_error.detach().cpu()),
        "lanczos_eigvals": json.dumps([float(x) for x in evals.detach().cpu().tolist()]),
        "lanczos_residuals": json.dumps(residuals),
        "lanczos_rel_residuals": json.dumps(rel_residuals),
    }


def q_eff(d: Tensor, hvp_fn: Callable[[Tensor], Tensor]) -> float:
    hd = hvp_fn(d).reshape(-1).double()
    d_flat = d.reshape(-1).double()
    denom = torch.dot(d_flat, d_flat).clamp_min(1e-30)
    return float((torch.dot(d_flat, hd) / denom).detach().cpu())


def hvp_quadratic_form(d: Tensor, hvp_fn: Callable[[Tensor], Tensor]) -> float:
    hd = hvp_fn(d).reshape(-1).double()
    d_flat = d.reshape(-1).double()
    return float(torch.dot(d_flat, hd).detach().cpu())


def safe_eta_name(eta: float) -> str:
    return f"{float(eta):.0e}".replace("-", "m").replace("+", "").replace(".", "p")


def current_lr_for_checkpoint(args: argparse.Namespace, checkpoint: Dict[str, Any], iter_num: int) -> float:
    cfg = checkpoint.get("config", {})
    return get_lr(
        int(iter_num),
        learning_rate=float(cfg.get("learning_rate", args.learning_rate)),
        min_lr=float(cfg.get("min_lr", args.min_lr)),
        warmup_iters=int(cfg.get("warmup_iters", args.warmup_iters)),
        lr_decay_iters=int(cfg.get("lr_decay_iters", args.lr_decay_iters)),
        decay_lr=not bool(cfg.get("no_decay_lr", args.no_decay_lr)),
    )


def direction_alignment_metrics(
    *,
    grad: Tensor,
    direction: Tensor,
    dHd: float,
    weight: Tensor,
    current_lr: float,
    quad_etas: Sequence[float],
) -> Dict[str, float]:
    g_flat = grad.reshape(-1).double()
    d_flat = direction.reshape(-1).double()
    grad_norm = torch.linalg.vector_norm(g_flat).clamp_min(1e-30)
    update_norm = torch.linalg.vector_norm(d_flat).clamp_min(1e-30)
    grad_dot_update = torch.dot(g_flat, d_flat)
    weight_norm = torch.linalg.vector_norm(weight.detach().reshape(-1).double()).clamp_min(1e-30)
    metrics: Dict[str, float] = {
        "grad_dot_update": float(grad_dot_update.detach().cpu()),
        "cos_grad_update": float((grad_dot_update / (grad_norm * update_norm)).detach().cpu()),
        "descent_per_update_norm": float((grad_dot_update / update_norm).detach().cpu()),
        "descent_per_grad_norm": float((grad_dot_update / grad_norm).detach().cpu()),
        "dHd": float(dHd),
        "weight_norm": float(weight_norm.detach().cpu()),
        "update_rms": float((update_norm / math.sqrt(max(1, direction.numel()))).detach().cpu()),
        "current_lr": float(current_lr),
        "update_to_weight_ratio": float((float(current_lr) * update_norm / weight_norm).detach().cpu()),
    }
    for eta in quad_etas:
        quad_delta = -float(eta) * metrics["grad_dot_update"] + 0.5 * (float(eta) ** 2) * float(dHd)
        metrics[f"quad_delta_eta_{safe_eta_name(float(eta))}"] = quad_delta
    return metrics


@torch.no_grad()
def loss_on_batch(model: torch.nn.Module, x: Tensor, y: Tensor) -> float:
    was_training = model.training
    model.eval()
    _, loss = model(x, y)
    if was_training:
        model.train()
    return float(loss.detach().cpu())


@torch.no_grad()
def line_search_rows_for_directions(
    *,
    model: torch.nn.Module,
    param: torch.nn.Parameter,
    directions: Dict[str, Tensor],
    x: Tensor,
    y: Tensor,
    etas: Sequence[float],
    metadata: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not etas:
        return []
    base = loss_on_batch(model, x, y)
    original = param.detach().clone()
    rows: List[Dict[str, Any]] = []
    for opt_name, direction in directions.items():
        d = direction.detach().to(device=param.device, dtype=param.dtype)
        for eta in etas:
            param.copy_(original)
            param.add_(d, alpha=-float(eta))
            stepped_loss = loss_on_batch(model, x, y)
            row = dict(metadata)
            row.update(
                {
                    "optimizer": opt_name,
                    "eta": float(eta),
                    "base_loss": base,
                    "stepped_loss": stepped_loss,
                    "loss_delta": stepped_loss - base,
                    "update_norm": tensor_norm(d),
                    "update_rms": float(tensor_norm(d) / math.sqrt(max(1, d.numel()))),
                }
            )
            rows.append(row)
    param.copy_(original)
    return rows


@torch.no_grad()
def load_model_and_optimizer(
    *,
    model_module: Any,
    ckpt_path: Path,
    device: str,
    args: argparse.Namespace,
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, Dict[str, Any]]:
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = ModelArgs(**checkpoint["model_args"])
    model = make_model(model_module, model_args, device)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    if any(k.startswith(unwanted_prefix) for k in state_dict):
        state_dict = {
            k[len(unwanted_prefix) :] if k.startswith(unwanted_prefix) else k: v
            for k, v in state_dict.items()
        }
    model.load_state_dict(state_dict)
    force_manual_attention(model, model_args.block_size, device)
    ckpt_config = checkpoint.get("config", {})
    saved_opt = checkpoint.get("optimizer", {})
    if "train_optimizer" in ckpt_config:
        optimizer_name = ckpt_config["train_optimizer"]
    elif isinstance(saved_opt, dict) and saved_opt.get("kind") == "hybrid_muon_adamw":
        optimizer_name = "muon_ns"
    else:
        optimizer_name = "adamw"
    optimizer = configure_optimizer(
        model,
        optimizer_name=optimizer_name,
        weight_decay=float(ckpt_config.get("weight_decay", args.weight_decay)),
        learning_rate=float(ckpt_config.get("learning_rate", args.learning_rate)),
        beta1=float(ckpt_config.get("beta1", args.beta1)),
        beta2=float(ckpt_config.get("beta2", args.beta2)),
        device=device,
        aux_learning_rate=ckpt_config.get("aux_learning_rate", args.aux_learning_rate),
        muon_momentum=float(ckpt_config.get("muon_momentum", args.muon_momentum)),
        ns_steps=int(ckpt_config.get("ns_steps", args.ns_steps)),
        ns_eps=float(ckpt_config.get("ns_eps", args.ns_eps)),
    )
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    model.eval()
    return model, optimizer, checkpoint


def diagnostic_for_checkpoint(
    args: argparse.Namespace,
    *,
    model_module: Any,
    ckpt_path: Path,
    seed: int,
    train_data: np.memmap,
    val_data: np.memmap,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    model, optimizer, checkpoint = load_model_and_optimizer(
        model_module=model_module,
        ckpt_path=ckpt_path,
        device=args.device,
        args=args,
    )
    iter_num = int(checkpoint["iter_num"])
    model_args = ModelArgs(**checkpoint["model_args"])

    diag_data = train_data if args.diag_split == "train" else val_data
    losses = estimate_loss(
        model,
        train_data,
        val_data,
        batch_size=args.batch_size,
        block_size=model_args.block_size,
        device=args.device,
        eval_iters=args.diag_eval_iters,
        seed=args.diag_batch_seed + seed + iter_num,
    )
    model.eval()

    max_k = max(args.k_values)
    rows: List[Dict[str, Any]] = []
    line_rows: List[Dict[str, Any]] = []
    current_lr = current_lr_for_checkpoint(args, checkpoint, iter_num)
    trajectory_optimizer = checkpoint.get("config", {}).get("train_optimizer", args.train_optimizer)
    if args.adamw_state_mode == "checkpoint":
        opt_state = optimizer.state
    elif args.adamw_state_mode == "fresh":
        opt_state = {}
    elif args.adamw_state_mode == "warmup":
        warmup_params = [selected_named_parameter(model, layer_name) for layer_name in args.layers]
        opt_state = build_adamw_warmup_state(
            model,
            warmup_params,
            train_data,
            batch_size=args.diag_batch_size,
            block_size=model_args.block_size,
            device=args.device,
            steps=args.adamw_warmup_steps,
            beta1=args.beta1,
            beta2=args.beta2,
            seed=args.diag_batch_seed + seed * 1_000_003 + iter_num * 9176 + 55_555,
        )
    else:
        raise ValueError(f"unknown adamw_state_mode: {args.adamw_state_mode}")

    for diag_batch_idx in range(args.num_diag_batches):
        hess_gen = torch.Generator()
        hess_gen.manual_seed(
            args.diag_batch_seed
            + seed * 1_000_003
            + iter_num * 9176
            + diag_batch_idx * 101
        )
        x_hess, y_hess = get_batch(
            diag_data,
            batch_size=args.diag_batch_size,
            block_size=model_args.block_size,
            device=args.device,
            generator=hess_gen,
        )
        if args.cross_batch_hessian:
            grad_gen = torch.Generator()
            grad_gen.manual_seed(
                args.diag_batch_seed
                + seed * 1_000_003
                + iter_num * 9176
                + diag_batch_idx * 101
                + 9_999_991
            )
            x_grad, y_grad = get_batch(
                diag_data,
                batch_size=args.diag_batch_size,
                block_size=model_args.block_size,
                device=args.device,
                generator=grad_gen,
            )
        else:
            x_grad, y_grad = x_hess, y_hess

        for layer_name in args.layers:
            param = selected_named_parameter(model, layer_name)
            grad, diag_loss = gradient_for_layer(model, param, x_grad, y_grad)
            _, hess_loss = gradient_for_layer(model, param, x_hess, y_hess)
            hvp_fn = make_block_hvp(model, param, x_hess, y_hess)
            lanczos_seed = (
                args.lanczos_seed
                + seed * 1_000_003
                + iter_num * 9176
                + diag_batch_idx * 101
                + stable_int_hash(layer_name) % 100_000
            )
            eigenspaces: List[Tuple[str, Tensor, Tensor, Dict[str, Any]]] = []
            top_evals, top_vecs, top_m_eff = lanczos_topk(
                hvp_fn,
                param.shape,
                k=max_k,
                steps=args.lanczos_steps,
                device=args.device,
                dtype=param.dtype,
                seed=lanczos_seed,
                which="top",
            )
            eigenspaces.append(
                (
                    "top",
                    top_evals,
                    top_vecs,
                    lanczos_quality(
                        hvp_fn,
                        top_evals,
                        top_vecs,
                        param.shape,
                        param.dtype,
                        top_m_eff,
                    ),
                )
            )
            if args.include_bottom_eigenspace:
                bottom_evals, bottom_vecs, bottom_m_eff = lanczos_topk(
                    hvp_fn,
                    param.shape,
                    k=max_k,
                    steps=args.lanczos_steps,
                    device=args.device,
                    dtype=param.dtype,
                    seed=lanczos_seed + 17_171,
                    which="bottom",
                )
                eigenspaces.append(
                    (
                        "bottom",
                        bottom_evals,
                        bottom_vecs,
                        lanczos_quality(
                            hvp_fn,
                            bottom_evals,
                            bottom_vecs,
                            param.shape,
                            param.dtype,
                            bottom_m_eff,
                        ),
                    )
                )

            directions: Dict[str, Tensor] = {
                "GD": grad,
                "MuonExact": polar_exact(grad),
                f"MuonNS{args.ns_steps}": newtonschulz5(
                    grad,
                    steps=args.ns_steps,
                    eps=args.ns_eps,
                ),
            }
            directions.update(
                adamw_direction_from_state(
                    param=param,
                    grad=grad,
                    state=opt_state.get(param, {}),
                    beta1=args.beta1,
                    beta2=args.beta2,
                    eps=args.adam_eps,
                    weight_decay=args.weight_decay,
                )
            )
            if args.include_extra_baselines:
                baseline_seed = (
                    args.random_baseline_seed
                    + seed * 1_000_003
                    + iter_num * 9176
                    + diag_batch_idx * 101
                    + stable_int_hash(layer_name) % 100_000
                )
                directions.update(
                    extra_baseline_directions(
                        grad,
                        seed=baseline_seed,
                        device=args.device,
                    )
                )

            q_cache: Dict[str, float] = {}
            dHd_cache: Dict[str, float] = {}
            alignment_cache: Dict[str, Dict[str, float]] = {}
            for opt_name, direction in directions.items():
                dHd = hvp_quadratic_form(direction, hvp_fn)
                dHd_cache[opt_name] = dHd
                denom = max(float(tensor_norm(direction)) ** 2, 1e-30)
                q_cache[opt_name] = dHd / denom
                alignment_cache[opt_name] = direction_alignment_metrics(
                    grad=grad,
                    direction=direction,
                    dHd=dHd,
                    weight=param,
                    current_lr=current_lr,
                    quad_etas=args.quad_etas,
                )

            if args.line_search_etas:
                line_rows.extend(
                    line_search_rows_for_directions(
                        model=model,
                        param=param,
                        directions=directions,
                        x=x_grad,
                        y=y_grad,
                        etas=args.line_search_etas,
                        metadata={
                            "seed": seed,
                            "trajectory_optimizer": trajectory_optimizer,
                            "iter": iter_num,
                            "diag_batch_idx": diag_batch_idx,
                            "cross_batch_hessian": bool(args.cross_batch_hessian),
                            "layer": layer_name,
                            "train_loss_diag_batch": diag_loss,
                            "val_loss": losses["val"],
                            "current_lr": current_lr,
                        },
                    )
                )

            for eigenspace_name, evals, vecs, quality in eigenspaces:
                rho_gd_by_k: Dict[int, float] = {}
                for k in args.k_values:
                    u_k = vecs[:, : min(k, vecs.shape[1])]
                    rho_gd_by_k[int(k)] = rho_k(grad, u_k)

                for opt_name, direction in directions.items():
                    for k in args.k_values:
                        k_eff = min(int(k), int(vecs.shape[1]))
                        u_k = vecs[:, :k_eff]
                        rho = rho_k(direction, u_k)
                        c_val = concentration_c(direction, u_k)
                        rho_gd = max(rho_gd_by_k[int(k)], 1e-30)
                        row = {
                            "seed": seed,
                            "trajectory_optimizer": trajectory_optimizer,
                            "iter": iter_num,
                            "diag_batch_idx": diag_batch_idx,
                            "cross_batch_hessian": bool(args.cross_batch_hessian),
                            "eigenspace": eigenspace_name,
                            "layer": layer_name,
                            "optimizer": opt_name,
                            "k": int(k),
                            "rho": rho,
                            "C": c_val,
                            "rho_over_gd": rho / rho_gd,
                            "q_eff": q_cache[opt_name],
                            "q_eff_over_gd": "",
                            "top_eig_1": float(evals[0].detach().cpu())
                            if len(evals) > 0
                            else float("nan"),
                            "top_eig_k": float(evals[k_eff - 1].detach().cpu())
                            if len(evals) >= k_eff
                            else float("nan"),
                            "grad_norm": tensor_norm(grad),
                            "update_norm": tensor_norm(direction),
                            "train_loss_diag_batch": diag_loss,
                            "train_loss_hessian_batch": hess_loss,
                            "train_loss_eval": losses["train"],
                            "val_loss": losses["val"],
                            "param_numel": int(param.numel()),
                            "param_shape": "x".join(str(v) for v in param.shape),
                            "lanczos_steps": args.lanczos_steps,
                            "diag_split": args.diag_split,
                        }
                        if "GD" in q_cache and abs(q_cache["GD"]) > 1e-30:
                            row["q_eff_over_gd"] = q_cache[opt_name] / q_cache["GD"]
                        row.update(alignment_cache[opt_name])
                        row.update(quality)
                        rows.append(row)
            print(
                f"diagnosed seed {seed} iter {iter_num} batch {diag_batch_idx} "
                f"layer {layer_name}: top_eig_1={float(top_evals[0].cpu()):.4e}, "
                f"top_rel_resid_max={eigenspaces[0][3]['lanczos_rel_resid_max']:.2e}, "
                f"diag_loss={diag_loss:.4f}",
                flush=True,
            )

    return rows, line_rows


def run_diagnostics(args: argparse.Namespace, model_module: Any) -> None:
    all_rows: List[Dict[str, Any]] = []
    all_line_rows: List[Dict[str, Any]] = []
    data_dir = Path(args.nanogpt_dir) / "data" / args.dataset
    train_data = load_memmap(data_dir, "train")
    val_data = load_memmap(data_dir, "val")
    diag_iters = parse_int_set(args.diag_iters)

    for seed in args.seeds:
        seed_dir = Path(args.out_dir) / f"seed{seed}"
        for iter_num in diag_iters:
            ckpt = checkpoint_path(seed_dir, iter_num)
            if not ckpt.exists():
                raise FileNotFoundError(f"Missing checkpoint {ckpt}")
            metric_rows, line_rows = diagnostic_for_checkpoint(
                args,
                model_module=model_module,
                ckpt_path=ckpt,
                seed=int(seed),
                train_data=train_data,
                val_data=val_data,
            )
            all_rows.extend(metric_rows)
            all_line_rows.extend(line_rows)

    fields = [
        "seed",
        "trajectory_optimizer",
        "iter",
        "diag_batch_idx",
        "cross_batch_hessian",
        "eigenspace",
        "layer",
        "optimizer",
        "k",
        "rho",
        "C",
        "rho_over_gd",
        "q_eff",
        "q_eff_over_gd",
        "top_eig_1",
        "top_eig_k",
        "grad_norm",
        "update_norm",
        "grad_dot_update",
        "cos_grad_update",
        "descent_per_update_norm",
        "descent_per_grad_norm",
        "dHd",
        "weight_norm",
        "update_rms",
        "current_lr",
        "update_to_weight_ratio",
        "train_loss_diag_batch",
        "train_loss_hessian_batch",
        "train_loss_eval",
        "val_loss",
        "param_numel",
        "param_shape",
        "lanczos_steps",
        "lanczos_m_eff",
        "lanczos_resid_mean",
        "lanczos_resid_max",
        "lanczos_rel_resid_mean",
        "lanczos_rel_resid_max",
        "lanczos_orth_error",
        "lanczos_eigvals",
        "lanczos_residuals",
        "lanczos_rel_residuals",
        "diag_split",
    ]
    for eta in args.quad_etas:
        fields.append(f"quad_delta_eta_{safe_eta_name(float(eta))}")
    csv_write(Path(args.out_dir) / "layerwise_hessian_top_metrics.csv", all_rows, fields)
    if all_line_rows:
        line_fields = [
            "seed",
            "trajectory_optimizer",
            "iter",
            "diag_batch_idx",
            "cross_batch_hessian",
            "layer",
            "optimizer",
            "eta",
            "base_loss",
            "stepped_loss",
            "loss_delta",
            "update_norm",
            "update_rms",
            "train_loss_diag_batch",
            "val_loss",
            "current_lr",
        ]
        csv_write(
            Path(args.out_dir) / "layerwise_hessian_top_line_search.csv",
            all_line_rows,
            line_fields,
        )
    summary_fields = [
        "trajectory_optimizer",
        "eigenspace",
        "layer",
        "optimizer",
        "k",
        "n",
        "mean_rho",
        "std_rho",
        "mean_C",
        "std_C",
        "mean_rho_over_gd",
        "std_rho_over_gd",
        "mean_q_eff",
        "std_q_eff",
        "mean_q_eff_over_gd",
        "std_q_eff_over_gd",
        "mean_cos_grad_update",
        "std_cos_grad_update",
        "mean_descent_per_update_norm",
        "std_descent_per_update_norm",
        "mean_update_to_weight_ratio",
        "std_update_to_weight_ratio",
        "mean_grad_norm",
        "std_grad_norm",
        "mean_update_norm",
        "std_update_norm",
    ]
    csv_write(
        Path(args.out_dir) / "layerwise_hessian_top_summary.csv",
        summarize_rows(all_rows),
        summary_fields,
    )
    seed_summary_fields = [
        "trajectory_optimizer",
        "eigenspace",
        "layer",
        "optimizer",
        "k",
        "n_seeds",
        "mean_rho",
        "std_rho",
        "mean_C",
        "std_C",
        "mean_rho_over_gd",
        "std_rho_over_gd",
        "mean_q_eff",
        "std_q_eff",
        "mean_q_eff_over_gd",
        "std_q_eff_over_gd",
        "mean_cos_grad_update",
        "std_cos_grad_update",
        "mean_descent_per_update_norm",
        "std_descent_per_update_norm",
        "mean_update_to_weight_ratio",
        "std_update_to_weight_ratio",
        "mean_grad_norm",
        "std_grad_norm",
        "mean_update_norm",
        "std_update_norm",
    ]
    csv_write(
        Path(args.out_dir) / "layerwise_hessian_top_seed_summary.csv",
        summarize_rows_over_seeds(all_rows),
        seed_summary_fields,
    )
    save_json(Path(args.out_dir) / "diagnostic_config.json", vars(args))
    print(f"wrote {len(all_rows)} rows to {Path(args.out_dir) / 'layerwise_hessian_top_metrics.csv'}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nanogpt_dir", type=str, default=".", help="Path to nanoGPT root")
    p.add_argument("--dataset", type=str, default="shakespeare_char")
    p.add_argument("--mode", choices=["train", "diag", "train_diag"], default="train_diag")
    p.add_argument("--out_dir", type=str, default="runs_nanogpt_hessian/tiny_adamw")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--diag_iters", type=int, nargs="+", default=[0, 1000, 2000])

    # Tiny nanoGPT defaults.
    p.add_argument("--n_layer", type=int, default=2)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--bias", action="store_true", help="Use bias terms in Linear/LayerNorm")
    p.add_argument("--dropout", type=float, default=0.0)

    # Training.
    p.add_argument("--train_optimizer", choices=["adamw", "muon_ns"], default="adamw")
    p.add_argument("--max_iters", type=int, default=2000)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--aux_learning_rate", type=float, default=None)
    p.add_argument("--min_lr", type=float, default=1e-4)
    p.add_argument("--lr_decay_iters", type=int, default=2000)
    p.add_argument("--warmup_iters", type=int, default=100)
    p.add_argument("--no_decay_lr", action="store_true")
    p.add_argument("--weight_decay", type=float, default=1e-1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--adam_eps", type=float, default=1e-8)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--muon_momentum", type=float, default=0.95)
    p.add_argument(
        "--early_stop_train_loss",
        type=float,
        default=None,
        help="Stop a seed after eval train loss is at or below this fixed threshold.",
    )
    p.add_argument("--early_stop_patience", type=int, default=1)

    # Diagnostics.
    p.add_argument("--layers", type=str, nargs="+", default=DEFAULT_LAYERS)
    p.add_argument("--k_values", type=int, nargs="+", default=[1, 5])
    p.add_argument("--lanczos_steps", type=int, default=30)
    p.add_argument("--lanczos_seed", type=int, default=91_337)
    p.add_argument("--diag_batch_size", type=int, default=64)
    p.add_argument("--diag_batch_seed", type=int, default=123_456)
    p.add_argument("--num_diag_batches", type=int, default=1)
    p.add_argument(
        "--cross_batch_hessian",
        action="store_true",
        help="Use one diagnostic batch for Hessian eigenspace and another for gradient/update.",
    )
    p.add_argument("--diag_split", choices=["train", "val"], default="train")
    p.add_argument("--diag_eval_iters", type=int, default=20)
    p.add_argument(
        "--adamw_state_mode",
        choices=["checkpoint", "fresh", "warmup"],
        default="checkpoint",
        help="How to obtain AdamW moments for same-state AdamW directions.",
    )
    p.add_argument("--adamw_warmup_steps", type=int, default=100)
    p.add_argument("--ns_steps", type=int, default=5)
    p.add_argument("--ns_eps", type=float, default=1e-7)
    p.add_argument(
        "--quad_etas",
        type=float,
        nargs="+",
        default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
        help="Etas for quadratic predicted loss change columns.",
    )
    p.add_argument(
        "--line_search_etas",
        type=float,
        nargs="*",
        default=[],
        help="If provided, evaluate virtual one-layer steps on the diagnostic gradient batch.",
    )
    p.add_argument(
        "--include_extra_baselines",
        action="store_true",
        help="Log UnitGrad, SignGrad, row/column/spectral normalization, and random baselines.",
    )
    p.add_argument("--random_baseline_seed", type=int, default=777_777)
    p.add_argument("--include_bottom_eigenspace", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    args.nanogpt_dir = str(Path(args.nanogpt_dir).resolve())
    args.out_dir = str(Path(args.out_dir).resolve())
    args.diag_iters = parse_int_set(args.diag_iters)
    args.k_values = parse_int_set(args.k_values)
    args.layers = list(args.layers)
    if max(args.k_values) > args.lanczos_steps:
        raise ValueError("--lanczos_steps must be at least max(--k_values)")
    if args.num_diag_batches < 1:
        raise ValueError("--num_diag_batches must be >= 1")
    if args.adamw_warmup_steps < 0:
        raise ValueError("--adamw_warmup_steps must be >= 0")
    if any(float(eta) <= 0 for eta in args.quad_etas):
        raise ValueError("--quad_etas must be positive")
    if any(float(eta) <= 0 for eta in args.line_search_etas):
        raise ValueError("--line_search_etas must be positive")

    model_module = load_nanogpt_model_module(Path(args.nanogpt_dir))
    if args.mode in ("train", "train_diag"):
        for seed in args.seeds:
            train_seed(args, model_module, int(seed))
    if args.mode in ("diag", "train_diag"):
        run_diagnostics(args, model_module)


if __name__ == "__main__":
    main()
