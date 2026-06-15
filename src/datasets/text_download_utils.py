import logging
import lzma
import re
import tarfile
import urllib.request
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.utils.io_utils import write_json

logger = logging.getLogger(__name__)

DEFAULT_TOKENIZER = "gpt2"
OPENWEBTEXT_10K_URL = (
    "https://cdn-datasets.huggingface.co/nlp/datasets/openwebtext/openwebtext-10k.tar.xz"
)


def _download_file(url: str, dest: Path) -> None:
    logger.info(f"Downloading {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def _extract_xz_archive(xz_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with lzma.open(xz_path, "rb") as compressed:
        with tarfile.open(fileobj=compressed, mode="r:") as tar:
            tar.extractall(path=dest_dir)


def load_openwebtext_10k_texts(data_dir: Path) -> list[str]:
    """
    Download and extract stas/openwebtext-10k without HuggingFace dataset scripts.

    Mirrors the original loading script:
    https://huggingface.co/datasets/stas/openwebtext-10k
    """
    marker = data_dir / ".raw_ready"
    raw_root = data_dir / "raw"
    if marker.exists():
        txt_files = sorted(raw_root.rglob("*.txt"))
        if txt_files:
            return [_read_text_file(path) for path in txt_files]

    archive_path = data_dir / "openwebtext-10k.tar.xz"
    if not archive_path.exists():
        _download_file(OPENWEBTEXT_10K_URL, archive_path)

    with tarfile.open(archive_path, mode="r:xz") as tar:
        tar.extractall(path=raw_root)

    owt_dir = raw_root / "openwebtext-10k"
    if not owt_dir.exists():
        raise FileNotFoundError(f"Expected extracted directory at {owt_dir}")

    txt_files = []
    for xz_path in sorted(owt_dir.glob("*.xz")):
        if xz_path.name.endswith(".lock"):
            continue
        extract_dir = owt_dir / xz_path.stem
        if not any(extract_dir.rglob("*.txt")):
            _extract_xz_archive(xz_path, extract_dir)
        txt_files.extend(sorted(extract_dir.rglob("*.txt")))

    if not txt_files:
        raise RuntimeError(f"No text files found under {owt_dir}")

    marker.write_text(str(len(txt_files)))
    return [_read_text_file(path) for path in txt_files]


def _read_text_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return re.sub(r"\n\n\n+", "\n\n", text).strip()


def prepare_tokenized_text_dataset(
    data_dir,
    texts,
    dataset_name,
    tokenizer_name=DEFAULT_TOKENIZER,
    val_ratio=0.01,
    min_seq_len=8,
):
    """
    Tokenize raw texts and build train/val index files.
    """
    logger.info(f"Tokenizing {len(texts)} examples for {dataset_name}...")
    data_dir.mkdir(parents=True, exist_ok=True)
    tokens_dir = data_dir / "tokens"
    tokens_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_index = []
    val_index = []
    pad_id = tokenizer.pad_token_id
    val_mod = max(1, int(round(1 / val_ratio)))

    for idx, text in enumerate(texts):
        if not text:
            continue

        tokens = tokenizer.encode(text, add_special_tokens=True)
        seq_len = len(tokens)
        if seq_len < min_seq_len:
            continue

        token_path = tokens_dir / f"{idx}.pt"
        torch.save(torch.tensor(tokens, dtype=torch.long), token_path)

        entry = {"text_id": str(idx), "seq_len": seq_len}
        if idx % val_mod == 0:
            val_index.append(entry)
        else:
            train_index.append(entry)

        if idx % 1000 == 0 and idx > 0:
            logger.info(f"Processed {idx} examples...")

    write_json(train_index, data_dir / "train_index.json")
    write_json(val_index, data_dir / "val_index.json")
    write_json(
        {
            "vocab_size": len(tokenizer),
            "pad_id": pad_id,
            "tokenizer_name": tokenizer_name,
            "dataset_name": dataset_name,
        },
        data_dir / "meta.json",
    )
    logger.info(
        f"Saved dataset index for {dataset_name}: "
        f"train={len(train_index)}, val={len(val_index)}"
    )


def prepare_openwebtext_10k_dataset(
    data_dir,
    dataset_name="stas/openwebtext-10k",
    tokenizer_name=DEFAULT_TOKENIZER,
    val_ratio=0.01,
    min_seq_len=8,
    download_limit=None,
):
    texts = load_openwebtext_10k_texts(data_dir)
    if download_limit is not None:
        texts = texts[:download_limit]
    prepare_tokenized_text_dataset(
        data_dir=data_dir,
        texts=texts,
        dataset_name=dataset_name,
        tokenizer_name=tokenizer_name,
        val_ratio=val_ratio,
        min_seq_len=min_seq_len,
    )
