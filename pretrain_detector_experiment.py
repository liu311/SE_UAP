#!/usr/bin/env python3
"""
Source-domain adversarial detector pretraining and validation.

This script is intended for the P0-1 detector sanity experiment before training SE-UAP:

    LibriSpeech -> train f_D^L -> evaluate on LibriSpeech
    TIMIT       -> train f_D^T -> evaluate on TIMIT

The detector keeps the existing project convention:
    class 0 = adversarial
    class 1 = benign

For reported metrics, however, "adversarial" is treated as the positive class, so
Precision/Recall/F1/AUROC measure attack detection quality.

Adversarial training/evaluation samples can be generated from:
    - BIM / PGD, which require the speaker model gradient
    - Gaussian noise, which does not require the speaker model
    - optional UAPGN generator checkpoints
    - optional UAP-Penalty perturbation pools (.npy)

The optional UAPGN/UAP-Penalty inputs are deliberately external. The current repository
does not contain faithful baseline implementations for them, and detector validation should
not fabricate baseline artifacts inside this script.
"""

import argparse
import csv
import os
import random
import time
from dataclasses import dataclass
from functools import partial
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from adversarial_utils import (
    AdversarialDetectorCNN,
    TransformerAdversarialDetector,
    load_speaker_model,
    read_list,
)
from audio_utils import load_audio
from common.dataset import TIMITDetectorDataset, LibriSpeechDetectorDataset
from common.trainer import load_checkpoint
from generator import Generator1D


BENIGN_LABEL = 1
ADV_LABEL = 0


@dataclass
class DomainConfig:
    name: str
    data_root: str
    speaker_model: Optional[str] = None
    speaker_cfg: Optional[str] = None
    uapgn_ckpt: Optional[str] = None
    uap_penalty_npy: Optional[str] = None


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_worker(worker_id: int, seed: int):
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def build_dataset(domain: DomainConfig, train: bool, wlen: int, frame_mode: str):
    if domain.name == "timit":
        return TIMITDetectorDataset(domain.data_root, train=train, wlen=wlen, frame_mode=frame_mode)
    if domain.name == "libri":
        return LibriSpeechDetectorDataset(domain.data_root, train=train, wlen=wlen, frame_mode=frame_mode)
    raise ValueError(f"Unsupported domain: {domain.name}")


def domain_paths(domain: DomainConfig):
    if domain.name == "timit":
        return {
            "data_folder": domain.data_root,
            "train_scp": os.path.join(domain.data_root, "processed", "train_8_2.scp"),
            "test_scp": os.path.join(domain.data_root, "processed", "test_8_2.scp"),
            "label_dict": os.path.join(domain.data_root, "processed", "TIMIT_labels.npy"),
        }
    if domain.name == "libri":
        split_dir = os.path.join(domain.data_root, "Librispeech_spkid_sel", "split_spk_list")
        return {
            "data_folder": os.path.join(domain.data_root, "Librispeech_spkid_sel"),
            "train_scp": os.path.join(split_dir, "libri_tr_8_2.scp"),
            "test_scp": os.path.join(split_dir, "libri_te_8_2.scp"),
            "label_dict": os.path.join(split_dir, "libri_dict.npy"),
        }
    raise ValueError(f"Unsupported domain: {domain.name}")


class UtteranceDataset(Dataset):
    def __init__(self, domain: DomainConfig, train: bool):
        paths = domain_paths(domain)
        self.file_list = read_list(paths["train_scp"] if train else paths["test_scp"])
        self.data_folder = paths["data_folder"]
        self.label_dict = np.load(paths["label_dict"], allow_pickle=True).item()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        fname = self.file_list[index]
        audio, _, norm_factor = load_audio(os.path.join(self.data_folder, fname),
                                           expected_fs=16000, normalize=True)
        return torch.from_numpy(audio).float(), int(self.label_dict[fname]), norm_factor


def utterance_collate(batch):
    audios, labels, norm_factors = zip(*batch)
    return list(audios), torch.tensor(labels).long(), torch.tensor(norm_factors).float()


def frame_starts(length: int, wlen: int, wshift: int, max_frames: int = None):
    if length <= wlen:
        return [0]
    starts = list(range(0, length - wlen + 1, wshift))
    if starts[-1] != length - wlen:
        starts.append(length - wlen)
    if max_frames and len(starts) > max_frames:
        idx = np.linspace(0, len(starts) - 1, max_frames).round().astype(int)
        starts = [starts[i] for i in idx]
    return starts


def extract_frames(wav: torch.Tensor, starts: List[int], wlen: int):
    frames = []
    length = wav.numel()
    for start in starts:
        if length <= wlen:
            frame = torch.zeros(wlen, device=wav.device, dtype=wav.dtype)
            frame[:length] = wav
        else:
            frame = wav[start:start + wlen]
        frames.append(frame)
    return torch.stack(frames, dim=0)


def build_detector(args):
    if args.detector_type == "transformer":
        kwargs = {
            "frame_dim": args.frame_dim,
            "patch_size": args.transformer_patch_size,
            "embed_dim": args.transformer_embed_dim,
            "num_heads": args.transformer_heads,
            "num_layers": args.transformer_layers,
            "ff_dim": args.transformer_ff_dim,
            "dropout": args.transformer_dropout,
        }
        return TransformerAdversarialDetector(**kwargs), kwargs
    return AdversarialDetectorCNN(), {}


def aggregate_utterance_logits(detector, frames: torch.Tensor):
    logits = detector(frames)
    probs = torch.softmax(logits, dim=1).mean(dim=0, keepdim=True)
    return torch.log(probs.clamp_min(1e-12)), logits


def project_linf(adv: torch.Tensor, clean: torch.Tensor, epsilon: float):
    perturbation = torch.clamp(adv - clean, -epsilon, epsilon)
    return torch.clamp(clean + perturbation, -1.0, 1.0)


def bim_attack(
    speaker_model,
    clean: torch.Tensor,
    speaker_id: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
):
    adv = clean.detach().clone()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(speaker_model(adv), speaker_id)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        adv = project_linf(adv, clean, epsilon).detach()
    return adv


def pgd_attack(
    speaker_model,
    clean: torch.Tensor,
    speaker_id: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
):
    adv = clean.detach() + torch.empty_like(clean).uniform_(-epsilon, epsilon)
    adv = torch.clamp(adv, -1.0, 1.0)
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(speaker_model(adv), speaker_id)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        adv = project_linf(adv, clean, epsilon).detach()
    return adv


def load_uapgn(domain: DomainConfig, noise_dim: int, frame_dim: int, device):
    if not domain.uapgn_ckpt:
        return None
    model = Generator1D(noise_dim=noise_dim, output_dim=frame_dim).to(device)
    load_checkpoint(model, domain.uapgn_ckpt, map_location=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_uap_penalty_pool(domain: DomainConfig):
    if not domain.uap_penalty_npy:
        return None
    pool = np.load(domain.uap_penalty_npy, allow_pickle=True)
    pool = np.asarray(pool, dtype=np.float32)
    if pool.ndim == 1:
        pool = pool[None, :]
    if pool.ndim != 2:
        raise ValueError(f"Expected 1-D or 2-D UAP-Penalty pool, got shape {pool.shape}")
    return pool


def attacks_require_speaker_model(attacks: Iterable[str]):
    return any(attack in {"bim", "pgd"} for attack in attacks)


def ensure_speaker_model_available(domain: DomainConfig, attacks: Iterable[str]):
    if attacks_require_speaker_model(attacks) and (not domain.speaker_model or not domain.speaker_cfg):
        raise ValueError(
            f"{domain.name} uses BIM/PGD, so --{domain.name}_speaker_model and "
            f"--{domain.name}_speaker_cfg are required."
        )


def resize_perturbation_np(delta: np.ndarray, frame_dim: int):
    if delta.shape[0] == frame_dim:
        return delta
    repeats = int(np.ceil(frame_dim / delta.shape[0]))
    return np.tile(delta, repeats)[:frame_dim]


def make_adversarial_batch(
    attacks: List[str],
    speaker_model,
    clean: torch.Tensor,
    speaker_id: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    uapgn_model,
    uap_penalty_pool,
    noise_dim: int,
):
    """Return concatenated adversarial examples for all configured attack types."""
    adv_batches = []
    device = clean.device
    frame_dim = clean.shape[1]

    for attack in attacks:
        if attack == "bim":
            if speaker_model is None:
                raise ValueError("BIM requires a speaker model")
            adv_batches.append(bim_attack(speaker_model, clean, speaker_id, epsilon, alpha, steps))
        elif attack == "pgd":
            if speaker_model is None:
                raise ValueError("PGD requires a speaker model")
            adv_batches.append(pgd_attack(speaker_model, clean, speaker_id, epsilon, alpha, steps))
        elif attack == "gaussian":
            noise = torch.randn_like(clean)
            peak = noise.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
            delta = epsilon * noise / peak
            adv_batches.append(torch.clamp(clean + delta, -1.0, 1.0))
        elif attack == "uapgn":
            if uapgn_model is None:
                continue
            with torch.no_grad():
                noise = torch.randn(clean.size(0), noise_dim, device=device)
                delta = torch.clamp(uapgn_model(noise), -epsilon, epsilon)
                adv_batches.append(torch.clamp(clean + delta, -1.0, 1.0))
        elif attack == "uap_penalty":
            if uap_penalty_pool is None:
                continue
            idx = np.random.randint(0, len(uap_penalty_pool), size=clean.size(0))
            deltas = np.stack([resize_perturbation_np(uap_penalty_pool[i], frame_dim) for i in idx])
            delta = torch.from_numpy(deltas).float().to(device).clamp(-epsilon, epsilon)
            adv_batches.append(torch.clamp(clean + delta, -1.0, 1.0))
        else:
            raise ValueError(f"Unknown attack: {attack}")

    if not adv_batches:
        raise ValueError("No adversarial samples were generated. Check --attacks and optional baseline paths.")
    return torch.cat(adv_batches, dim=0)


def train_detector_for_domain_utterance(
    train_domain: DomainConfig,
    args,
    device,
):
    train_dataset = UtteranceDataset(train_domain, train=True)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=partial(init_worker, seed=args.seed),
        collate_fn=utterance_collate,
    )

    ensure_speaker_model_available(train_domain, args.attacks)
    speaker_model = None
    if attacks_require_speaker_model(args.attacks):
        speaker_model = load_speaker_model(train_domain.speaker_model, train_domain.speaker_cfg, device)
    uapgn_model = load_uapgn(train_domain, args.noise_dim, args.frame_dim, device)
    uap_penalty_pool = load_uap_penalty_pool(train_domain)

    detector, detector_kwargs = build_detector(args)
    detector = detector.to(device)
    optimizer = torch.optim.Adam(detector.parameters(), lr=args.lr)

    print(f"\n=== Utterance-level train {args.detector_type} detector on {train_domain.name} ===")
    print(f"Attacks: {', '.join(args.attacks)}")
    print(f"Train utterances: {len(train_dataset)}")

    for epoch in range(args.epochs):
        detector.train()
        losses = []
        pbar = tqdm(loader, desc=f"{train_domain.name} utterance epoch {epoch + 1}/{args.epochs}", unit="batch")
        for audios, speaker_ids, _ in pbar:
            speaker_ids = speaker_ids.long().to(device)
            loss_items = []

            for audio_cpu, speaker_id in zip(audios, speaker_ids):
                audio = audio_cpu.to(device)
                starts = frame_starts(audio.numel(), args.frame_dim, args.train_wshift, args.train_max_frames)
                clean = extract_frames(audio, starts, args.frame_dim)
                frame_speaker_ids = torch.full((clean.size(0),), int(speaker_id.item()),
                                               dtype=torch.long, device=device)

                clean_logits, _ = aggregate_utterance_logits(detector, clean)
                clean_loss = F.cross_entropy(clean_logits, torch.tensor([BENIGN_LABEL], device=device))

                adv_losses = []
                for attack in args.attacks:
                    adv = make_adversarial_batch(
                        [attack],
                        speaker_model,
                        clean,
                        frame_speaker_ids,
                        args.epsilon,
                        args.alpha,
                        args.steps,
                        uapgn_model,
                        uap_penalty_pool,
                        args.noise_dim,
                    )
                    adv_logits, _ = aggregate_utterance_logits(detector, adv)
                    adv_losses.append(F.cross_entropy(adv_logits, torch.tensor([ADV_LABEL], device=device)))

                loss_items.append(0.5 * clean_loss + 0.5 * torch.stack(adv_losses).mean())

            loss = torch.stack(loss_items).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            pbar.set_postfix(loss=f"{np.mean(losses):.4f}")

        print(f"Epoch {epoch + 1}: loss={np.mean(losses):.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, f"{args.detector_type}_utterance_detector_trained_on_{train_domain.name}.pth")
    torch.save(
        {
            "state_dict": detector.state_dict(),
            "detector_type": args.detector_type,
            "detector_kwargs": detector_kwargs,
            "training_level": "utterance",
            "train_domain": train_domain.name,
            "attacks": args.attacks,
            "epsilon": args.epsilon,
            "alpha": args.alpha,
            "steps": args.steps,
            "wlen": args.wlen,
            "frame_dim": args.frame_dim,
            "train_wshift": args.train_wshift,
            "train_max_frames": args.train_max_frames,
        },
        ckpt_path,
    )
    print(f"Saved detector: {ckpt_path}")
    return detector.eval(), ckpt_path


def train_detector_for_domain(
    train_domain: DomainConfig,
    args,
    device,
):
    if args.training_level == "utterance":
        return train_detector_for_domain_utterance(train_domain, args, device)

    train_dataset = build_dataset(train_domain, train=True, wlen=args.wlen, frame_mode=args.frame_mode)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=partial(init_worker, seed=args.seed),
    )

    ensure_speaker_model_available(train_domain, args.attacks)
    speaker_model = None
    if attacks_require_speaker_model(args.attacks):
        speaker_model = load_speaker_model(train_domain.speaker_model, train_domain.speaker_cfg, device)
    uapgn_model = load_uapgn(train_domain, args.noise_dim, args.frame_dim, device)
    uap_penalty_pool = load_uap_penalty_pool(train_domain)

    detector, detector_kwargs = build_detector(args)
    detector = detector.to(device)
    optimizer = torch.optim.Adam(detector.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    print(f"\n=== Train detector on {train_domain.name} ===")
    print(f"Attacks: {', '.join(args.attacks)}")
    print(f"Train files: {len(train_dataset)}")

    for epoch in range(args.epochs):
        detector.train()
        losses = []
        pbar = tqdm(loader, desc=f"{train_domain.name} epoch {epoch + 1}/{args.epochs}", unit="batch")
        for clean, speaker_id, _ in pbar:
            clean = clean.float().to(device)
            speaker_id = speaker_id.long().to(device)

            adv = make_adversarial_batch(
                args.attacks,
                speaker_model,
                clean,
                speaker_id,
                args.epsilon,
                args.alpha,
                args.steps,
                uapgn_model,
                uap_penalty_pool,
                args.noise_dim,
            )

            inputs = torch.cat([clean, adv], dim=0)
            labels = torch.cat([
                torch.full((clean.size(0),), BENIGN_LABEL, dtype=torch.long, device=device),
                torch.full((adv.size(0),), ADV_LABEL, dtype=torch.long, device=device),
            ])

            optimizer.zero_grad(set_to_none=True)
            logits = detector(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            pbar.set_postfix(loss=f"{np.mean(losses):.4f}")

        print(f"Epoch {epoch + 1}: loss={np.mean(losses):.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, f"detector_trained_on_{train_domain.name}.pth")
    torch.save(
        {
            "state_dict": detector.state_dict(),
            "detector_type": args.detector_type,
            "detector_kwargs": detector_kwargs,
            "training_level": "frame",
            "train_domain": train_domain.name,
            "attacks": args.attacks,
            "epsilon": args.epsilon,
            "alpha": args.alpha,
            "steps": args.steps,
            "wlen": args.wlen,
            "frame_dim": args.frame_dim,
        },
        ckpt_path,
    )
    print(f"Saved detector: {ckpt_path}")
    return detector.eval(), ckpt_path


def binary_metrics_from_scores(y_true_adv: np.ndarray, adv_scores: np.ndarray, y_pred_adv: np.ndarray):
    tp = int(((y_pred_adv == 1) & (y_true_adv == 1)).sum())
    tn = int(((y_pred_adv == 0) & (y_true_adv == 0)).sum())
    fp = int(((y_pred_adv == 1) & (y_true_adv == 0)).sum())
    fn = int(((y_pred_adv == 0) & (y_true_adv == 1)).sum())

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    auroc = auroc_from_scores(y_true_adv, adv_scores)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
    }


def prefixed_metrics(prefix: str, metrics: Dict):
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def auroc_from_scores(y_true_adv: np.ndarray, scores: np.ndarray):
    """Mann-Whitney U based AUROC without sklearn."""
    pos = scores[y_true_adv == 1]
    neg = scores[y_true_adv == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)

    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[start:end] = avg_rank
        start = end

    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    pos_ranks = original_ranks[y_true_adv == 1].sum()
    n_pos = len(pos)
    n_neg = len(neg)
    return (pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def evaluate_detector_on_domain_utterance(
    detector,
    detector_train_domain: str,
    test_domain: DomainConfig,
    args,
    device,
):
    test_dataset = UtteranceDataset(test_domain, train=False)
    if args.max_eval_files > 0:
        test_dataset.file_list = test_dataset.file_list[:args.max_eval_files]
    loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=utterance_collate,
    )

    if args.eval_speaker_effect and (not test_domain.speaker_model or not test_domain.speaker_cfg):
        raise ValueError(
            f"--eval_speaker_effect requires --{test_domain.name}_speaker_model and "
            f"--{test_domain.name}_speaker_cfg, even for model-independent attacks such as Gaussian noise."
        )

    ensure_speaker_model_available(test_domain, args.attacks)
    speaker_model = None
    if args.eval_speaker_effect or attacks_require_speaker_model(args.attacks):
        speaker_model = load_speaker_model(test_domain.speaker_model, test_domain.speaker_cfg, device)
    uapgn_model = load_uapgn(test_domain, args.noise_dim, args.frame_dim, device)
    uap_penalty_pool = load_uap_penalty_pool(test_domain)

    y_true = []
    y_pred = []
    adv_scores = []
    y_true_success = []
    y_pred_success = []
    adv_scores_success = []

    clean_speaker_correct = 0
    adv_speaker_correct = 0
    adv_speaker_total = 0
    successful_adv_total = 0

    print(f"\n=== Utterance-level evaluate {args.detector_type} detector trained on "
          f"{detector_train_domain} against {test_domain.name} ===")
    pbar = tqdm(loader, desc=f"eval {detector_train_domain}->{test_domain.name}", unit="utt")
    for audios, speaker_ids, _ in pbar:
        audio = audios[0].to(device)
        speaker_id = speaker_ids.long().to(device)
        starts = frame_starts(audio.numel(), args.frame_dim, args.eval_wshift, args.eval_max_frames)
        clean = extract_frames(audio, starts, args.frame_dim)
        frame_speaker_ids = torch.full((clean.size(0),), int(speaker_id.item()),
                                       dtype=torch.long, device=device)

        with torch.no_grad():
            clean_logits, _ = aggregate_utterance_logits(detector, clean)
            clean_probs = torch.softmax(clean_logits, dim=1)[0]
            clean_pred_internal = clean_probs.argmax().item()
            if args.eval_speaker_effect:
                clean_speaker_probs = torch.softmax(speaker_model(clean), dim=1).mean(dim=0)
                clean_speaker_correct += int(clean_speaker_probs.argmax().item() == int(speaker_id.item()))

        y_true.append(np.array([0], dtype=np.int32))
        y_pred.append(np.array([int(clean_pred_internal == ADV_LABEL)], dtype=np.int32))
        adv_scores.append(np.array([float(clean_probs[ADV_LABEL].item())], dtype=np.float64))

        clean_success_true = np.array([0], dtype=np.int32)
        clean_success_pred = np.array([int(clean_pred_internal == ADV_LABEL)], dtype=np.int32)
        clean_success_score = np.array([float(clean_probs[ADV_LABEL].item())], dtype=np.float64)

        for attack in args.attacks:
            with torch.enable_grad():
                adv = make_adversarial_batch(
                    [attack],
                    speaker_model,
                    clean,
                    frame_speaker_ids,
                    args.epsilon,
                    args.alpha,
                    args.steps,
                    uapgn_model,
                    uap_penalty_pool,
                    args.noise_dim,
                )

            with torch.no_grad():
                adv_logits, _ = aggregate_utterance_logits(detector, adv)
                adv_probs = torch.softmax(adv_logits, dim=1)[0]
                adv_pred_internal = adv_probs.argmax().item()

                y_true.append(np.array([1], dtype=np.int32))
                y_pred.append(np.array([int(adv_pred_internal == ADV_LABEL)], dtype=np.int32))
                adv_scores.append(np.array([float(adv_probs[ADV_LABEL].item())], dtype=np.float64))

                if args.eval_speaker_effect:
                    adv_speaker_probs = torch.softmax(speaker_model(adv), dim=1).mean(dim=0)
                    adv_success = adv_speaker_probs.argmax().item() != int(speaker_id.item())
                    adv_speaker_correct += int(not adv_success)
                    adv_speaker_total += 1
                    successful_adv_total += int(adv_success)

                    if adv_success:
                        y_true_success.append(np.concatenate([clean_success_true, np.array([1], dtype=np.int32)]))
                        y_pred_success.append(np.concatenate([
                            clean_success_pred,
                            np.array([int(adv_pred_internal == ADV_LABEL)], dtype=np.int32),
                        ]))
                        adv_scores_success.append(np.concatenate([
                            clean_success_score,
                            np.array([float(adv_probs[ADV_LABEL].item())], dtype=np.float64),
                        ]))

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    adv_scores = np.concatenate(adv_scores)
    metrics = prefixed_metrics("detector_all_adv", binary_metrics_from_scores(y_true, adv_scores, y_pred))

    if args.eval_speaker_effect:
        clean_total = len(test_dataset)
        clean_speaker_acc = clean_speaker_correct / max(clean_total, 1)
        adv_speaker_acc = adv_speaker_correct / max(adv_speaker_total, 1)
        fooling_rate = successful_adv_total / max(adv_speaker_total, 1)
        metrics.update({
            "speaker_clean_accuracy": clean_speaker_acc,
            "speaker_adv_accuracy": adv_speaker_acc,
            "speaker_fooling_rate": fooling_rate,
            "successful_adv_examples": successful_adv_total,
        })

        if y_true_success:
            y_true_success = np.concatenate(y_true_success)
            y_pred_success = np.concatenate(y_pred_success)
            adv_scores_success = np.concatenate(adv_scores_success)
            metrics.update(prefixed_metrics(
                "detector_success_adv",
                binary_metrics_from_scores(y_true_success, adv_scores_success, y_pred_success),
            ))
        else:
            metrics.update({
                "detector_success_adv_accuracy": float("nan"),
                "detector_success_adv_precision": float("nan"),
                "detector_success_adv_recall": float("nan"),
                "detector_success_adv_f1": float("nan"),
                "detector_success_adv_auroc": float("nan"),
            })

    metrics.update({
        "detector_training": detector_train_domain,
        "detector_test": test_domain.name,
        "num_eval_examples": int(len(y_true)),
    })
    return metrics


@torch.no_grad()
def evaluate_detector_on_domain(
    detector,
    detector_train_domain: str,
    test_domain: DomainConfig,
    args,
    device,
):
    if args.training_level == "utterance":
        return evaluate_detector_on_domain_utterance(
            detector, detector_train_domain, test_domain, args, device
        )

    test_dataset = build_dataset(test_domain, train=False, wlen=args.wlen, frame_mode="fixed")
    loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    if args.eval_speaker_effect and (not test_domain.speaker_model or not test_domain.speaker_cfg):
        raise ValueError(
            f"--eval_speaker_effect requires --{test_domain.name}_speaker_model and "
            f"--{test_domain.name}_speaker_cfg, even for model-independent attacks such as Gaussian noise."
        )

    ensure_speaker_model_available(test_domain, args.attacks)
    speaker_model = None
    if args.eval_speaker_effect or attacks_require_speaker_model(args.attacks):
        speaker_model = load_speaker_model(test_domain.speaker_model, test_domain.speaker_cfg, device)
    uapgn_model = load_uapgn(test_domain, args.noise_dim, args.frame_dim, device)
    uap_penalty_pool = load_uap_penalty_pool(test_domain)

    y_true = []
    y_pred = []
    adv_scores = []
    y_true_success = []
    y_pred_success = []
    adv_scores_success = []

    clean_speaker_correct = 0
    adv_speaker_correct = 0
    adv_speaker_total = 0
    successful_adv_total = 0

    print(f"\n=== Evaluate detector trained on {detector_train_domain} against {test_domain.name} ===")
    pbar = tqdm(loader, desc=f"eval {detector_train_domain}->{test_domain.name}", unit="batch")
    for clean, speaker_id, _ in pbar:
        clean = clean.float().to(device)
        speaker_id = speaker_id.long().to(device)

        with torch.enable_grad():
            adv = make_adversarial_batch(
                args.attacks,
                speaker_model,
                clean,
                speaker_id,
                args.epsilon,
                args.alpha,
                args.steps,
                uapgn_model,
                uap_penalty_pool,
                args.noise_dim,
            )

        inputs = torch.cat([clean, adv], dim=0)
        labels_internal = torch.cat([
            torch.full((clean.size(0),), BENIGN_LABEL, dtype=torch.long, device=device),
            torch.full((adv.size(0),), ADV_LABEL, dtype=torch.long, device=device),
        ])

        logits = detector(inputs)
        probs = torch.softmax(logits, dim=1)
        pred_internal = logits.argmax(dim=1)

        if args.eval_speaker_effect:
            clean_pred = speaker_model(clean).argmax(dim=1)
            clean_speaker_correct += int((clean_pred == speaker_id).sum().item())

            repeat_factor = adv.size(0) // clean.size(0)
            adv_speaker_id = speaker_id.repeat(repeat_factor)
            adv_pred = speaker_model(adv).argmax(dim=1)
            adv_success = adv_pred != adv_speaker_id

            adv_speaker_correct += int((adv_pred == adv_speaker_id).sum().item())
            adv_speaker_total += int(adv.size(0))
            successful_adv_total += int(adv_success.sum().item())

            if adv_success.sum().item() > 0:
                clean_true = torch.zeros(clean.size(0), dtype=torch.long, device=device)
                success_true = torch.ones(int(adv_success.sum().item()), dtype=torch.long, device=device)
                y_true_success.append(torch.cat([clean_true, success_true], dim=0).cpu().numpy().astype(np.int32))

                clean_pred_adv = (pred_internal[:clean.size(0)] == ADV_LABEL).long()
                success_pred_adv = (pred_internal[clean.size(0):][adv_success] == ADV_LABEL).long()
                y_pred_success.append(torch.cat([clean_pred_adv, success_pred_adv], dim=0).cpu().numpy().astype(np.int32))

                clean_scores = probs[:clean.size(0), ADV_LABEL]
                success_scores = probs[clean.size(0):][adv_success, ADV_LABEL]
                adv_scores_success.append(torch.cat([clean_scores, success_scores], dim=0).cpu().numpy())

        # Report adversarial as positive.
        y_true.append((labels_internal == ADV_LABEL).cpu().numpy().astype(np.int32))
        y_pred.append((pred_internal == ADV_LABEL).cpu().numpy().astype(np.int32))
        adv_scores.append(probs[:, ADV_LABEL].cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    adv_scores = np.concatenate(adv_scores)
    metrics = prefixed_metrics("detector_all_adv", binary_metrics_from_scores(y_true, adv_scores, y_pred))

    if args.eval_speaker_effect:
        clean_total = int(len(y_true) - adv_speaker_total)
        clean_speaker_acc = clean_speaker_correct / max(clean_total, 1)
        adv_speaker_acc = adv_speaker_correct / max(adv_speaker_total, 1)
        fooling_rate = successful_adv_total / max(adv_speaker_total, 1)
        metrics.update({
            "speaker_clean_accuracy": clean_speaker_acc,
            "speaker_adv_accuracy": adv_speaker_acc,
            "speaker_fooling_rate": fooling_rate,
            "successful_adv_examples": successful_adv_total,
        })

        if y_true_success:
            y_true_success = np.concatenate(y_true_success)
            y_pred_success = np.concatenate(y_pred_success)
            adv_scores_success = np.concatenate(adv_scores_success)
            metrics.update(prefixed_metrics(
                "detector_success_adv",
                binary_metrics_from_scores(y_true_success, adv_scores_success, y_pred_success),
            ))
        else:
            metrics.update({
                "detector_success_adv_accuracy": float("nan"),
                "detector_success_adv_precision": float("nan"),
                "detector_success_adv_recall": float("nan"),
                "detector_success_adv_f1": float("nan"),
                "detector_success_adv_auroc": float("nan"),
            })

    metrics.update({
        "detector_training": detector_train_domain,
        "detector_test": test_domain.name,
        "num_eval_examples": int(len(y_true)),
    })
    return metrics


def write_results(results: List[Dict], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"source_domain_detector_results_{timestamp}.csv")
    md_path = os.path.join(output_dir, f"source_domain_detector_results_{timestamp}.md")

    columns = [
        "detector_training",
        "detector_test",
        "speaker_clean_accuracy",
        "speaker_adv_accuracy",
        "speaker_fooling_rate",
        "successful_adv_examples",
        "detector_all_adv_accuracy",
        "detector_all_adv_precision",
        "detector_all_adv_recall",
        "detector_all_adv_f1",
        "detector_all_adv_auroc",
        "detector_success_adv_accuracy",
        "detector_success_adv_precision",
        "detector_success_adv_recall",
        "detector_success_adv_f1",
        "detector_success_adv_auroc",
        "num_eval_examples",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, float("nan")) for k in columns})

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(
            "| Detector training | Detector test | Speaker clean ACC | Speaker adv ACC | "
            "Fooling rate | Detector all ACC | Detector all precision | Detector all recall | "
            "Detector all F1 | Detector all AUROC | Detector successful-adv ACC | "
            "Detector successful-adv precision | Detector successful-adv recall | "
            "Detector successful-adv F1 | Detector successful-adv AUROC |\n"
        )
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in results:
            f.write(
                f"| {row['detector_training']} | {row['detector_test']} | "
                f"{row.get('speaker_clean_accuracy', float('nan')):.4f} | "
                f"{row.get('speaker_adv_accuracy', float('nan')):.4f} | "
                f"{row.get('speaker_fooling_rate', float('nan')):.4f} | "
                f"{row.get('detector_all_adv_accuracy', float('nan')):.4f} | "
                f"{row.get('detector_all_adv_precision', float('nan')):.4f} | "
                f"{row.get('detector_all_adv_recall', float('nan')):.4f} | "
                f"{row.get('detector_all_adv_f1', float('nan')):.4f} | "
                f"{row.get('detector_all_adv_auroc', float('nan')):.4f} | "
                f"{row.get('detector_success_adv_accuracy', float('nan')):.4f} | "
                f"{row.get('detector_success_adv_precision', float('nan')):.4f} | "
                f"{row.get('detector_success_adv_recall', float('nan')):.4f} | "
                f"{row.get('detector_success_adv_f1', float('nan')):.4f} | "
                f"{row.get('detector_success_adv_auroc', float('nan')):.4f} |\n"
            )

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved Markdown table: {md_path}")
    return csv_path, md_path


def parse_args():
    parser = argparse.ArgumentParser("Source-domain adversarial detector experiment")

    parser.add_argument("--timit_data_root", type=str, required=True)
    parser.add_argument("--libri_data_root", type=str, required=True)
    parser.add_argument("--timit_speaker_model", type=str, default=None)
    parser.add_argument("--timit_speaker_cfg", type=str, default=None)
    parser.add_argument("--libri_speaker_model", type=str, default=None)
    parser.add_argument("--libri_speaker_cfg", type=str, default=None)

    parser.add_argument("--timit_uapgn_ckpt", type=str, default=None)
    parser.add_argument("--libri_uapgn_ckpt", type=str, default=None)
    parser.add_argument("--timit_uap_penalty_npy", type=str, default=None)
    parser.add_argument("--libri_uap_penalty_npy", type=str, default=None)

    parser.add_argument(
        "--attacks",
        nargs="+",
        choices=["bim", "pgd", "gaussian", "uapgn", "uap_penalty"],
        default=["bim", "pgd"],
        help="Adversarial sources for D_adv. BIM/PGD require speaker models; gaussian does not.",
    )
    parser.add_argument("--train_domain", choices=["all", "timit", "libri"], default="all")
    parser.add_argument("--output_dir", type=str, default="./output/pretrain_detector")

    parser.add_argument("--wlen", type=int, default=200, help="frame length in ms")
    parser.add_argument("--frame_dim", type=int, default=3200)
    parser.add_argument("--frame_mode", choices=["fixed", "random"], default="random")
    parser.add_argument("--noise_dim", type=int, default=100)
    parser.add_argument("--training_level", choices=["frame", "utterance"], default="frame",
                        help="frame keeps the original detector training; utterance uses sentence-level aggregation.")
    parser.add_argument("--detector_type", choices=["cnn", "transformer"], default="cnn")
    parser.add_argument("--train_wshift", type=int, default=3200)
    parser.add_argument("--eval_wshift", type=int, default=3200)
    parser.add_argument("--train_max_frames", type=int, default=12)
    parser.add_argument("--eval_max_frames", type=int, default=0,
                        help="0 means use all sliding-window frames during utterance-level evaluation.")
    parser.add_argument("--max_eval_files", type=int, default=0,
                        help="0 means evaluate all test utterances in utterance-level mode.")
    parser.add_argument("--transformer_patch_size", type=int, default=80)
    parser.add_argument("--transformer_embed_dim", type=int, default=128)
    parser.add_argument("--transformer_heads", type=int, default=4)
    parser.add_argument("--transformer_layers", type=int, default=3)
    parser.add_argument("--transformer_ff_dim", type=int, default=256)
    parser.add_argument("--transformer_dropout", type=float, default=0.1)

    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.002)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--no_eval_speaker_effect",
        action="store_true",
        help="Skip speaker-model fooling-rate evaluation.",
    )
    parser.add_argument(
        "--eval_cross_dataset",
        action="store_true",
        help="Also evaluate each trained detector on the other dataset.",
    )
    args = parser.parse_args()
    args.eval_speaker_effect = not args.no_eval_speaker_effect
    return args


def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device(args.device)

    timit = DomainConfig(
        name="timit",
        data_root=args.timit_data_root,
        speaker_model=args.timit_speaker_model,
        speaker_cfg=args.timit_speaker_cfg,
        uapgn_ckpt=args.timit_uapgn_ckpt,
        uap_penalty_npy=args.timit_uap_penalty_npy,
    )
    libri = DomainConfig(
        name="libri",
        data_root=args.libri_data_root,
        speaker_model=args.libri_speaker_model,
        speaker_cfg=args.libri_speaker_cfg,
        uapgn_ckpt=args.libri_uapgn_ckpt,
        uap_penalty_npy=args.libri_uap_penalty_npy,
    )
    domains = {"timit": timit, "libri": libri}

    train_domains = ["libri", "timit"] if args.train_domain == "all" else [args.train_domain]
    results = []

    for train_name in train_domains:
        detector, _ = train_detector_for_domain(domains[train_name], args, device)
        eval_names = [train_name]
        if args.eval_cross_dataset:
            eval_names.append("timit" if train_name == "libri" else "libri")
        for test_name in eval_names:
            metrics = evaluate_detector_on_domain(
                detector,
                detector_train_domain=train_name,
                test_domain=domains[test_name],
                args=args,
                device=device,
            )
            results.append(metrics)
            print(
                f"{train_name}->{test_name}: "
                f"SpeakerFool={metrics.get('speaker_fooling_rate', float('nan')):.4f}, "
                f"DetAllAcc={metrics['detector_all_adv_accuracy']:.4f}, "
                f"DetAllP={metrics['detector_all_adv_precision']:.4f}, "
                f"DetAllF1={metrics['detector_all_adv_f1']:.4f}, "
                f"DetAllAUROC={metrics['detector_all_adv_auroc']:.4f}, "
                f"DetAllRecall={metrics['detector_all_adv_recall']:.4f}, "
                f"DetSuccessAcc={metrics.get('detector_success_adv_accuracy', float('nan')):.4f}, "
                f"DetSuccessP={metrics.get('detector_success_adv_precision', float('nan')):.4f}, "
                f"DetSuccessF1={metrics.get('detector_success_adv_f1', float('nan')):.4f}, "
                f"DetSuccessAUROC={metrics.get('detector_success_adv_auroc', float('nan')):.4f}, "
                f"DetSuccessRecall={metrics.get('detector_success_adv_recall', float('nan')):.4f}"
            )

    write_results(results, args.output_dir)


if __name__ == "__main__":
    main()
