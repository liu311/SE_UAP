#!/usr/bin/env python3
"""
Utterance-level cross-dataset detector baseline experiment.

Purpose:
    Before claiming SE-UAP is stealthy, validate whether the frozen detector itself
    can detect ordinary baseline adversarial utterances across datasets.

Experiments:
    LibriSpeech -> train f_D^L -> evaluate on LibriSpeech and TIMIT
    TIMIT       -> train f_D^T -> evaluate on TIMIT and LibriSpeech

Labels follow the project convention:
    class 0 = adversarial
    class 1 = benign

Reported Precision/Recall/F1/AUROC treat adversarial as the positive class.
Recall is the adversarial detection rate; FRD is 1 - Recall.
"""

import argparse
import csv
import os
import random
import time
from dataclasses import dataclass
from functools import partial
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from adversarial_utils import load_detector_model, load_speaker_model, read_list
from audio_utils import load_audio
from common.trainer import load_checkpoint
from detector_utils import AdversarialDetectorCNN
from generator import Generator1D


BENIGN_LABEL = 1
ADV_LABEL = 0


@dataclass
class DomainConfig:
    name: str
    data_root: str
    data_folder: str
    train_scp: str
    test_scp: str
    label_dict_path: str
    speaker_model: Optional[str]
    speaker_cfg: Optional[str]
    uapgn_ckpt: Optional[str] = None


class UtteranceDataset(Dataset):
    def __init__(self, domain: DomainConfig, split: str):
        self.domain = domain
        scp = domain.train_scp if split == "train" else domain.test_scp
        self.file_list = read_list(scp)
        self.label_dict = np.load(domain.label_dict_path, allow_pickle=True).item()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        fname = self.file_list[index]
        audio, _, _ = load_audio(os.path.join(self.domain.data_folder, fname),
                                 expected_fs=16000, normalize=True)
        return torch.from_numpy(audio).float(), int(self.label_dict[fname]), fname


def utterance_collate(batch):
    audios, labels, fnames = zip(*batch)
    return list(audios), torch.tensor(labels).long(), list(fnames)


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_worker(worker_id: int, seed: int):
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def make_domain(name: str, args) -> DomainConfig:
    if name == "timit":
        root = args.timit_data_root
        return DomainConfig(
            name="timit",
            data_root=root,
            data_folder=root,
            train_scp=os.path.join(root, "processed", "train_8_2.scp"),
            test_scp=os.path.join(root, "processed", "test_8_2.scp"),
            label_dict_path=os.path.join(root, "processed", "TIMIT_labels.npy"),
            speaker_model=args.timit_speaker_model,
            speaker_cfg=args.timit_speaker_cfg,
            uapgn_ckpt=args.timit_uapgn_ckpt,
        )
    if name == "libri":
        root = args.libri_data_root
        split_dir = os.path.join(root, "Librispeech_spkid_sel", "split_spk_list")
        return DomainConfig(
            name="libri",
            data_root=root,
            data_folder=os.path.join(root, "Librispeech_spkid_sel"),
            train_scp=os.path.join(split_dir, "libri_tr_8_2.scp"),
            test_scp=os.path.join(split_dir, "libri_te_8_2.scp"),
            label_dict_path=os.path.join(split_dir, "libri_dict.npy"),
            speaker_model=args.libri_speaker_model,
            speaker_cfg=args.libri_speaker_cfg,
            uapgn_ckpt=args.libri_uapgn_ckpt,
        )
    raise ValueError(f"Unsupported domain: {name}")


def attacks_require_speaker_model(attacks: Iterable[str]):
    return any(attack in {"bim", "pgd"} for attack in attacks)


def ensure_speaker_model(domain: DomainConfig, attacks: Iterable[str]):
    if attacks_require_speaker_model(attacks) and (not domain.speaker_model or not domain.speaker_cfg):
        raise ValueError(
            f"{domain.name} evaluation uses BIM/PGD, so its speaker model and cfg are required."
        )


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


def project_linf(adv: torch.Tensor, clean: torch.Tensor, epsilon: float):
    perturbation = torch.clamp(adv - clean, -epsilon, epsilon)
    return torch.clamp(clean + perturbation, -1.0, 1.0)


def bim_attack(speaker_model, clean: torch.Tensor, speaker_id: torch.Tensor,
               epsilon: float, alpha: float, steps: int):
    adv = clean.detach().clone()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(speaker_model(adv), speaker_id)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        adv = project_linf(adv, clean, epsilon).detach()
    return adv


def pgd_attack(speaker_model, clean: torch.Tensor, speaker_id: torch.Tensor,
               epsilon: float, alpha: float, steps: int):
    adv = clean.detach() + torch.empty_like(clean).uniform_(-epsilon, epsilon)
    adv = torch.clamp(adv, -1.0, 1.0)
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(speaker_model(adv), speaker_id)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        adv = project_linf(adv, clean, epsilon).detach()
    return adv


def gaussian_attack(clean: torch.Tensor, epsilon: float):
    noise = torch.randn_like(clean)
    peak = noise.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    delta = epsilon * noise / peak
    return torch.clamp(clean + delta, -1.0, 1.0)


def load_uapgn(domain: DomainConfig, args, device):
    if not domain.uapgn_ckpt:
        return None
    model = Generator1D(args.noise_dim, args.frame_dim).to(device)
    load_checkpoint(model, domain.uapgn_ckpt, map_location=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def make_adv_frames(attack: str, clean_frames: torch.Tensor, speaker_ids: torch.Tensor,
                    speaker_model, uapgn_model, args, device):
    if attack == "bim":
        return bim_attack(speaker_model, clean_frames, speaker_ids,
                          args.epsilon, args.attack_alpha, args.attack_steps)
    if attack == "pgd":
        return pgd_attack(speaker_model, clean_frames, speaker_ids,
                          args.epsilon, args.attack_alpha, args.attack_steps)
    if attack == "gaussian":
        return gaussian_attack(clean_frames, args.epsilon)
    if attack == "uapgn":
        if uapgn_model is None:
            raise ValueError("uapgn requires the corresponding --*_uapgn_ckpt")
        with torch.no_grad():
            z = torch.randn(clean_frames.size(0), args.noise_dim, device=device)
            delta = torch.clamp(uapgn_model(z), -args.epsilon, args.epsilon)
            return torch.clamp(clean_frames + delta, -1.0, 1.0)
    raise ValueError(f"Unknown attack: {attack}")


def aggregate_logits(model, frames: torch.Tensor):
    logits = model(frames)
    probs = torch.softmax(logits, dim=1).mean(dim=0, keepdim=True)
    return torch.log(probs.clamp_min(1e-12)), logits


def auroc_from_scores(y_true_adv: np.ndarray, scores: np.ndarray):
    pos = scores[y_true_adv == 1]
    neg = scores[y_true_adv == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    rank_sum_pos = ranks[y_true_adv == 1].sum()
    n_pos = len(pos)
    n_neg = len(neg)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def binary_metrics(y_true_adv: np.ndarray, adv_scores: np.ndarray, y_pred_adv: np.ndarray):
    tp = int(((y_pred_adv == 1) & (y_true_adv == 1)).sum())
    tn = int(((y_pred_adv == 0) & (y_true_adv == 0)).sum())
    fp = int(((y_pred_adv == 1) & (y_true_adv == 0)).sum())
    fn = int(((y_pred_adv == 0) & (y_true_adv == 1)).sum())
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc_from_scores(y_true_adv, adv_scores),
        "frd": 1.0 - recall,
        "clean_benign_rate": tn / max(tn + fp, 1),
        "adv_detection_rate": recall,
    }


def train_detector_utterance(train_domain: DomainConfig, args, device):
    ensure_speaker_model(train_domain, args.attacks)
    speaker_model = None
    if attacks_require_speaker_model(args.attacks):
        speaker_model = load_speaker_model(train_domain.speaker_model, train_domain.speaker_cfg, device)
        for p in speaker_model.parameters():
            p.requires_grad = False
    uapgn_model = load_uapgn(train_domain, args, device)

    dataset = UtteranceDataset(train_domain, split="train")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=partial(init_worker, seed=args.seed),
        collate_fn=utterance_collate,
    )

    detector = AdversarialDetectorCNN().to(device)
    optimizer = torch.optim.Adam(detector.parameters(), lr=args.lr)

    print(f"\n=== Utterance-level train detector on {train_domain.name} ===")
    print(f"Attacks: {', '.join(args.attacks)}")
    print(f"Train utterances: {len(dataset)}")

    for epoch in range(args.epochs):
        detector.train()
        losses = []
        pbar = tqdm(loader, desc=f"{train_domain.name} detector epoch {epoch + 1}/{args.epochs}", unit="batch")
        for audios, labels, _ in pbar:
            labels = labels.long().to(device)
            loss_items = []

            for audio_cpu, label in zip(audios, labels):
                audio = audio_cpu.to(device)
                starts = frame_starts(audio.numel(), args.frame_dim, args.train_wshift, args.train_max_frames)
                clean_frames = extract_frames(audio, starts, args.frame_dim)
                frame_labels = torch.full((clean_frames.size(0),), int(label.item()),
                                          dtype=torch.long, device=device)

                clean_utt_logits, _ = aggregate_logits(detector, clean_frames)
                clean_loss = F.cross_entropy(clean_utt_logits, torch.tensor([BENIGN_LABEL], device=device))

                adv_losses = []
                for attack in args.attacks:
                    with torch.enable_grad():
                        adv_frames = make_adv_frames(attack, clean_frames, frame_labels,
                                                     speaker_model, uapgn_model, args, device)
                    adv_utt_logits, _ = aggregate_logits(detector, adv_frames)
                    adv_losses.append(F.cross_entropy(adv_utt_logits, torch.tensor([ADV_LABEL], device=device)))

                loss_items.append(0.5 * clean_loss + 0.5 * torch.stack(adv_losses).mean())

            loss = torch.stack(loss_items).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        print(f"Epoch {epoch + 1}: loss={float(np.mean(losses)):.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, f"utterance_detector_trained_on_{train_domain.name}.pth")
    torch.save({
        "state_dict": detector.state_dict(),
        "train_domain": train_domain.name,
        "attacks": args.attacks,
        "epsilon": args.epsilon,
        "attack_alpha": args.attack_alpha,
        "attack_steps": args.attack_steps,
        "frame_dim": args.frame_dim,
        "train_wshift": args.train_wshift,
        "train_max_frames": args.train_max_frames,
    }, ckpt_path)
    print(f"Saved detector: {ckpt_path}")
    return detector.eval(), ckpt_path


def evaluate_detector_utterance(detector, detector_train_domain: str, test_domain: DomainConfig, args, device):
    ensure_speaker_model(test_domain, args.attacks)
    speaker_model = None
    if attacks_require_speaker_model(args.attacks):
        speaker_model = load_speaker_model(test_domain.speaker_model, test_domain.speaker_cfg, device)
        for p in speaker_model.parameters():
            p.requires_grad = False
    uapgn_model = load_uapgn(test_domain, args, device)

    dataset = UtteranceDataset(test_domain, split="test")
    if args.max_eval_files > 0:
        dataset.file_list = dataset.file_list[:args.max_eval_files]
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=utterance_collate)

    rows = []
    for attack in args.attacks:
        y_true = []
        y_pred = []
        adv_scores = []
        speaker_fooled = 0
        speaker_total = 0

        pbar = tqdm(loader, desc=f"eval {detector_train_domain}->{test_domain.name} {attack}", unit="utt")
        for audios, labels, _ in pbar:
            audio = audios[0].to(device)
            label = int(labels[0].item())
            starts = frame_starts(audio.numel(), args.frame_dim, args.eval_wshift, args.eval_max_frames)
            clean_frames = extract_frames(audio, starts, args.frame_dim)
            frame_labels = torch.full((clean_frames.size(0),), label, dtype=torch.long, device=device)

            with torch.enable_grad():
                adv_frames = make_adv_frames(attack, clean_frames, frame_labels,
                                             speaker_model, uapgn_model, args, device)

            with torch.no_grad():
                clean_utt_logits, _ = aggregate_logits(detector, clean_frames)
                adv_utt_logits, _ = aggregate_logits(detector, adv_frames)
                clean_prob = torch.softmax(clean_utt_logits, dim=1)[0]
                adv_prob = torch.softmax(adv_utt_logits, dim=1)[0]

                y_true.extend([0, 1])
                y_pred.extend([
                    int(clean_prob.argmax().item() == ADV_LABEL),
                    int(adv_prob.argmax().item() == ADV_LABEL),
                ])
                adv_scores.extend([
                    float(clean_prob[ADV_LABEL].item()),
                    float(adv_prob[ADV_LABEL].item()),
                ])

                if speaker_model is not None:
                    speaker_prob = torch.softmax(speaker_model(adv_frames), dim=1).mean(dim=0)
                    speaker_fooled += int(speaker_prob.argmax().item() != label)
                    speaker_total += 1

        metrics = binary_metrics(
            np.asarray(y_true, dtype=np.int32),
            np.asarray(adv_scores, dtype=np.float64),
            np.asarray(y_pred, dtype=np.int32),
        )
        row = {
            "detector_training": detector_train_domain,
            "detector_test": test_domain.name,
            "attack": attack,
            "speaker_fooling_rate": speaker_fooled / speaker_total if speaker_total else float("nan"),
            "num_eval_utterances": len(dataset),
            **metrics,
        }
        rows.append(row)
        print(
            f"{detector_train_domain}->{test_domain.name} {attack}: "
            f"Acc={row['accuracy']:.4f}, P={row['precision']:.4f}, "
            f"R={row['recall']:.4f}, F1={row['f1']:.4f}, AUROC={row['auroc']:.4f}, "
            f"FRD={row['frd']:.4f}, SpeakerFool={row['speaker_fooling_rate']:.4f}"
        )
    return rows


def write_results(rows: List[Dict], args):
    os.makedirs(args.output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output_dir, f"cross_dataset_detector_baseline_utterance_{ts}.csv")
    md_path = os.path.join(args.output_dir, f"cross_dataset_detector_baseline_utterance_{ts}.md")
    columns = [
        "detector_training",
        "detector_test",
        "attack",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auroc",
        "frd",
        "adv_detection_rate",
        "clean_benign_rate",
        "speaker_fooling_rate",
        "num_eval_utterances",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| Detector training | Detector test | Attack | Accuracy | Precision | Recall | F1 | AUROC | FRD | Speaker fooling |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['detector_training']} | {row['detector_test']} | {row['attack']} | "
                f"{row['accuracy']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | "
                f"{row['f1']:.4f} | {row['auroc']:.4f} | {row['frd']:.4f} | "
                f"{row['speaker_fooling_rate']:.4f} |\n"
            )
    print(f"Saved CSV: {csv_path}")
    print(f"Saved Markdown table: {md_path}")


def parse_args():
    parser = argparse.ArgumentParser("Utterance-level cross-dataset detector baseline")
    parser.add_argument("--train_domain", choices=["all", "timit", "libri"], default="all")
    parser.add_argument("--timit_data_root", type=str, required=True)
    parser.add_argument("--libri_data_root", type=str, required=True)
    parser.add_argument("--timit_speaker_model", type=str, default=None)
    parser.add_argument("--timit_speaker_cfg", type=str, default=None)
    parser.add_argument("--libri_speaker_model", type=str, default=None)
    parser.add_argument("--libri_speaker_cfg", type=str, default=None)
    parser.add_argument("--timit_uapgn_ckpt", type=str, default=None)
    parser.add_argument("--libri_uapgn_ckpt", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./output/cross_dataset_detector_baseline_utterance")

    parser.add_argument("--attacks", nargs="+", choices=["bim", "pgd", "gaussian", "uapgn"],
                        default=["bim", "pgd", "gaussian"])
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--attack_alpha", type=float, default=0.002)
    parser.add_argument("--attack_steps", type=int, default=10)
    parser.add_argument("--noise_dim", type=int, default=100)
    parser.add_argument("--frame_dim", type=int, default=3200)
    parser.add_argument("--train_wshift", type=int, default=3200)
    parser.add_argument("--eval_wshift", type=int, default=3200)
    parser.add_argument("--train_max_frames", type=int, default=12)
    parser.add_argument("--eval_max_frames", type=int, default=0,
                        help="0 means use all sliding-window frames during evaluation.")
    parser.add_argument("--max_eval_files", type=int, default=0,
                        help="0 means evaluate all test utterances.")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device(args.device)

    timit = make_domain("timit", args)
    libri = make_domain("libri", args)
    domains = {"timit": timit, "libri": libri}
    train_domains = ["libri", "timit"] if args.train_domain == "all" else [args.train_domain]

    all_rows = []
    for train_name in train_domains:
        detector, _ = train_detector_utterance(domains[train_name], args, device)
        for test_name in [train_name, "timit" if train_name == "libri" else "libri"]:
            rows = evaluate_detector_utterance(detector, train_name, domains[test_name], args, device)
            all_rows.extend(rows)

    write_results(all_rows, args)


if __name__ == "__main__":
    main()
