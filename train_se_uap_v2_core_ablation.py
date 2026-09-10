#!/usr/bin/env python3
"""
Train and evaluate the SE-UAP v2 core 2x2 ablation.

The two controlled factors are:
    1. UAP family: fixed (one learned vector) vs distribution (Generator1D(z)).
    2. Detector loss: off (L_R only) vs on (L_R + alpha * L_D).

All four conditions keep the data split, perturbation constraint, random phase,
recognition objective, checkpoint selection, and evaluation protocol unchanged.
The detector is always evaluated; --detector_loss only controls whether L_D is
part of the training objective.

Follow-up controls isolate two optimization changes for Distribution / L_D on:
    - --detector_aggregation mean_logit aligns training with utterance evaluation.
    - --utterance_grad_clip enables gradient clipping only in utterance training.

This script is a drop-in replacement for train_se_uap_main_experiment.py. It keeps the
manuscript's theoretical framework intact:

    z ~ N(0, I)
    delta_tilde = G_theta(z)                 (3200 samples, 0.2 s base UAP)
    delta_base  = Pi_Delta(H @ delta_tilde)  (H: FIXED spectral shaper; Delta: L2 ball, radius tau)
    delta_L     = roll(tile(delta_base), s)  (random phase shift s)
    x_adv       = clip(x + delta_L, -1, 1)
    L           = L_R + alpha * L_D
    L_R         = max(0, s_y - max_{i!=y} s_i + kappa)   (margin kappa >= 0)
    L_D         = p_b(x_adv)^(-beta) - 1                  (UNCHANGED, Prop. 1 safe)

Why Prop. 1 and Prop. 2 still hold:
    - Prop. 1 depends only on the form of L_D and a frozen detector; both are untouched.
    - Prop. 2 requires (i) a fixed input-independent feasible set Delta, (ii) a generator
      family containing constant mappings, (iii) objective R(theta) = E_z[F(delta_theta(z))].
      H is a fixed linear map (part of the generator family), tau is fixed within each
      curriculum stage, and the random phase shift is absorbed into the sampling
      distribution. Constant generators remain representable, so the degenerate fixed-UAP
      special case is preserved.

New features relative to the manuscript baseline script:
    1. Tau curriculum: --tau_start anneals to --tau (final budget) over
       --tau_anneal_epochs epochs (--tau_schedule {exp, linear, const}).
    2. Margin kappa in the recognition loss (default 2.0 instead of 0.0).
    3. Random phase shift when tiling the base UAP to utterance length
       (disable with --no_random_phase).
    4. Fixed spectral shaping H estimated offline from the average speech spectrum
       (--mode precompute_shaping, then --shaping_npy path/to/shaping.npy).
    5. Training-efficiency defaults: batch_size 64, num_z 4, train_max_frames 8,
       random frame offsets during training, linear LR decay --lr -> --lr_final.
    6. Optional frame-consistency recognition loss (--frame_loss_weight) that keeps
       the utterance-level objective primary while supervising difficult frames.

Main-experiment detector setting (unchanged):
    - Attacking TIMIT uses the detector pre-trained on LibriSpeech.
    - Attacking LibriSpeech uses the detector pre-trained on TIMIT.
"""

import argparse
import csv
import math
import os
import random
import time
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter
from tqdm import tqdm

from adversarial_utils import load_detector_model, load_speaker_model, read_list
from audio_utils import load_audio, repeat_to_length
from common.dataset import TIMITDetectorDataset, LibriSpeechDetectorDataset
from common.trainer import load_checkpoint, save_checkpoint
from common.utils import PESQ, SNR
from generator import Generator1D


BENIGN_LABEL = 1


class FixedUAP(nn.Module):
    """A single learned base perturbation with the same interface as Generator1D."""

    def __init__(self, frame_dim: int):
        super().__init__()
        # Generator1D starts with a zero-initialized output layer, so zero is the
        # matched initialization for the fixed-UAP condition.
        self.delta = nn.Parameter(torch.zeros(frame_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.delta.unsqueeze(0).expand(z.size(0), -1)


def build_uap_model(args, device):
    if args.uap_family == "fixed":
        return FixedUAP(args.frame_dim).to(device)
    return Generator1D(args.noise_dim, args.frame_dim).to(device)


def ablation_tag(args) -> str:
    clip_tag = "ugradclip" if args.utterance_grad_clip else "no_ugradclip"
    return (
        f"{args.uap_family}_detector_{args.detector_loss}_"
        f"{args.detector_aggregation}_{clip_tag}"
    )


def add_boolean_argument(parser, name: str, default: bool, help_text: str):
    """Python 3.8-compatible replacement for argparse.BooleanOptionalAction.

    Adds --<name> / --no-<name> flags.
    """
    dest = name.lstrip("-").replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(name, dest=dest, action="store_true", help=help_text)
    group.add_argument(f"--no_{dest}", f"--no-{dest}", dest=dest, action="store_false",
                       help=f"Disable {name}.")
    parser.set_defaults(**{dest: default})


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_worker(worker_id: int, seed: int):
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


# --------------------------------------------------------------------------- #
# P0-2: tau curriculum
# --------------------------------------------------------------------------- #
def tau_at_epoch(args, epoch: int) -> float:
    """Current L2 budget for the base UAP under the curriculum schedule."""
    if args.tau_schedule == "const" or args.tau_start <= 0 or args.tau_start == args.tau:
        return float(args.tau)
    t = min((epoch + 1) / max(args.tau_anneal_epochs, 1), 1.0)
    if args.tau_schedule == "linear":
        return float(args.tau_start + (args.tau - args.tau_start) * t)
    # exponential (geometric) annealing, default
    return float(args.tau_start * (args.tau / args.tau_start) ** t)


# --------------------------------------------------------------------------- #
# P0-3: fixed spectral shaping H
# --------------------------------------------------------------------------- #
class SpectralShaper:
    """Fixed FFT-domain spectral shaper (part of the frozen generator pipeline).

    delta_shaped = irfft(rfft(delta) * w).  The weights w are non-negative,
    energy-neutral on average (mean(w^2) = 1), and estimated offline from the
    average speech magnitude spectrum, so perturbation energy is concentrated
    in bands where speech energy is high (better perceptual masking).
    """

    def __init__(self, weights: np.ndarray, frame_dim: int):
        w = np.asarray(weights, dtype=np.float32).reshape(-1)
        expected = frame_dim // 2 + 1
        if w.size != expected:
            raise ValueError(f"Shaping weights length {w.size} != rfft length {expected}")
        self.frame_dim = frame_dim
        self.weights_np = w
        self._w_torch = None

    def to(self, device):
        self._w_torch = torch.from_numpy(self.weights_np).to(device)
        return self

    @property
    def w_torch(self):
        if self._w_torch is None:
            self._w_torch = torch.from_numpy(self.weights_np)
        return self._w_torch

    def apply_torch(self, delta: torch.Tensor) -> torch.Tensor:
        """delta: (..., frame_dim) real tensor."""
        w = self.w_torch.to(delta.device)
        spec = torch.fft.rfft(delta, dim=-1)
        return torch.fft.irfft(spec * w, n=delta.size(-1), dim=-1)

    def apply_np(self, delta: np.ndarray) -> np.ndarray:
        spec = np.fft.rfft(delta)
        return np.fft.irfft(spec * self.weights_np, n=delta.shape[-1]).astype(np.float32)


def precompute_shaping_filter(file_list: List[str], data_folder: str, args) -> np.ndarray:
    """Estimate the fixed shaping weights from the average speech magnitude spectrum.

    s_f = smoothed(avg_speech_magnitude ** exponent), normalized so mean(s^2) = 1.
    A frequency floor prevents zero weights (which would delete gradient directions).
    """
    rng = random.Random(args.seed)
    files = list(file_list)
    rng.shuffle(files)
    if args.shaping_max_files > 0:
        files = files[: args.shaping_max_files]

    acc = np.zeros(args.frame_dim // 2 + 1, dtype=np.float64)
    n_frames = 0
    for fname in tqdm(files, desc="Estimating average speech spectrum", unit="file"):
        audio, _, _ = load_audio(os.path.join(data_folder, fname), expected_fs=16000, normalize=True)
        if len(audio) < args.frame_dim:
            continue
        for start in range(0, len(audio) - args.frame_dim + 1, args.frame_dim):
            frame = audio[start : start + args.frame_dim]
            acc += np.abs(np.fft.rfft(frame))
            n_frames += 1
    if n_frames == 0:
        raise ValueError("No usable frames found while estimating the shaping filter.")
    mag = acc / n_frames

    # Frequency floor at the 10th percentile to keep every band trainable.
    floor = np.percentile(mag, 10)
    mag = np.clip(mag, floor, None)
    w = mag ** args.shaping_exponent

    # Light smoothing to avoid per-bin noise.
    kernel = np.ones(9, dtype=np.float64) / 9.0
    w = np.convolve(np.pad(w, 4, mode="edge"), kernel, mode="valid")

    # Energy neutral on average: mean(w^2) = 1.
    w = w / max(np.sqrt(np.mean(w ** 2)), 1e-12)
    return w.astype(np.float32)


# --------------------------------------------------------------------------- #
# Constraint / loss primitives (L_D form unchanged, Prop. 1 safe)
# --------------------------------------------------------------------------- #
def project_l2(delta: torch.Tensor, tau: float, eps: float = 1e-12):
    flat = delta.view(delta.size(0), -1)
    norm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
    scale = torch.clamp(tau / norm, max=1.0)
    return (flat * scale).view_as(delta)


def recognition_attack_loss(logits: torch.Tensor, labels: torch.Tensor, kappa: float = 0.0):
    """Non-targeted logit-margin loss with confidence margin kappa (Eq. 8 + margin)."""
    true_logits = logits[torch.arange(logits.size(0), device=logits.device), labels]
    mask = torch.nn.functional.one_hot(labels, num_classes=logits.size(1)).bool()
    max_other = logits.masked_fill(mask, -1e9).max(dim=1).values
    return torch.clamp(true_logits - max_other + kappa, min=0.0).mean()


def detection_evasion_loss(detector_logits: torch.Tensor, beta: float, eps: float):
    """Detector-aware evasion loss from Eq. (6): p_b(x_adv)^(-beta) - 1. UNCHANGED."""
    benign_prob = torch.softmax(detector_logits, dim=1)[:, BENIGN_LABEL].clamp_min(eps)
    return (torch.exp(beta * (-torch.log(benign_prob))) - 1.0).mean()


def utterance_detection_evasion_loss(detector_logits: torch.Tensor, args):
    """Aggregate frame outputs before applying the utterance detector loss."""
    if args.detector_aggregation == "mean_logit":
        utterance_logits = detector_logits.mean(dim=0, keepdim=True)
        return detection_evasion_loss(
            utterance_logits, beta=args.beta, eps=args.detector_eps
        )

    # Backward-compatible behavior: average benign probabilities first.
    benign_prob = (
        torch.softmax(detector_logits, dim=1)[:, BENIGN_LABEL]
        .mean()
        .clamp_min(args.detector_eps)
    )
    return torch.exp(args.beta * (-torch.log(benign_prob))) - 1.0


# --------------------------------------------------------------------------- #
# Framing helpers
# --------------------------------------------------------------------------- #
def frame_starts(length: int, wlen: int, wshift: int, max_frames: int = None, offset: int = 0):
    if length <= wlen:
        return [0]
    starts = list(range(offset, length - wlen + 1, wshift))
    if not starts or starts[0] != 0 and offset == 0:
        starts.insert(0, 0)
    if starts[-1] != length - wlen:
        starts.append(length - wlen)
    if max_frames and len(starts) > max_frames:
        idx = np.random.choice(len(starts), size=max_frames, replace=False)
        starts = sorted(starts[i] for i in idx)
    return starts


def extract_frames(wav: torch.Tensor, starts: List[int], wlen: int):
    frames = []
    length = wav.numel()
    for start in starts:
        if length <= wlen:
            frame = torch.zeros(wlen, device=wav.device, dtype=wav.dtype)
            frame[:length] = wav
        else:
            frame = wav[start : start + wlen]
        frames.append(frame)
    return torch.stack(frames, dim=0)


@torch.no_grad()
def sentence_test_logits(model, wav_data: torch.Tensor, wlen: int = 3200,
                         wshift: int = 160, batch_size: int = 128):
    """Sentence-level prediction using the same complete framing policy as training."""
    wav_data = wav_data.squeeze()
    starts = frame_starts(wav_data.numel(), wlen, wshift, max_frames=None, offset=0)
    frames = extract_frames(wav_data, starts, wlen)
    pred_all = []
    for begin in range(0, frames.size(0), batch_size):
        pred_all.append(model(frames[begin:begin + batch_size]))
    logits = torch.cat(pred_all, dim=0)
    return logits.mean(dim=0).argmax()


# --------------------------------------------------------------------------- #
# Perturbation construction: shaping -> projection -> tiling -> random phase
# --------------------------------------------------------------------------- #
def repeat_delta_to_utterance(delta: torch.Tensor, length: int):
    repeats = int(np.ceil(length / delta.numel()))
    return delta.repeat(repeats)[:length]


def _roll_base(base: torch.Tensor, random_phase: bool) -> torch.Tensor:
    if random_phase and base.numel() > 1:
        shift = int(torch.randint(0, base.numel(), (1,)).item())
        base = torch.roll(base, shifts=shift)
    return base


def _roll_base_np(base: np.ndarray, random_phase: bool) -> np.ndarray:
    if random_phase and base.size > 1:
        shift = int(np.random.randint(0, base.size))
        base = np.roll(base, shift)
    return base


def build_frame_delta(raw_delta: torch.Tensor, args, tau: float = None,
                      shaper: Optional[SpectralShaper] = None):
    """Project a 0.2 s perturbation for legacy frame-level training (with shaping)."""
    tau = args.tau if tau is None else tau
    if shaper is not None:
        raw_delta = shaper.apply_torch(raw_delta)
    return project_l2(raw_delta, tau)


def build_utterance_delta(raw_delta: torch.Tensor, length: int, args, tau: float = None,
                          shaper: Optional[SpectralShaper] = None):
    """Build the actual perturbation added to an utterance.

    Order: (optional fixed spectral shaping) -> L2 projection (radius tau) ->
    repeat/truncate to utterance length -> random phase roll of the base UAP.
    """
    tau = args.tau if tau is None else tau
    if raw_delta.dim() == 2:
        raw_delta = raw_delta.squeeze(0)
    if shaper is not None:
        raw_delta = shaper.apply_torch(raw_delta)
    if args.constraint_scope == "base":
        base_delta = project_l2(raw_delta.view(1, -1), tau).squeeze(0)
        base_delta = _roll_base(base_delta, args.random_phase)
        return repeat_delta_to_utterance(base_delta, length)
    delta_full = repeat_delta_to_utterance(raw_delta, length)
    delta_full = project_l2(delta_full.view(1, -1), tau).squeeze(0)
    return _roll_base(delta_full, args.random_phase)


def build_utterance_delta_np(raw_delta: np.ndarray, length: int, args, tau: float = None,
                             shaper: Optional[SpectralShaper] = None):
    tau = args.tau if tau is None else tau
    raw_delta = np.asarray(raw_delta, dtype=np.float32).reshape(-1)
    if shaper is not None:
        raw_delta = shaper.apply_np(raw_delta)
    if args.constraint_scope == "base":
        norm = max(float(np.linalg.norm(raw_delta, ord=2)), 1e-12)
        base_delta = raw_delta * min(tau / norm, 1.0)
        base_delta = _roll_base_np(base_delta, args.random_phase)
        return repeat_to_length(base_delta, length)
    delta_full = repeat_to_length(raw_delta, length)
    norm = max(float(np.linalg.norm(delta_full, ord=2)), 1e-12)
    delta_full = delta_full * min(tau / norm, 1.0)
    return _roll_base_np(delta_full, args.random_phase)


def utterance_frame_logits(model, adv_audio: torch.Tensor, args):
    max_frames = None if args.train_max_frames <= 0 else args.train_max_frames
    offset = 0
    if args.frame_random_offset and args.train_wshift > 1:
        offset = int(torch.randint(0, args.train_wshift, (1,)).item())
    starts = frame_starts(adv_audio.numel(), args.frame_dim, args.train_wshift,
                          max_frames, offset=offset)
    frames = extract_frames(adv_audio, starts, args.frame_dim)
    return model(frames)


def utterance_attack_loss(model, adv_audio: torch.Tensor, label: torch.Tensor,
                          args, use_detector_loss: bool = False):
    logits = utterance_frame_logits(model, adv_audio, args)

    if use_detector_loss:
        return utterance_detection_evasion_loss(logits, args), logits

    utterance_logits = logits.mean(dim=0, keepdim=True)
    loss_utt = recognition_attack_loss(utterance_logits, label.view(1), args.kappa)
    if args.frame_loss_weight <= 0:
        return loss_utt, logits

    frame_labels = label.reshape(1).expand(logits.size(0))
    loss_frame = recognition_attack_loss(logits, frame_labels, args.kappa)
    return loss_utt + args.frame_loss_weight * loss_frame, logits


class RunningAverage:
    def __init__(self):
        self.sums = {}
        self.counts = {}

    def update(self, values: Dict[str, float], counts: Dict[str, int]):
        for key, value in values.items():
            self.sums[key] = self.sums.get(key, 0.0) + float(value) * counts[key]
            self.counts[key] = self.counts.get(key, 0) + counts[key]

    def average(self):
        return {key: self.sums[key] / max(self.counts[key], 1) for key in self.sums}


# --------------------------------------------------------------------------- #
# Datasets / paths (unchanged)
# --------------------------------------------------------------------------- #
def dataset_paths(dataset: str, data_root: str):
    if dataset == "timit":
        return {
            "data_folder": data_root,
            "train_scp": os.path.join(data_root, "processed", "train_8_2.scp"),
            "test_scp": os.path.join(data_root, "processed", "test_8_2.scp"),
            "label_dict": os.path.join(data_root, "processed", "TIMIT_labels.npy"),
        }
    if dataset == "libri":
        split_dir = os.path.join(data_root, "Librispeech_spkid_sel", "split_spk_list")
        return {
            "data_folder": os.path.join(data_root, "Librispeech_spkid_sel"),
            "train_scp": os.path.join(split_dir, "libri_tr_8_2.scp"),
            "test_scp": os.path.join(split_dir, "libri_te_8_2.scp"),
            "label_dict": os.path.join(split_dir, "libri_dict.npy"),
        }
    raise ValueError(f"Unsupported dataset: {dataset}")


def build_train_dataset(dataset: str, data_root: str, wlen: int, frame_mode: str):
    if dataset == "timit":
        return TIMITDetectorDataset(data_root, train=True, wlen=wlen, frame_mode=frame_mode)
    if dataset == "libri":
        return LibriSpeechDetectorDataset(data_root, train=True, wlen=wlen, frame_mode=frame_mode)
    raise ValueError(f"Unsupported dataset: {dataset}")


class SEUAPUtteranceDataset(Dataset):
    """Full-utterance dataset for utterance-aware SE-UAP training."""

    def __init__(self, file_list: List[str], data_folder: str, label_dict: Dict, fs: int = 16000):
        self.file_list = file_list
        self.data_folder = data_folder
        self.label_dict = label_dict
        self.fs = fs

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        fname = self.file_list[index]
        audio, _, norm_factor = load_audio(os.path.join(self.data_folder, fname),
                                           expected_fs=self.fs, normalize=True)
        return torch.from_numpy(audio).float(), int(self.label_dict[fname]), norm_factor


def utterance_collate(batch):
    audios, labels, norm_factors = zip(*batch)
    return list(audios), torch.tensor(labels).long(), torch.tensor(norm_factors).float()


def resolve_detector_path(args):
    if args.detector_model:
        return args.detector_model
    if args.dataset == "timit":
        if not args.libri_pretrained_detector:
            raise ValueError("TIMIT main experiment requires --libri_pretrained_detector")
        return args.libri_pretrained_detector
    if args.dataset == "libri":
        if not args.timit_pretrained_detector:
            raise ValueError("LibriSpeech main experiment requires --timit_pretrained_detector")
        return args.timit_pretrained_detector
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def detector_checkpoint_metadata(detector_path: str):
    state = torch.load(detector_path, map_location="cpu")
    if not isinstance(state, dict):
        return {
            "detector_type": "cnn",
            "training_level": "unknown",
            "detector_train_domain": "unknown",
            "detector_attacks": "unknown",
        }
    return {
        "detector_type": state.get("detector_type", "cnn"),
        "training_level": state.get("training_level", "frame"),
        "detector_train_domain": state.get("train_domain", "unknown"),
        "detector_attacks": ",".join(state.get("attacks", [])),
    }


# --------------------------------------------------------------------------- #
# Training loops
# --------------------------------------------------------------------------- #
def train_one_epoch(generator, speaker_model, detector, loader, optimizer, args, device,
                    epoch, tau, shaper):
    generator.train()
    speaker_model.eval()
    detector.eval()

    avg = RunningAverage()
    pbar = tqdm(loader, desc=f"SE-UAP v2 train epoch {epoch + 1}/{args.epochs}", unit="batch")
    for clean, speaker_id, _ in pbar:
        clean = clean.float().to(device)
        speaker_id = speaker_id.long().to(device)
        batch_size = clean.size(0)

        loss_r_items = []
        loss_d_items = []
        fr_items = []
        frd_items = []
        l2_items = []
        linf_items = []

        for _ in range(args.num_z):
            z = torch.randn(1, args.noise_dim, device=device)
            delta = build_frame_delta(generator(z), args, tau=tau, shaper=shaper).expand_as(clean)
            adv = torch.clamp(clean + delta, -1.0, 1.0)

            speaker_logits = speaker_model(adv)
            loss_r_items.append(recognition_attack_loss(speaker_logits, speaker_id, args.kappa))
            if args.detector_loss == "on":
                detector_logits = detector(adv)
                loss_d_items.append(
                    detection_evasion_loss(
                        detector_logits, beta=args.beta, eps=args.detector_eps
                    )
                )
            else:
                with torch.no_grad():
                    detector_logits = detector(adv)
                loss_d_items.append(torch.zeros((), device=device, dtype=adv.dtype))
            with torch.no_grad():
                pred_speaker = speaker_logits.argmax(dim=1)
                fr_items.append((pred_speaker != speaker_id).float().mean().item())
                pred_detector = detector_logits.argmax(dim=1)
                frd_items.append((pred_detector == BENIGN_LABEL).float().mean().item())
                diff = adv - clean
                l2_items.append(diff.view(batch_size, -1).norm(p=2, dim=1).mean().item())
                linf_items.append(diff.abs().amax(dim=1).mean().item())

        loss_r = torch.stack(loss_r_items).mean()
        loss_d = torch.stack(loss_d_items).mean()
        loss = loss_r + args.effective_alpha * loss_d

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            generator.parameters(),
            max_norm=args.grad_clip
        )

        optimizer.step()

        values = {
            "loss": loss.item(),
            "loss_r": loss_r.item(),
            "loss_d": loss_d.item(),
            "train_fr": float(np.mean(fr_items)),
            "train_frd": float(np.mean(frd_items)),
            "delta_l2": float(np.mean(l2_items)),
            "delta_linf": float(np.mean(linf_items)),
            "tau": tau,
        }
        avg.update(values, {key: batch_size for key in values})
        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         fr=f"{values['train_fr']:.3f}",
                         frd=f"{values['train_frd']:.3f}",
                         tau=f"{tau:.3f}")

    return avg.average()


def train_one_epoch_utterance(generator, speaker_model, detector, loader, optimizer, args,
                              device, epoch, tau, shaper):
    generator.train()
    speaker_model.eval()
    detector.eval()

    avg = RunningAverage()
    pbar = tqdm(loader, desc=f"SE-UAP v2 utterance train epoch {epoch + 1}/{args.epochs}", unit="batch")
    for clean_list, speaker_ids, _ in pbar:
        speaker_ids = speaker_ids.long().to(device)
        batch_size = len(clean_list)

        loss_r_items = []
        loss_d_items = []
        fr_items = []
        frd_items = []
        l2_items = []
        linf_items = []

        for _ in range(args.num_z):
            z = torch.randn(1, args.noise_dim, device=device)
            raw_delta = generator(z).squeeze(0)

            for clean_cpu, speaker_id in zip(clean_list, speaker_ids):
                clean = clean_cpu.to(device)
                delta_full = build_utterance_delta(raw_delta, clean.numel(), args,
                                                   tau=tau, shaper=shaper)
                adv = torch.clamp(clean + delta_full, -1.0, 1.0)

                loss_r_i, speaker_logits = utterance_attack_loss(
                    speaker_model, adv, speaker_id, args, use_detector_loss=False
                )
                if args.detector_loss == "on":
                    loss_d_i, detector_logits = utterance_attack_loss(
                        detector, adv, speaker_id, args, use_detector_loss=True
                    )
                else:
                    with torch.no_grad():
                        detector_logits = utterance_frame_logits(detector, adv, args)
                    loss_d_i = torch.zeros((), device=device, dtype=adv.dtype)
                loss_r_items.append(loss_r_i)
                loss_d_items.append(loss_d_i)
                with torch.no_grad():
                    speaker_utt_logits = speaker_logits.mean(dim=0)
                    detector_utt_logits = detector_logits.mean(dim=0)
                    fr_items.append(float((speaker_utt_logits.argmax() != speaker_id).item()))
                    frd_items.append(float(detector_utt_logits.argmax().item() == BENIGN_LABEL))
                    actual_delta = adv - clean
                    l2_items.append(actual_delta.norm(p=2).item())
                    linf_items.append(actual_delta.abs().max().item())

        loss_r = torch.stack(loss_r_items).mean()
        loss_d = torch.stack(loss_d_items).mean()
        loss = loss_r + args.effective_alpha * loss_d

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = None
        if args.utterance_grad_clip:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                generator.parameters(),
                max_norm=args.grad_clip,
            )
        optimizer.step()

        values = {
            "loss": loss.item(),
            "loss_r": loss_r.item(),
            "loss_d": loss_d.item(),
            "train_fr": float(np.mean(fr_items)),
            "train_frd": float(np.mean(frd_items)),
            "delta_l2": float(np.mean(l2_items)),
            "delta_linf": float(np.mean(linf_items)),
            "tau": tau,
        }
        if grad_norm is not None:
            # clip_grad_norm_ returns the total norm before clipping.
            values["grad_norm"] = float(grad_norm.item())
        avg.update(values, {key: batch_size * args.num_z for key in values})
        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         fr=f"{values['train_fr']:.3f}",
                         frd=f"{values['train_frd']:.3f}",
                         tau=f"{tau:.3f}")

    return avg.average()


def precompute_uap_pool(generator, args, device):
    generator.eval()
    pool = []
    torch_gen = torch.Generator(device=device).manual_seed(args.seed)
    with torch.no_grad():
        for _ in range(args.pool_size):
            z = torch.randn(1, args.noise_dim, device=device, generator=torch_gen)
            raw_delta = generator(z)
            pool.append(raw_delta.squeeze(0).detach().cpu().numpy())
    return pool


def choose_uap_delta(uap_pool: List[np.ndarray], length: int, args,
                     shaper: Optional[SpectralShaper] = None, tau: float = None):
    raw_delta = random.choice(uap_pool)
    # Evaluation uses the final tau budget unless an explicit override is given.
    return build_utterance_delta_np(raw_delta, length, args,
                                    tau=args.tau if tau is None else tau, shaper=shaper)


def random_clip_np(audio: np.ndarray, clip_len: int, rng: np.random.RandomState) -> np.ndarray:
    """Randomly crop a fixed-length clip (zero-pad if shorter), matching the
    manuscript protocol of evaluating on randomly sampled short clips."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size <= clip_len:
        out = np.zeros(clip_len, dtype=np.float32)
        out[: audio.size] = audio
        return out
    start = int(rng.randint(0, audio.size - clip_len + 1))
    return audio[start : start + clip_len]


def select_uap_pool(generator, speaker_model, detector, file_list, data_folder,
                    label_dict, args, device, shaper):
    """Score --pool_candidates candidate UAPs on validation utterances and keep the
    best --pool_size by FR. Detector metrics are deliberately excluded so the
    detector-loss-off cells remain detector-blind during model selection."""
    generator.eval()
    torch_gen = torch.Generator(device=device).manual_seed(args.seed + 777)
    candidates = []
    with torch.no_grad():
        for _ in range(args.pool_candidates):
            z = torch.randn(1, args.noise_dim, device=device, generator=torch_gen)
            candidates.append(generator(z).squeeze(0).detach().cpu().numpy())

    files = list(file_list)
    if args.pool_select_max_files > 0:
        files = files[: args.pool_select_max_files]
    audios = []
    for fname in files:
        audio, _, _ = load_audio(os.path.join(data_folder, fname), expected_fs=16000, normalize=True)
        audios.append((audio, label_dict[fname]))

    scores = []
    for ci, raw_delta in enumerate(candidates):
        fr_hits, frd_hits = 0, 0
        for audio, label in audios:
            delta_full = build_utterance_delta_np(raw_delta, len(audio), args,
                                                  tau=args.tau, shaper=shaper)
            adv = np.clip(audio + delta_full, -1.0, 1.0)
            adv_tensor = torch.from_numpy(adv).float().to(device)
            with torch.no_grad():
                pred_speaker = sentence_test_logits(speaker_model, adv_tensor, wlen=args.frame_dim,
                                                    wshift=args.eval_wshift,
                                                    batch_size=args.eval_batch_size)
                pred_detector = sentence_test_logits(detector, adv_tensor, wlen=args.frame_dim,
                                                     wshift=args.eval_wshift,
                                                     batch_size=args.eval_batch_size)
            if torch.is_tensor(pred_speaker):
                pred_speaker = pred_speaker.item()
            if torch.is_tensor(pred_detector):
                pred_detector = pred_detector.item()
            fr_hits += int(pred_speaker != label)
            frd_hits += int(pred_detector == BENIGN_LABEL)
        n = max(len(audios), 1)
        score = fr_hits / n
        scores.append(score)
        if (ci + 1) % 10 == 0:
            print(f"  pool candidate {ci + 1}/{len(candidates)}: running best score={max(scores):.4f}")

    order = np.argsort(scores)[::-1][: args.pool_size]
    pool = [candidates[i] for i in order]
    print(f"Selected pool candidates {sorted(order.tolist())} "
          f"with scores {[round(scores[i], 4) for i in order]}")
    return pool


def split_train_validation(file_list: List[str], val_fraction: float, seed: int) -> Tuple[List[str], List[str]]:
    if val_fraction <= 0.0:
        return file_list, []
    indices = list(range(len(file_list)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = max(1, int(round(len(indices) * val_fraction)))
    val_indices = set(indices[:val_size])
    train_files = [fname for idx, fname in enumerate(file_list) if idx not in val_indices]
    val_files = [fname for idx, fname in enumerate(file_list) if idx in val_indices]
    return train_files, val_files


def validate(generator, speaker_model, detector, file_list: List[str], data_folder: str,
             label_dict: Dict, args, device, epoch: int,
             shaper: Optional[SpectralShaper] = None):
    generator.eval()
    speaker_model.eval()
    detector.eval()

    eval_files = file_list
    if args.val_max_files > 0:
        eval_files = file_list[: args.val_max_files]

    uap_pool = precompute_uap_pool(generator, args, device)
    metrics = RunningAverage()
    pbar = tqdm(eval_files, desc=f"SE-UAP v2 val epoch {epoch + 1}", unit="file", leave=False)
    for fname in pbar:
        audio, _, _ = load_audio(os.path.join(data_folder, fname), expected_fs=16000, normalize=True)
        label = label_dict[fname]
        delta_full = choose_uap_delta(uap_pool, len(audio), args, shaper=shaper)
        adv = np.clip(audio + delta_full, -1.0, 1.0)
        snr = SNR(adv, audio)

        adv_tensor = torch.from_numpy(adv).float().to(device)
        with torch.no_grad():
            pred_speaker = sentence_test_logits(speaker_model, adv_tensor, wlen=args.frame_dim,
                                                wshift=args.eval_wshift,
                                                batch_size=args.eval_batch_size)
            pred_detector = sentence_test_logits(detector, adv_tensor, wlen=args.frame_dim,
                                                 wshift=args.eval_wshift,
                                                 batch_size=args.eval_batch_size)
        if torch.is_tensor(pred_speaker):
            pred_speaker = pred_speaker.item()
        if torch.is_tensor(pred_detector):
            pred_detector = pred_detector.item()

        values = {
            "val_fr": float(pred_speaker != label),
            "val_frd": float(pred_detector == BENIGN_LABEL),
            "val_snr": snr,
        }
        metrics.update(values, {key: 1 for key in values})
        pbar.set_postfix(fr=f"{values['val_fr']:.0f}",
                         frd=f"{values['val_frd']:.0f}",
                         snr=f"{snr:.2f}")

    if not eval_files:
        return {}
    out = metrics.average()
    # Use the same recognition-only checkpoint criterion in all four cells.
    # FRD remains an evaluation outcome and cannot leak into model selection.
    out["val_score"] = out["val_fr"]
    return out


def evaluate(generator, speaker_model, detector, file_list: List[str], data_folder: str,
             label_dict: Dict, args, device, shaper: Optional[SpectralShaper] = None,
             pool: Optional[List[np.ndarray]] = None, tau_override: float = None):
    generator.eval()
    speaker_model.eval()
    detector.eval()
    uap_pool = pool if pool is not None else precompute_uap_pool(generator, args, device)
    metrics = RunningAverage()
    clip_rng = np.random.RandomState(args.seed)

    pbar = tqdm(file_list, desc=f"SE-UAP v2 eval {args.dataset}", unit="file")
    for fname in pbar:
        audio, fs, _ = load_audio(os.path.join(data_folder, fname), expected_fs=16000, normalize=True)
        label = label_dict[fname]

        if args.eval_protocol == "clip":
            audio = random_clip_np(audio, args.clip_len, clip_rng)

        tic = time.perf_counter()
        delta_full = choose_uap_delta(uap_pool, len(audio), args, shaper=shaper, tau=tau_override)
        adv = np.clip(audio + delta_full, -1.0, 1.0)
        atc = time.perf_counter() - tic

        snr = SNR(adv, audio)
        try:
            pesq = PESQ(audio, adv, fs)
        except Exception as exc:
            print(f"Warning: PESQ failed for {fname}: {exc}")
            pesq = float("nan")

        adv_tensor = torch.from_numpy(adv).float().to(device)
        with torch.no_grad():
            pred_speaker = sentence_test_logits(speaker_model, adv_tensor, wlen=args.frame_dim,
                                                wshift=args.eval_wshift,
                                                batch_size=args.eval_batch_size)
            pred_detector = sentence_test_logits(detector, adv_tensor, wlen=args.frame_dim,
                                                 wshift=args.eval_wshift,
                                                 batch_size=args.eval_batch_size)

        if torch.is_tensor(pred_speaker):
            pred_speaker = pred_speaker.item()
        if torch.is_tensor(pred_detector):
            pred_detector = pred_detector.item()

        fr = float(pred_speaker != label)
        frd = float(pred_detector == BENIGN_LABEL)
        values = {
            "FR": fr,
            "FRD": frd,
            "SNR": snr,
            "PESQ": pesq,
            "ATC": atc,
        }
        metrics.update(values, {key: 1 for key in values})
        pbar.set_postfix(fr=f"{fr:.0f}", frd=f"{frd:.0f}", snr=f"{snr:.2f}", pesq=f"{pesq:.2f}")

    out = metrics.average()
    out["SEI"] = out.get("FRD", 0.0) / max(out.get("ATC", 0.0), 1e-12)
    return out


def write_eval_results(metrics: Dict, args, detector_meta: Dict):
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(
        args.output_dir,
        f"se_uap_v2_core_ablation_{ablation_tag(args)}_{args.dataset}_"
        f"{time.strftime('%Y%m%d_%H%M%S')}.csv",
    )
    columns = [
        "dataset",
        "uap_family",
        "detector_loss",
        "detector_aggregation",
        "utterance_grad_clip",
        "grad_clip",
        "effective_alpha",
        "detector_model",
        "detector_type",
        "detector_training_level",
        "detector_train_domain",
        "detector_attacks",
        "FR",
        "FRD",
        "SNR",
        "PESQ",
        "ATC",
        "SEI",
        "tau",
        "tau_start",
        "tau_schedule",
        "tau_anneal_epochs",
        "kappa",
        "alpha",
        "constraint_scope",
        "random_phase",
        "eval_protocol",
        "shaping_npy",
        "num_z",
        "train_max_frames",
        "frame_loss_weight",
        "frame_random_offset",
    ]
    row = {
        "dataset": args.dataset,
        "uap_family": args.uap_family,
        "detector_loss": args.detector_loss,
        "detector_aggregation": args.detector_aggregation,
        "utterance_grad_clip": args.utterance_grad_clip,
        "grad_clip": args.grad_clip,
        "effective_alpha": args.effective_alpha,
        "detector_model": resolve_detector_path(args),
        "detector_type": detector_meta["detector_type"],
        "detector_training_level": detector_meta["training_level"],
        "detector_train_domain": detector_meta["detector_train_domain"],
        "detector_attacks": detector_meta["detector_attacks"],
        "tau": args.tau,
        "tau_start": args.tau_start,
        "tau_schedule": args.tau_schedule,
        "tau_anneal_epochs": args.tau_anneal_epochs,
        "kappa": args.kappa,
        "alpha": args.alpha,
        "constraint_scope": args.constraint_scope,
        "random_phase": args.random_phase,
        "eval_protocol": args.eval_protocol,
        "shaping_npy": args.shaping_npy,
        "num_z": args.num_z,
        "train_max_frames": args.train_max_frames,
        "frame_loss_weight": args.frame_loss_weight,
        "frame_random_offset": args.frame_random_offset,
        **metrics,
    }
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in columns})
    print(f"Saved evaluation CSV: {csv_path}")


# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser("SE-UAP v2 core 2x2 ablation")
    parser.add_argument("--mode", choices=["train", "test", "precompute_shaping"], default="train")
    parser.add_argument("--dataset", choices=["timit", "libri"], required=True)
    parser.add_argument(
        "--uap_family",
        choices=["fixed", "distribution"],
        required=True,
        help="fixed learns one UAP vector; distribution learns Generator1D(z)",
    )
    parser.add_argument(
        "--detector_loss",
        choices=["off", "on"],
        required=True,
        help="whether alpha * L_D is included in the training objective",
    )
    parser.add_argument(
        "--detector_aggregation",
        choices=["mean_probability", "mean_logit"],
        default="mean_probability",
        help="aggregate detector frame outputs before the utterance detector loss",
    )
    add_boolean_argument(
        parser,
        "--utterance_grad_clip",
        default=False,
        help_text="apply --grad_clip to generator gradients in utterance training",
    )
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--speaker_model", type=str, default=None)
    parser.add_argument("--speaker_cfg", type=str, default=None)

    parser.add_argument("--detector_model", type=str, default=None,
                        help="Optional explicit detector path. Overrides cross-dataset detector selection.")
    parser.add_argument("--libri_pretrained_detector", type=str, default=None,
                        help="Detector pre-trained on LibriSpeech, used when --dataset timit.")
    parser.add_argument("--timit_pretrained_detector", type=str, default=None,
                        help="Detector pre-trained on TIMIT, used when --dataset libri.")

    parser.add_argument("--output_dir", type=str, default="./output/se_uap_v2_main")
    parser.add_argument("--test_ckpt", type=str, default=None)
    parser.add_argument("--noise_dim", type=int, default=100)
    parser.add_argument("--frame_dim", type=int, default=3200)
    parser.add_argument("--wlen", type=int, default=200, help="training frame length in milliseconds")
    parser.add_argument("--frame_mode", choices=["fixed", "random"], default="random")
    parser.add_argument("--train_objective", choices=["utterance", "frame"], default="utterance",
                        help="utterance matches sentence-level evaluation; frame keeps the legacy frame loss")
    parser.add_argument("--train_wshift", type=int, default=3200,
                        help="sliding-window shift in samples for utterance-aware training")
    parser.add_argument("--train_max_frames", type=int, default=8,
                        help="maximum frames sampled per utterance; 0 uses all frames (slow)")
    parser.add_argument("--frame_loss_weight", type=float, default=0.0,
                        help="weight of the auxiliary frame-level recognition margin loss; "
                             "0 preserves the utterance-only objective")
    add_boolean_argument(parser, "--frame_random_offset", default=True,
                         help_text="randomize frame start offsets during training")
    parser.add_argument("--pool_size", type=int, default=5)
    parser.add_argument("--pool_candidates", type=int, default=5,
                        help="candidate UAPs scored on validation FR; the best pool_size are kept. "
                             "Equal to pool_size disables selection (pure random pool).")
    parser.add_argument("--pool_select_max_files", type=int, default=100,
                        help="validation utterances used to score each pool candidate; 0 uses all")
    parser.add_argument("--pool_npy", type=str, default=None,
                        help="optional .npy file with a pre-selected UAP pool (test mode)")
    parser.add_argument("--eval_protocol", choices=["utterance", "clip"], default="utterance",
                        help="clip matches the manuscript protocol: classify a randomly cropped "
                             "clip of --clip_len samples; utterance classifies the full utterance")
    parser.add_argument("--clip_len", type=int, default=3200,
                        help="clip length in samples for --eval_protocol clip (3200 = 0.2 s)")

    # P0-2: tau curriculum. --tau is the FINAL budget; --tau_start the initial one.
    parser.add_argument("--tau", type=float, default=0.1, help="FINAL L2 perturbation bound (base UAP)")
    parser.add_argument("--tau_start", type=float, default=1.0,
                        help="initial L2 budget for the curriculum; equal to --tau disables annealing")
    parser.add_argument("--tau_schedule", choices=["exp", "linear", "const"], default="exp")
    parser.add_argument("--tau_anneal_epochs", type=int, default=20,
                        help="epochs over which tau anneals from tau_start to tau")
    parser.add_argument("--constraint_scope", choices=["base", "full_utterance"], default="base",
                        help="base constrains the fixed 0.2 s UAP before repeat/truncate adaptation")

    # P1-4: margin
    parser.add_argument("--kappa", type=float, default=2.0,
                        help="positive logit-margin confidence for the recognition attack loss")
    parser.add_argument("--num_z", type=int, default=4,
                        help="number of latent UAP samples averaged per batch")

    # P0-3: spectral shaping
    parser.add_argument("--shaping_npy", type=str, default=None,
                        help="path to fixed spectral shaping weights (.npy, length frame_dim//2+1)")
    parser.add_argument("--shaping_exponent", type=float, default=0.5,
                        help="exponent applied to the average speech magnitude spectrum")
    parser.add_argument("--shaping_max_files", type=int, default=2000,
                        help="max utterances used to estimate the shaping filter")

    # P1-6: random phase
    add_boolean_argument(parser, "--random_phase", default=True,
                         help_text="randomly roll the base UAP phase when tiling to utterance length")

    parser.add_argument("--alpha", type=float, default=1.0, help="detector evasion loss weight")
    parser.add_argument("--beta", type=float, default=1.0, help="detector evasion loss sharpness")
    parser.add_argument("--detector_eps", type=float, default=1e-8)
    parser.add_argument("--val_fraction", type=float, default=0.1,
                        help="fraction of training utterances reserved for validation; 0 disables validation")
    parser.add_argument("--val_max_files", type=int, default=0,
                        help="maximum validation utterances per epoch; 0 uses the full validation split")
    parser.add_argument("--val_frd_weight", type=float, default=0.25,
                        help="retained for CLI compatibility; ignored by the core ablation")
    parser.add_argument("--best_min_frd", type=float, default=0.0,
                        help="retained for CLI compatibility; ignored by the core ablation")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_final", type=float, default=2e-5,
                        help="final learning rate for the linear decay schedule")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--eval_wshift", type=int, default=3200,
                        help="sliding-window shift in samples for sentence-level evaluation")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=5.0
    )
    return parser.parse_args()


def set_epoch_lr(optimizer, args, epoch: int):
    """Linear LR decay from --lr to --lr_final across training (matches the manuscript)."""
    if args.epochs <= 1:
        return args.lr
    t = min(epoch / (args.epochs - 1), 1.0)
    lr = args.lr + (args.lr_final - args.lr) * t
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def load_shaper(args) -> Optional[SpectralShaper]:
    if not args.shaping_npy:
        return None
    weights = np.load(args.shaping_npy)
    return SpectralShaper(weights, args.frame_dim)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    if args.train_objective != "utterance":
        raise ValueError("Core 2x2 ablation requires --train_objective utterance")
    if args.frame_loss_weight != 0:
        raise ValueError("Core 2x2 ablation requires --frame_loss_weight 0")
    if args.num_z <= 0:
        raise ValueError("--num_z must be positive")
    if args.detector_loss == "on" and args.alpha <= 0:
        raise ValueError("--alpha must be positive when --detector_loss on")
    if args.utterance_grad_clip and args.grad_clip <= 0:
        raise ValueError("--grad_clip must be positive when --utterance_grad_clip is enabled")
    args.effective_alpha = args.alpha if args.detector_loss == "on" else 0.0
    seed_all(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    paths = dataset_paths(args.dataset, args.data_root)
    label_dict = np.load(paths["label_dict"], allow_pickle=True).item()

    # P0-3 utility mode: estimate the fixed spectral shaping filter and exit.
    if args.mode == "precompute_shaping":
        train_files = read_list(paths["train_scp"])
        weights = precompute_shaping_filter(train_files, paths["data_folder"], args)
        out_path = os.path.join(args.output_dir, f"shaping_{args.dataset}.npy")
        np.save(out_path, weights)
        print(f"Saved spectral shaping weights ({weights.shape[0]} bins): {out_path}")
        print(f"w stats: min={weights.min():.4f}, max={weights.max():.4f}, "
              f"mean(w^2)={np.mean(weights ** 2):.4f}")
        return

    if not args.speaker_model or not args.speaker_cfg:
        raise ValueError("--speaker_model and --speaker_cfg are required for train/test modes")

    detector_path = resolve_detector_path(args)
    detector_meta = detector_checkpoint_metadata(detector_path)
    print(f"Dataset: {args.dataset}")
    print(f"Using cross-dataset detector: {detector_path}")
    print(
        "Detector checkpoint: "
        f"type={detector_meta['detector_type']}, "
        f"training_level={detector_meta['training_level']}, "
        f"train_domain={detector_meta['detector_train_domain']}, "
        f"attacks={detector_meta['detector_attacks']}"
    )

    speaker_model = load_speaker_model(args.speaker_model, args.speaker_cfg, device)
    detector = load_detector_model(detector_path, device)
    for model in [speaker_model, detector]:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

    file_list = read_list(paths["test_scp"])

    generator = build_uap_model(args, device)
    trainable_parameters = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    print(
        f"Core ablation: uap_family={args.uap_family}, "
        f"detector_loss={args.detector_loss}, effective_alpha={args.effective_alpha}, "
        f"detector_aggregation={args.detector_aggregation}, "
        f"utterance_grad_clip={args.utterance_grad_clip}, grad_clip={args.grad_clip}, "
        f"trainable_parameters={trainable_parameters}"
    )

    shaper = load_shaper(args)
    if shaper is not None:
        shaper.to(device)
        print(f"Loaded spectral shaping filter: {args.shaping_npy}")

    # tau_sweep mode: evaluate ONE trained generator at several tau budgets without
    # retraining (the L2 projection is applied at deployment time, so the learned
    # perturbation directions can be rescaled post hoc). Produces the FR-SNR curve
    # used to pick the operating point.
    if args.mode == "tau_sweep":
        if not args.test_ckpt:
            raise ValueError("--test_ckpt is required in tau_sweep mode")
        load_checkpoint(generator, args.test_ckpt, map_location=device)
        taus = [float(t) for t in args.tau_list.split(",")]
        eval_files = file_list
        if args.tau_sweep_max_files > 0:
            eval_files = file_list[: args.tau_sweep_max_files]
        pool = precompute_uap_pool(generator, args, device)
        rows = []
        for tau in taus:
            m = evaluate(generator, speaker_model, detector, eval_files,
                         paths["data_folder"], label_dict, args, device,
                         shaper=shaper, pool=pool, tau_override=tau)
            rows.append({"tau": tau, **m})
            print(f"tau={tau}: FR={m['FR']:.4f}, FRD={m['FRD']:.4f}, "
                  f"SNR={m['SNR']:.2f}, PESQ={m.get('PESQ', float('nan')):.4f}")
        sweep_path = os.path.join(
            args.output_dir, f"tau_sweep_{args.dataset}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        with open(sweep_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["tau", "FR", "FRD", "SNR", "PESQ", "ATC", "SEI"])
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in writer.fieldnames})
        print(f"Saved tau sweep CSV: {sweep_path}")
        return

    print(
        f"P0/P1 config: tau {args.tau_start}->{args.tau} ({args.tau_schedule}, "
        f"{args.tau_anneal_epochs} epochs), kappa={args.kappa}, num_z={args.num_z}, "
        f"random_phase={args.random_phase}, train_max_frames={args.train_max_frames}, "
        f"frame_loss_weight={args.frame_loss_weight}, "
        f"frame_random_offset={args.frame_random_offset}, batch_size={args.batch_size}, "
        f"uap_family={args.uap_family}, detector_loss={args.detector_loss}, "
        f"effective_alpha={args.effective_alpha}, "
        f"detector_aggregation={args.detector_aggregation}, "
        f"utterance_grad_clip={args.utterance_grad_clip}, grad_clip={args.grad_clip}"
    )

    if args.mode == "train":
        val_files = []
        if args.train_objective == "utterance":
            all_train_files = read_list(paths["train_scp"])
            train_file_list, val_files = split_train_validation(all_train_files, args.val_fraction, args.seed)
            print(
                f"Train utterances: {len(train_file_list)}, "
                f"validation utterances: {len(val_files)}"
            )
            train_dataset = SEUAPUtteranceDataset(train_file_list, paths["data_folder"], label_dict)
            collate_fn = utterance_collate
        else:
            train_dataset = build_train_dataset(args.dataset, args.data_root, args.wlen, args.frame_mode)
            collate_fn = None
        loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=partial(init_worker, seed=args.seed),
            collate_fn=collate_fn,
        )

        optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.95, 0.999))
        writer = SummaryWriter(os.path.join(args.output_dir, f"run_{time.strftime('%Y%m%d_%H%M%S')}"))
        best_score = -float("inf")
        best_ckpt_path = os.path.join(
            args.output_dir,
            f"se_uap_v2_{ablation_tag(args)}_{args.dataset}_best.pth",
        )
        for epoch in range(args.epochs):
            tau_now = tau_at_epoch(args, epoch)
            lr_now = set_epoch_lr(optimizer, args, epoch)
            if args.train_objective == "utterance":
                train_metrics = train_one_epoch_utterance(
                    generator, speaker_model, detector, loader, optimizer, args, device,
                    epoch, tau_now, shaper
                )
            else:
                train_metrics = train_one_epoch(
                    generator, speaker_model, detector, loader, optimizer, args, device,
                    epoch, tau_now, shaper
                )
            train_metrics["lr"] = lr_now
            print(f"Epoch {epoch + 1}: {train_metrics}")
            for key, value in train_metrics.items():
                writer.add_scalar(f"train/{key}", value, epoch)

            if args.train_objective == "utterance" and val_files:
                val_metrics = validate(
                    generator, speaker_model, detector, val_files,
                    paths["data_folder"], label_dict, args, device, epoch, shaper=shaper
                )
                print(f"Validation epoch {epoch + 1}: {val_metrics}")
                for key, value in val_metrics.items():
                    writer.add_scalar(f"val/{key}", value, epoch)
                # Do not select best checkpoints while tau is still annealing above 2x target.
                selection_eligible = tau_now <= 2.0 * args.tau
                if selection_eligible and val_metrics.get("val_score", -float("inf")) > best_score:
                    best_score = val_metrics["val_score"]
                    save_checkpoint(
                        generator,
                        best_ckpt_path,
                        optimizer=optimizer,
                        meta={
                            "epoch": epoch + 1,
                            "val_metrics": val_metrics,
                            "tau": args.tau,
                            "tau_start": args.tau_start,
                            "tau_schedule": args.tau_schedule,
                            "kappa": args.kappa,
                            "alpha": args.alpha,
                            "effective_alpha": args.effective_alpha,
                            "uap_family": args.uap_family,
                            "detector_loss": args.detector_loss,
                            "detector_aggregation": args.detector_aggregation,
                            "utterance_grad_clip": args.utterance_grad_clip,
                            "grad_clip": args.grad_clip,
                            "constraint_scope": args.constraint_scope,
                            "random_phase": args.random_phase,
                            "shaping_npy": args.shaping_npy,
                            "num_z": args.num_z,
                            "train_max_frames": args.train_max_frames,
                            "frame_loss_weight": args.frame_loss_weight,
                        },
                    )
                    print(f"Saved best generator: {best_ckpt_path}")

        ckpt_path = os.path.join(
            args.output_dir,
            f"se_uap_v2_{ablation_tag(args)}_{args.dataset}_final.pth",
        )
        save_checkpoint(
            generator,
            ckpt_path,
            optimizer=optimizer,
            meta={
                "epoch": args.epochs,
                "tau": args.tau,
                "tau_start": args.tau_start,
                "tau_schedule": args.tau_schedule,
                "kappa": args.kappa,
                "alpha": args.alpha,
                "effective_alpha": args.effective_alpha,
                "uap_family": args.uap_family,
                "detector_loss": args.detector_loss,
                "detector_aggregation": args.detector_aggregation,
                "utterance_grad_clip": args.utterance_grad_clip,
                "grad_clip": args.grad_clip,
                "constraint_scope": args.constraint_scope,
                "random_phase": args.random_phase,
                "shaping_npy": args.shaping_npy,
                "num_z": args.num_z,
                "train_max_frames": args.train_max_frames,
                "frame_loss_weight": args.frame_loss_weight,
            },
        )
        print(f"Saved generator: {ckpt_path}")
        if val_files and best_score > -float("inf"):
            print(f"Best generator by validation score: {best_ckpt_path} (score={best_score:.4f})")
            print(f"Loading best generator for final test evaluation: {best_ckpt_path}")
            load_checkpoint(generator, best_ckpt_path, map_location=device)
        elif val_files:
            print("No best checkpoint selected (tau curriculum still annealing); using final generator.")
        writer.close()
    else:
        if not args.test_ckpt:
            raise ValueError("--test_ckpt is required in test mode")
        load_checkpoint(generator, args.test_ckpt, map_location=device)

    eval_metrics = evaluate(generator, speaker_model, detector, file_list,
                            paths["data_folder"], label_dict, args, device, shaper=shaper)
    print(
        f"SE-UAP v2 core ablation {ablation_tag(args)} {args.dataset}: "
        f"FR={eval_metrics['FR']:.4f}, FRD={eval_metrics['FRD']:.4f}, "
        f"SNR={eval_metrics['SNR']:.2f}, PESQ={eval_metrics['PESQ']:.4f}, "
        f"ATC={eval_metrics['ATC'] * 1000:.4f} ms, SEI={eval_metrics['SEI']:.4f}"
    )
    write_eval_results(eval_metrics, args, detector_meta)


if __name__ == "__main__":
    main()
