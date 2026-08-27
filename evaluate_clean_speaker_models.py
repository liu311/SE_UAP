#!/usr/bin/env python3
"""
Evaluate speaker recognition models on clean, non-adversarial audio.

This script reports the baseline performance of the speaker models on TIMIT and
LibriSpeech before any adversarial attack is applied. It uses the existing project
convention:

    TIMIT test list:
        <timit_data_root>/processed/test_8_2.scp
    TIMIT labels:
        <timit_data_root>/processed/TIMIT_labels.npy

    LibriSpeech test list:
        <libri_data_root>/Librispeech_spkid_sel/split_spk_list/libri_te_8_2.scp
    LibriSpeech labels:
        <libri_data_root>/Librispeech_spkid_sel/split_spk_list/libri_dict.npy

Audio is loaded as 16 kHz mono waveform and max-abs normalized by audio_utils.load_audio.
No adversarial perturbation, noise, or signal transformation is applied.
"""

import argparse
import csv
import os
import time
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from adversarial_utils import load_speaker_model, read_list, sentence_test
from audio_utils import load_audio


def default_dataset_paths(dataset: str, data_root: str):
    if dataset == "timit":
        return {
            "data_folder": data_root,
            "test_scp": os.path.join(data_root, "processed", "test_8_2.scp"),
            "label_dict": os.path.join(data_root, "processed", "TIMIT_labels.npy"),
        }
    if dataset == "libri":
        split_dir = os.path.join(data_root, "Librispeech_spkid_sel", "split_spk_list")
        return {
            "data_folder": os.path.join(data_root, "Librispeech_spkid_sel"),
            "test_scp": os.path.join(split_dir, "libri_te_8_2.scp"),
            "label_dict": os.path.join(split_dir, "libri_dict.npy"),
        }
    raise ValueError(f"Unsupported dataset: {dataset}")


def evaluate_clean_speaker_model(
    dataset: str,
    data_root: str,
    speaker_model_path: str,
    speaker_cfg_path: str,
    device,
    wlen: int = 3200,
    wshift: int = 160,
    batch_size: int = 128,
) -> Dict:
    paths = default_dataset_paths(dataset, data_root)
    for key in ["test_scp", "label_dict"]:
        if not os.path.exists(paths[key]):
            raise FileNotFoundError(f"{dataset} {key} not found: {paths[key]}")

    file_list = read_list(paths["test_scp"])
    label_dict = np.load(paths["label_dict"], allow_pickle=True).item()
    speaker_model = load_speaker_model(speaker_model_path, speaker_cfg_path, device)

    correct = 0
    total = 0
    skipped = 0

    speaker_model.eval()
    with torch.no_grad():
        for fname in tqdm(file_list, desc=f"Clean eval {dataset}", unit="file"):
            audio_path = os.path.join(paths["data_folder"], fname)
            if not os.path.exists(audio_path):
                print(f"Warning: missing file, skipped: {audio_path}")
                skipped += 1
                continue

            label = label_dict.get(fname)
            if label is None:
                print(f"Warning: missing label, skipped: {fname}")
                skipped += 1
                continue

            try:
                audio_norm, _, _ = load_audio(audio_path, expected_fs=16000, normalize=True)
            except Exception as exc:
                print(f"Warning: unreadable audio, skipped: {audio_path} ({exc})")
                skipped += 1
                continue

            wav_tensor = torch.from_numpy(audio_norm).float().to(device)
            pred = sentence_test(
                speaker_model,
                wav_tensor,
                wlen=wlen,
                wshift=wshift,
                batch_size=batch_size,
            )
            if torch.is_tensor(pred):
                pred = pred.item()

            total += 1
            correct += int(pred == label)

    accuracy = correct / total if total > 0 else 0.0
    return {
        "dataset": dataset,
        "test_files": len(file_list),
        "evaluated": total,
        "skipped": skipped,
        "correct": correct,
        "accuracy": accuracy,
        "error_rate": 1.0 - accuracy,
        "test_scp": paths["test_scp"],
        "label_dict": paths["label_dict"],
    }


def write_results(results: List[Dict], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"clean_speaker_model_results_{timestamp}.csv")
    md_path = os.path.join(output_dir, f"clean_speaker_model_results_{timestamp}.md")

    columns = ["dataset", "test_files", "evaluated", "skipped", "correct", "accuracy", "error_rate"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row[key] for key in columns})

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| Dataset | Test files | Evaluated | Skipped | Correct | Accuracy | Error rate |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in results:
            f.write(
                f"| {row['dataset']} | {row['test_files']} | {row['evaluated']} | "
                f"{row['skipped']} | {row['correct']} | {row['accuracy']:.4f} | "
                f"{row['error_rate']:.4f} |\n"
            )

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved Markdown table: {md_path}")


def parse_args():
    parser = argparse.ArgumentParser("Evaluate clean speaker-model performance on TIMIT and LibriSpeech")
    parser.add_argument("--dataset", choices=["all", "timit", "libri"], default="all")

    parser.add_argument("--timit_data_root", type=str, required=True)
    parser.add_argument("--timit_speaker_model", type=str, required=True)
    parser.add_argument("--timit_speaker_cfg", type=str, required=True)

    parser.add_argument("--libri_data_root", type=str, required=True)
    parser.add_argument("--libri_speaker_model", type=str, required=True)
    parser.add_argument("--libri_speaker_cfg", type=str, required=True)

    parser.add_argument("--wlen", type=int, default=3200, help="speaker-model frame length in samples")
    parser.add_argument("--wshift", type=int, default=160, help="speaker-model sliding-window shift in samples")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="./output/clean_speaker_eval")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    jobs = []
    if args.dataset in ["all", "timit"]:
        jobs.append((
            "timit",
            args.timit_data_root,
            args.timit_speaker_model,
            args.timit_speaker_cfg,
        ))
    if args.dataset in ["all", "libri"]:
        jobs.append((
            "libri",
            args.libri_data_root,
            args.libri_speaker_model,
            args.libri_speaker_cfg,
        ))

    results = []
    for dataset, data_root, speaker_model, speaker_cfg in jobs:
        result = evaluate_clean_speaker_model(
            dataset=dataset,
            data_root=data_root,
            speaker_model_path=speaker_model,
            speaker_cfg_path=speaker_cfg,
            device=device,
            wlen=args.wlen,
            wshift=args.wshift,
            batch_size=args.batch_size,
        )
        results.append(result)
        print(
            f"{dataset}: accuracy={result['accuracy']:.4f}, "
            f"error_rate={result['error_rate']:.4f}, "
            f"evaluated={result['evaluated']}, skipped={result['skipped']}"
        )

    write_results(results, args.output_dir)


if __name__ == "__main__":
    main()
