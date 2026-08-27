#!/usr/bin/env python3
"""
Evaluate a pre-trained speaker recognition model on a test set.
Supports TIMIT and LibriSpeech datasets.
"""

import os
import argparse
import numpy as np
import torch
import tqdm
from audio_utils import load_audio

from adversarial_utils import load_speaker_model, sentence_test, read_list


def evaluate_speaker_model(speaker_model, file_list, data_folder, label_dict, device):
    """
    Evaluate speaker model accuracy on clean audio.

    Args:
        speaker_model: PyTorch model (e.g., SpeakerNet)
        file_list: list of relative audio file paths (wav/scp)
        data_folder: root directory containing audio files
        label_dict: dict mapping filename -> speaker label (int)
        device: torch device

    Returns:
        accuracy (float)
    """
    speaker_model.eval()
    correct = 0
    total = 0
    skipped = 0

    with torch.no_grad():
        for fname in tqdm.tqdm(file_list, desc="Evaluating speaker model"):
            # Load audio
            audio_path = os.path.join(data_folder, fname)
            if not os.path.exists(audio_path):
                print(f"Warning: {audio_path} not found, skipping")
                skipped += 1
                continue

            try:
                audio_norm, sr, _ = load_audio(audio_path, expected_fs=16000, normalize=True)
            except Exception as e:
                print(f"Error reading {audio_path}: {e}, skipping")
                skipped += 1
                continue

            # Convert to tensor and move to device
            input_tensor = torch.from_numpy(audio_norm).float().to(device)

            # Get predicted speaker ID
            pred = sentence_test(speaker_model, input_tensor)  # returns int or 0-dim tensor
            if torch.is_tensor(pred):
                pred = pred.item()

            label = label_dict.get(fname)
            if label is None:
                print(f"Warning: No label for {fname}, skipping")
                skipped += 1
                continue

            total += 1
            if pred == label:
                correct += 1

    if total == 0:
        print("No valid files found to evaluate.")
        return 0.0, 0, 0

    accuracy = correct / total
    if skipped > 0:
        print(f"Skipped {skipped} files due to missing or unreadable files.")
    return accuracy, correct, total


def main():
    parser = argparse.ArgumentParser(description="Evaluate speaker model on test set")
    parser.add_argument("--dataset", type=str, choices=['timit', 'libri'], default=None,
                        help="Dataset type (optional if --test_scp and --label_dict are given)")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Root directory of dataset (used to construct default paths if --test_scp not given)")
    parser.add_argument("--speaker_model", type=str, required=True,
                        help="Path to speaker model checkpoint (.pt or .pth)")
    parser.add_argument("--speaker_cfg", type=str, required=True,
                        help="Path to speaker model config file (e.g., model_config.yaml)")
    parser.add_argument("--test_scp", type=str, default=None,
                        help="Path to test list file (scp) containing relative audio paths. If not provided, will be constructed from --dataset and --data_root.")
    parser.add_argument("--label_dict", type=str, default=None,
                        help="Path to label dictionary (.npy). If not provided, will be constructed from --dataset and --data_root.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda/cpu)")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Determine test_scp and label_dict paths
    if args.test_scp is None or args.label_dict is None:
        if args.dataset is None or args.data_root is None:
            parser.error("Either provide --test_scp and --label_dict, or provide both --dataset and --data_root.")
        if args.dataset == 'timit':
            default_test_scp = os.path.join(args.data_root, "processed", "test_8_2.scp")
            default_label_dict = os.path.join(args.data_root, "processed", "TIMIT_labels.npy")
            data_folder = args.data_root
        elif args.dataset == 'libri':
            default_test_scp = os.path.join(args.data_root, "Librispeech_spkid_sel/split_spk_list", "libri_te_8_2.scp")
            default_label_dict = os.path.join(args.data_root, "Librispeech_spkid_sel/split_spk_list", "libri_dict.npy")
            data_folder = os.path.join(args.data_root, "Librispeech_spkid_sel")
        else:
            raise ValueError("Unsupported dataset")

        test_scp = args.test_scp if args.test_scp is not None else default_test_scp
        label_dict_path = args.label_dict if args.label_dict is not None else default_label_dict
    else:
        test_scp = args.test_scp
        label_dict_path = args.label_dict
        # When user provides explicit files, data_root must also be provided
        if args.data_root is None:
            parser.error("--data_root is required when providing custom --test_scp and --label_dict")
        data_folder = args.data_root

    # Check existence of required files
    if not os.path.exists(test_scp):
        raise FileNotFoundError(f"Test list file not found: {test_scp}")
    if not os.path.exists(label_dict_path):
        raise FileNotFoundError(f"Label dictionary not found: {label_dict_path}")

    # Load speaker model
    print("Loading speaker model...")
    speaker_model = load_speaker_model(args.speaker_model, args.speaker_cfg, device)
    print("Model loaded.")

    # Read file list and labels
    file_list = read_list(test_scp)
    label_dict = np.load(label_dict_path, allow_pickle=True).item()

    print(f"Test set size: {len(file_list)} utterances")
    print(f"Number of unique speakers: {len(set(label_dict.values()))}")

    # Evaluate
    accuracy, correct, total = evaluate_speaker_model(speaker_model, file_list, data_folder, label_dict, device)

    # Print results
    print("\n" + "=" * 50)
    print("Speaker Model Evaluation Results")
    print("=" * 50)
    print(f"Total utterances: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Error rate: {1-accuracy:.4f} ({ (1-accuracy)*100:.2f}%)")
    print("=" * 50)


if __name__ == "__main__":
    main()
