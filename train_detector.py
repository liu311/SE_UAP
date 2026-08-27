"""
Modified train_detector.py using TIMITDetectorDataset / LibriSpeechDetectorDataset
Key changes:
- Replace TIMIT_speaker_norm / LibriSpeech_speaker with detector-specific classes
- Reduced epochs (from 10 to 5) since less redundant data
- Increased batch_size for better GPU utilization
"""

import sys

sys.path.append('/home/chenyuying/Projects/me')
import os
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from functools import partial
from tqdm import tqdm
from common.dataset import TIMITDetectorDataset, LibriSpeechDetectorDataset  # <-- NEW
from adversarial_utils import (
    AdversarialDetectorCNN, read_list, load_speaker_model,
    fgsm_attack, pgd_attack, evaluate_detector_on_real, evaluate_detector_on_fake
)
def _init_fn(work_id, seed):
    np.random.seed(seed + work_id)


def get_parser():
    parser = argparse.ArgumentParser("Train Adversarial Detector (No Oversampling)")
    parser.add_argument("--dataset", choices=['timit', 'libri'], default='timit')
    parser.add_argument("--data_root", type=str, required=True, help="root path of dataset")
    parser.add_argument("--speaker_model", type=str, required=True)
    parser.add_argument("--speaker_cfg", type=str, required=True)
    parser.add_argument("--attack", choices=['fgsm', 'pgd'], default='fgsm')
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=256)  # <-- INCREASED from 128
    parser.add_argument("--epochs", type=int, default=5)  # <-- REDUCED from 10
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wlen", type=int, default=200, help="frame length in ms (200ms = 3200 samples @ 16kHz)")
    parser.add_argument("--output_dir", type=str, default="./output/detector")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--frame_mode", choices=['fixed', 'random'], default='random',
                        help="'fixed': center frame per file, 'random': random offset per epoch")
    return parser


def main():
    args = get_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Dataset selection
    if args.dataset == 'timit':
        # <-- CHANGED: Use TIMITDetectorDataset instead of TIMIT_speaker_norm
        train_dataset = TIMITDetectorDataset(
            args.data_root, train=True, wlen=args.wlen, frame_mode=args.frame_mode
        )
        test_scp = os.path.join(args.data_root, "processed", "test_8_2.scp")
        label_dict_path = os.path.join(args.data_root, "processed", "TIMIT_labels.npy")
        audio_path = args.data_root
        dataset_name = "timit"
    else:  # libri
        # <-- CHANGED: Use LibriSpeechDetectorDataset instead of LibriSpeech_speaker
        train_dataset = LibriSpeechDetectorDataset(
            args.data_root, train=True, wlen=args.wlen, frame_mode=args.frame_mode
        )
        test_scp = os.path.join(args.data_root, "Librispeech_spkid_sel/split_spk_list", "libri_te_8_2.scp")
        label_dict_path = os.path.join(args.data_root, "Librispeech_spkid_sel/split_spk_list", "libri_dict.npy")
        audio_path = os.path.join(args.data_root, "Librispeech_spkid_sel")
        dataset_name = "libri"

    test_file_list = read_list(test_scp)
    label_dict = np.load(label_dict_path, allow_pickle=True).item()

    # Load speaker recognition model
    speaker_model = load_speaker_model(args.speaker_model, args.speaker_cfg, device)
    print("speaker_model----------", speaker_model)

    # Create detector
    detector = AdversarialDetectorCNN().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(detector.parameters(), lr=args.lr)

    # DataLoader
    loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
        worker_init_fn=partial(_init_fn, seed=args.seed)
    )

    # TensorBoard
    log_dir = os.path.join(args.output_dir, f"run_{time.strftime('%Y%m%d_%H%M%S')}")
    writer = SummaryWriter(log_dir)
    global_step = 0

    # Print training info
    print(f"\n{'=' * 60}")
    print(f"Detector Training (No Oversampling)")
    print(f"{'=' * 60}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Physical train samples: {len(train_dataset)} files")  # <-- Much smaller!
    print(f"Logical epoch size: {len(train_dataset)} files (vs 100x in old code)")
    print(f"Batch size: {args.batch_size}")
    print(f"Expected batches/epoch: {len(loader)}")
    print(f"Epochs: {args.epochs}")
    print(f"Frame mode: {args.frame_mode}")
    print(f"Test files: {len(test_file_list)}")
    print(f"{'=' * 60}\n")

    for epoch in range(args.epochs):
        detector.train()
        acc_real_epoch = []
        acc_fake_epoch = []
        loss_epoch = []

        pbar = tqdm(loader, desc=f'Epoch {epoch + 1}/{args.epochs}', unit='batch')
        for real_wav, spk_id, _ in pbar:
            real_wav = real_wav.float().to(device)
            spk_id = spk_id.long().to(device)

            # Generate adversarial samples (ONLINE - provides diversity)
            if args.attack == 'fgsm':
                fake_wav = fgsm_attack(speaker_model, real_wav, spk_id, args.epsilon)
            else:
                fake_wav = pgd_attack(speaker_model, real_wav, spk_id, args.epsilon)

            optimizer.zero_grad()

            # Real sample loss (label=1)
            real_out = detector(real_wav)
            real_loss = criterion(real_out, torch.ones_like(spk_id))

            # Fake sample loss (label=0)
            fake_out = detector(fake_wav)
            fake_loss = criterion(fake_out, torch.zeros_like(spk_id))

            loss = (real_loss + fake_loss) * 0.5
            loss.backward()
            optimizer.step()

            loss_epoch.append(loss.item())
            acc_real = (real_out.argmax(1) == 1).float().mean().item()
            acc_fake = (fake_out.argmax(1) == 0).float().mean().item()
            acc_real_epoch.append(acc_real)
            acc_fake_epoch.append(acc_fake)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc_r': f'{acc_real:.4f}',
                'acc_f': f'{acc_fake:.4f}'
            })

            writer.add_scalar('train/batch_loss', loss.item(), global_step)
            writer.add_scalar('train/batch_acc_real', acc_real, global_step)
            writer.add_scalar('train/batch_acc_fake', acc_fake, global_step)
            global_step += 1

        avg_loss = np.mean(loss_epoch)
        avg_acc_real = np.mean(acc_real_epoch)
        avg_acc_fake = np.mean(acc_fake_epoch)
        print(
            f"Epoch {epoch + 1}/{args.epochs} | Loss: {avg_loss:.4f} | Acc_real: {avg_acc_real:.4f} | Acc_fake: {avg_acc_fake:.4f}")
        writer.add_scalar('epoch/loss', avg_loss, epoch)
        writer.add_scalar('epoch/acc_real', avg_acc_real, epoch)
        writer.add_scalar('epoch/acc_fake', avg_acc_fake, epoch)

    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"detector_{dataset_name}_{args.attack}_noovs.pth")
    torch.save(detector.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Test
    detector.eval()
    acc_real = evaluate_detector_on_real(detector, test_file_list, audio_path, device,wshift=3200)
    acc_fake = evaluate_detector_on_fake(detector, speaker_model, test_file_list, audio_path,
                                         label_dict, args.attack, device, args.epsilon,wshift=3200)
    print(f"Test Results: Real ACC = {acc_real:.4f}, Fake ACC = {acc_fake:.4f}")
    writer.add_scalar('test/acc_real', acc_real, 0)
    writer.add_scalar('test/acc_fake', acc_fake, 0)
    writer.close()


if __name__ == "__main__":
    main()