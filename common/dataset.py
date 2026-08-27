"""
Specialized dataset for adversarial detector training.
NO oversampling - only uses physical data, relies on online FGSM/PGD generation for diversity.
"""
import os
import os.path as osp
import numpy as np
from torch.utils.data import Dataset
from audio_utils import load_audio, max_abs_normalize


def read_list(filename):
    with open(filename, "r") as fp:
        data = fp.readlines()
        data = [_l.strip() for _l in data]
    return data


class TIMITDetectorDataset(Dataset):
    """
    Non-oversampled dataset for detector training on TIMIT.
    - Physical dataset size: N files
    - Logical dataset size: N files (no 100x multiplication)
    - Frame extraction: random within each file (for diversity) or fixed (for reproducibility)
    """

    def __init__(self, data_root, train=True, fs=16000, wlen=200, frame_mode='fixed'):
        """
        Args:
            data_root: path to TIMIT dataset root
            train: boolean, train or test split
            fs: sample rate (Hz)
            wlen: frame length (milliseconds)
            frame_mode: 'fixed' (center frame) or 'random' (random frame per epoch)
        """
        super().__init__()
        data_root_processed = osp.join(data_root, "processed")
        self.data_root = data_root
        self.fs = fs
        self.f_wlen = int(fs * wlen / 1000)
        self.frame_mode = frame_mode
        self.split = "train" if train else "test"

        # Read file list and labels
        scp_name = "train_8_2.scp" if train else "test_8_2.scp"
        data_list_file = osp.join(data_root_processed, scp_name)
        print(f"[DetectorDataset] Loading {self.split} list from: {data_list_file}")
        self.data = read_list(data_list_file)

        label_file = osp.join(data_root_processed, "TIMIT_labels.npy")
        print(f"[DetectorDataset] Loading labels from: {label_file}")
        self.label_dict = np.load(label_file, allow_pickle=True).item()

        print(f"[DetectorDataset] {self.split} split: {len(self.data)} files (no oversampling)")

    @staticmethod
    def preprocess(wav_data):
        """Normalize audio to the speaker model's valid waveform range."""
        return max_abs_normalize(wav_data)

    def __len__(self):
        # KEY DIFFERENCE: return actual file count, not 100x
        return len(self.data)

    def __getitem__(self, index):
        filename = self.data[index]
        wav_path = osp.join(self.data_root, filename)

        # Load normalized 16 kHz mono waveform.
        data_wav, fs, norm_factor = load_audio(wav_path, expected_fs=self.fs, normalize=True)

        # Extract frame
        data_len = len(data_wav)
        if data_len < self.f_wlen:
            # Pad if too short
            data_wav = np.concatenate([data_wav, np.zeros(self.f_wlen - data_len)])
            frame = data_wav[:self.f_wlen]
        elif self.frame_mode == 'fixed':
            # Extract center frame (reproducible)
            center = data_len // 2
            start = max(0, min(center - self.f_wlen // 2, data_len - self.f_wlen))
            frame = data_wav[start:start + self.f_wlen]
        else:  # 'random'
            # Random frame (diversity)
            offset = np.random.randint(0, data_len - self.f_wlen + 1)
            frame = data_wav[offset:offset + self.f_wlen]

        speaker_id = self.label_dict[filename]

        return frame, speaker_id, norm_factor


class LibriSpeechDetectorDataset(Dataset):
    """
    Non-oversampled dataset for detector training on LibriSpeech.
    Same logic as TIMITDetectorDataset but for LibriSpeech paths.
    """

    def __init__(self, data_root, train=True, fs=16000, wlen=200, frame_mode='fixed'):
        """
        Args:
            data_root: path to LibriSpeech dataset root
            train: boolean, train or test split
            fs: sample rate (Hz)
            wlen: frame length (milliseconds)
            frame_mode: 'fixed' (center frame) or 'random' (random frame per epoch)
        """
        super().__init__()
        data_root_processed = osp.join(data_root, 'Librispeech_spkid_sel/split_spk_list')
        self.data_root = data_root
        self.fs = fs
        self.f_wlen = int(fs * wlen / 1000)
        self.frame_mode = frame_mode
        self.split = "train" if train else "test"

        # Read file list and labels
        scp_file = "libri_tr_8_2.scp" if train else "libri_te_8_2.scp"
        data_list_file = osp.join(data_root_processed, scp_file)
        print(f"[DetectorDataset] Loading {self.split} list from: {data_list_file}")
        self.data = read_list(data_list_file)

        label_file = osp.join(data_root_processed, "libri_dict.npy")
        print(f"[DetectorDataset] Loading labels from: {label_file}")
        self.label_dict = np.load(label_file, allow_pickle=True).item()

        print(f"[DetectorDataset] {self.split} split: {len(self.data)} files (no oversampling)")

    @staticmethod
    def preprocess(wav_data):
        """Normalize audio to the speaker model's valid waveform range."""
        return max_abs_normalize(wav_data)

    def __len__(self):
        # KEY DIFFERENCE: return actual file count, not 100x
        return len(self.data)

    def __getitem__(self, index):
        filename = self.data[index]
        wav_path = osp.join(self.data_root, "Librispeech_spkid_sel", filename)

        # Load normalized 16 kHz mono waveform.
        data_wav, fs, norm_factor = load_audio(wav_path, expected_fs=self.fs, normalize=True)

        # Extract frame
        data_len = len(data_wav)
        if data_len < self.f_wlen:
            # Pad if too short
            data_wav = np.concatenate([data_wav, np.zeros(self.f_wlen - data_len)])
            frame = data_wav[:self.f_wlen]
        elif self.frame_mode == 'fixed':
            # Extract center frame (reproducible)
            center = data_len // 2
            start = max(0, min(center - self.f_wlen // 2, data_len - self.f_wlen))
            frame = data_wav[start:start + self.f_wlen]
        else:  # 'random'
            # Random frame (diversity)
            offset = np.random.randint(0, data_len - self.f_wlen + 1)
            frame = data_wav[offset:offset + self.f_wlen]

        speaker_id = self.label_dict[filename]

        return frame, speaker_id, norm_factor
