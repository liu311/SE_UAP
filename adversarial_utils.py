# adversarial_utils.py
import os
import os.path as osp
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from audio_utils import load_audio, max_abs_normalize
from common.model import SincClassifier
from common.utils import read_conf, get_dict_from_args, SNR, PESQ
from common.trainer import load_checkpoint
import argparse
from tqdm import tqdm

# -------------------- 模型定义 --------------------
class AdversarialDetectorCNN(nn.Module):
    def __init__(self):
        super(AdversarialDetectorCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=3)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=3)
        self.fc1 = nn.Linear(896, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = x.unsqueeze(1)                     # (B, L) -> (B, 1, L)
        x = F.max_pool1d(self.conv1(x), 2)
        x = F.max_pool1d(self.conv2(x), 2)
        x = F.max_pool1d(self.conv3(x), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# -------------------- 工具函数 --------------------
class TransformerAdversarialDetector(nn.Module):
    """Frame-level Transformer detector for fixed-length waveform frames."""

    def __init__(self, frame_dim=3200, patch_size=80, embed_dim=128,
                 num_heads=4, num_layers=3, ff_dim=256, dropout=0.1):
        super().__init__()
        if frame_dim % patch_size != 0:
            raise ValueError("frame_dim must be divisible by patch_size")
        self.frame_dim = frame_dim
        self.patch_size = patch_size
        self.num_patches = frame_dim // patch_size

        self.patch_embed = nn.Linear(patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 2)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        if x.dim() != 2:
            x = x.squeeze()
        x = x.view(x.size(0), self.num_patches, self.patch_size)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)


def read_list(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]

def preprocess(wav_data):
    """归一化到 [-1,1]，返回 (归一化信号, 归一化因子)"""
    return max_abs_normalize(wav_data)

@torch.no_grad()
def sentence_test(model, wav_data, wlen=3200, wshift=160, batch_size=128):
    """
    滑窗投票得到整条语音的预测标签。
    model: 分类模型 (SincClassifier 或 AdversarialDetectorCNN)
    wav_data: 1D tensor 或 (1, L) tensor
    """
    wav_data = wav_data.squeeze()
    L = wav_data.shape[0]
    pred_all = []
    batch_data = []

    if L <= wlen:
        padded = torch.zeros(wlen, device=wav_data.device)
        padded[:L] = wav_data
        pred = model(padded.unsqueeze(0))
        pred_all.append(pred)
    else:
        begin = 0
        while begin <= L - wlen:
            batch_data.append(wav_data[begin:begin+wlen])
            if len(batch_data) >= batch_size:
                pred_all.append(model(torch.stack(batch_data)))
                batch_data = []
            begin += wshift
        if batch_data:
            pred_all.append(model(torch.stack(batch_data)))

    probs = torch.softmax(torch.cat(pred_all, dim=0), dim=1)
    _, best = torch.max(torch.sum(probs, dim=0), 0)
    return best


# -------------------- 加载预训练模型 --------------------
def load_speaker_model(model_path, cfg_path, device):
    """加载 SincNet 说话人识别模型，支持 .pkl 和 .pth"""
    args_temp = argparse.Namespace()
    args = read_conf(cfg_path, args_temp)

    CNN_arch = get_dict_from_args(['cnn_input_dim', 'cnn_N_filt', 'cnn_len_filt', 'cnn_max_pool_len',
                                   'cnn_use_laynorm_inp', 'cnn_use_batchnorm_inp', 'cnn_use_laynorm',
                                   'cnn_use_batchnorm', 'cnn_act', 'cnn_drop'], args.cnn)
    DNN_arch = get_dict_from_args(['fc_input_dim', 'fc_lay', 'fc_drop', 'fc_use_batchnorm', 'fc_use_laynorm',
                                   'fc_use_laynorm_inp', 'fc_use_batchnorm_inp', 'fc_act'], args.dnn)
    Classifier = get_dict_from_args(['fc_input_dim', 'fc_lay', 'fc_drop', 'fc_use_batchnorm', 'fc_use_laynorm',
                                     'fc_use_laynorm_inp', 'fc_use_batchnorm_inp', 'fc_act'], args.classifier)
    CNN_arch['fs'] = args.windowing.fs
    model = SincClassifier(CNN_arch, DNN_arch, Classifier)

    checkpoint = torch.load(model_path, map_location=device)
    if model_path.endswith('.pkl'):
        model.load_raw_state_dict(checkpoint)
    else:
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

def load_detector_model(model_path, device):
    state = torch.load(model_path, map_location=device)
    detector_type = state.get("detector_type", "cnn") if isinstance(state, dict) else "cnn"
    detector_kwargs = state.get("detector_kwargs", {}) if isinstance(state, dict) else {}
    if detector_type == "transformer":
        model = TransformerAdversarialDetector(**detector_kwargs)
    else:
        model = AdversarialDetectorCNN()
    if 'state_dict' in state:
        model.load_state_dict(state['state_dict'])
    else:
        model.load_state_dict(state)
    return model.to(device).eval()


# -------------------- 对抗攻击方法 --------------------
def fgsm_attack(speaker_model, wav_data, speaker_id, epsilon=0.01):
    wav_adv = wav_data.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(speaker_model(wav_adv), speaker_id)
    loss.backward()
    pert = epsilon * wav_adv.grad.sign()
    return (wav_data + pert).clamp(-1, 1).detach()

def pgd_attack(speaker_model, wav_data, speaker_id, epsilon=0.01, alpha=0.003, steps=10):
    adv = wav_data.clone().detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(speaker_model(adv), speaker_id)
        loss.backward()
        adv = adv + alpha * adv.grad.sign()
        pert = torch.clamp(adv - wav_data, -epsilon, epsilon)
        adv = (wav_data + pert).clamp(-1, 1).detach()
    return adv


# -------------------- 评估函数 --------------------
def evaluate_detector_on_real(detector, file_list, data_path, device, wlen=3200, wshift=160):
    """测试检测器对真实语音的准确率 (标签=1)，带进度条"""
    if not file_list:
        return 0.0
    detector.eval()
    correct = 0
    total = len(file_list)

    # 添加 tqdm 进度条
    for fname in tqdm(file_list, desc="Testing real samples", unit="file"):
        audio_norm, _, _ = load_audio(osp.join(data_path, fname), expected_fs=16000, normalize=True)
        pred = sentence_test(detector, torch.from_numpy(audio_norm).float().to(device), wlen, wshift)
        if pred == 1:
            correct += 1
    return correct / total


def evaluate_detector_on_fake(detector, speaker_model, file_list, data_path, label_dict,
                              attack_method, device, epsilon=0.01, wlen=3200, wshift=160):
    """在线生成对抗样本并测试检测器识别fake样本的能力 (标签=0)，带进度条"""
    if not file_list:
        return 0.0
    detector.eval()
    correct = 0
    total = len(file_list)

    for fname in tqdm(file_list, desc="Testing fake samples", unit="file"):
        audio_norm, _, _ = load_audio(osp.join(data_path, fname), expected_fs=16000, normalize=True)
        wav_tensor = torch.from_numpy(audio_norm).float().to(device).unsqueeze(0)
        spk_id = torch.tensor([label_dict[fname]]).long().to(device)

        L = wav_tensor.shape[1]
        fake_frames = []
        if L <= wlen:
            padded = torch.zeros(1, wlen, device=device)
            padded[0, :L] = wav_tensor[0, :L]
            with torch.enable_grad():
                if attack_method == 'fgsm':
                    fake = fgsm_attack(speaker_model, padded, spk_id, epsilon)
                else:
                    fake = pgd_attack(speaker_model, padded, spk_id, epsilon)
            fake_frames.append(fake)
        else:
            begin = 0
            while begin <= L - wlen:
                frame = wav_tensor[:, begin:begin + wlen]
                with torch.enable_grad():
                    if attack_method == 'fgsm':
                        fake = fgsm_attack(speaker_model, frame, spk_id, epsilon)
                    else:
                        fake = pgd_attack(speaker_model, frame, spk_id, epsilon)
                fake_frames.append(fake)
                begin += wshift

        if not fake_frames:
            continue
        fake_all = torch.cat(fake_frames, dim=0)
        out = detector(fake_all)
        probs = torch.softmax(out, dim=1)
        _, best = torch.max(torch.sum(probs, dim=0), 0)
        if best == 0:
            correct += 1
    return correct / total
