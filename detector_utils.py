# detector_utils.py
import numpy as np
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from audio_utils import load_audio
from common.model import SincClassifier
from common.utils import read_conf, get_dict_from_args
import argparse


class AdversarialDetectorCNN(nn.Module):
    """对抗样本检测器 CNN 模型"""
    def __init__(self):
        super(AdversarialDetectorCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=3)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=3)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=3)
        self.fc1 = nn.Linear(in_features=896, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=2)

    def forward(self, x):
        x = x.unsqueeze(1)               # (B, L) -> (B, 1, L)
        x = self.conv1(x)
        x = F.max_pool1d(x, 2)
        x = self.conv2(x)
        x = F.max_pool1d(x, 2)
        x = self.conv3(x)
        x = F.max_pool1d(x, 2)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def _init_fn(work_id, seed):
    """DataLoader 的 worker 初始化函数，保证可重复性"""
    np.random.seed(seed + work_id)


def read_list(filename):
    """读取文件列表（每行一个路径）"""
    with open(filename, "r") as fp:
        data = fp.readlines()
        data = [_l.strip() for _l in data]
    return data


def sentence_test(model, wav_data, wlen=3200, wshift=160, batch_size=128):
    """
    对整条语音做滑窗分帧预测，投票得出最终结果。
    修复短音频（长度 < wlen）的情况：补零后预测。
    """
    wav_data = wav_data.squeeze()          # 确保为 1D
    L = wav_data.shape[0]
    pred_all = []
    batch_data = []

    if L <= wlen:
        # 短音频：补零到 wlen 长度
        padded = torch.zeros(wlen, device=wav_data.device)
        padded[:L] = wav_data
        with torch.no_grad():
            pred = model(padded.unsqueeze(0))
        pred_all.append(pred)
    else:
        begin_idx = 0
        while begin_idx <= L - wlen:
            batch_data.append(wav_data[begin_idx:begin_idx + wlen])
            if len(batch_data) >= batch_size:
                pred_batch = model(torch.stack(batch_data))
                pred_all.append(pred_batch)
                batch_data = []
            begin_idx += wshift
        if batch_data:
            pred_batch = model(torch.stack(batch_data))
            pred_all.append(pred_batch)

    pred_probs = torch.softmax(torch.cat(pred_all, dim=0), dim=1)
    _, best_class = torch.max(torch.sum(pred_probs, dim=0), 0)
    return best_class


def evaluate_model(model, file_list, data_path, label, device, wlen=3200, wshift=160, batch_size=128):
    """
    测试 real 样本（标签为 label）的准确率。
    model: 检测器模型
    file_list: 音频文件名列表
    data_path: 音频文件所在目录
    label: 期望的标签（1 表示 real，0 表示 fake）
    device: 计算设备
    """
    if not file_list:
        return 0.0
    model.eval()
    correct = 0
    total = len(file_list)
    with torch.no_grad():
        for filename in file_list:
            data_norm, fs, _ = load_audio(osp.join(data_path, filename), expected_fs=16000, normalize=True)
            pred = sentence_test(
                model,
                torch.from_numpy(data_norm).float().to(device).unsqueeze(0),
                wlen=wlen, wshift=wshift, batch_size=batch_size
            )
            if pred == label:
                correct += 1
    return correct / total


def load_speaker_model(speaker_model_path, speaker_cfg_path, device):
    """加载预训练的 SincNet 说话人识别模型，支持 .pkl 和 .pth 格式"""
    args_temp = argparse.Namespace()
    args_speaker = read_conf(speaker_cfg_path, args_temp)

    CNN_arch = get_dict_from_args(['cnn_input_dim', 'cnn_N_filt', 'cnn_len_filt', 'cnn_max_pool_len',
                                   'cnn_use_laynorm_inp', 'cnn_use_batchnorm_inp', 'cnn_use_laynorm',
                                   'cnn_use_batchnorm', 'cnn_act', 'cnn_drop'], args_speaker.cnn)
    DNN_arch = get_dict_from_args(['fc_input_dim', 'fc_lay', 'fc_drop',
                                   'fc_use_batchnorm', 'fc_use_laynorm', 'fc_use_laynorm_inp',
                                   'fc_use_batchnorm_inp', 'fc_act'], args_speaker.dnn)
    Classifier = get_dict_from_args(['fc_input_dim', 'fc_lay', 'fc_drop',
                                     'fc_use_batchnorm', 'fc_use_laynorm', 'fc_use_laynorm_inp',
                                     'fc_use_batchnorm_inp', 'fc_act'], args_speaker.classifier)
    CNN_arch['fs'] = args_speaker.windowing.fs

    model = SincClassifier(CNN_arch, DNN_arch, Classifier)
    checkpoint = torch.load(speaker_model_path, map_location=device)

    if speaker_model_path.endswith('.pkl'):
        model.load_raw_state_dict(checkpoint)
    else:
        # 兼容 .pth 或直接 state_dict
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def fgsm_attack(speaker_model, wav_data, speaker_id, epsilon=0.01):
    """FGSM 单步对抗攻击"""
    wav_data_adv = wav_data.clone().detach().requires_grad_(True)
    outputs = speaker_model(wav_data_adv)
    loss = F.cross_entropy(outputs, speaker_id)
    loss.backward()
    perturbation = epsilon * wav_data_adv.grad.sign()
    fake_data = (wav_data + perturbation).clamp(-1, 1)
    return fake_data.detach()


def pgd_attack(speaker_model, wav_data, speaker_id, epsilon=0.01, alpha=0.003, num_steps=10):
    """PGD 多步对抗攻击"""
    fake_data = wav_data.clone().detach()
    for _ in range(num_steps):
        fake_data.requires_grad_(True)
        outputs = speaker_model(fake_data)
        loss = F.cross_entropy(outputs, speaker_id)
        loss.backward()
        fake_data = fake_data + alpha * fake_data.grad.sign()
        perturbation = torch.clamp(fake_data - wav_data, -epsilon, epsilon)
        fake_data = (wav_data + perturbation).clamp(-1, 1).detach()
    return fake_data


def evaluate_model_with_attack(detector_model, speaker_model, file_list, data_path,
                               label_dict, attack_method, device,
                               epsilon=0.01, wlen=3200, wshift=160):
    """
    在线生成对抗样本并测试检测器的准确率（fake 样本期望标签为 0）。
    label_dict: 文件名 -> 说话人 ID 的映射
    """
    if not file_list:
        return 0.0
    detector_model.eval()
    correct = 0
    total = len(file_list)

    for filename in file_list:
        data_norm, fs, _ = load_audio(osp.join(data_path, filename), expected_fs=16000, normalize=True)
        wav_tensor = torch.from_numpy(data_norm).float().to(device).unsqueeze(0)

        speaker_id = torch.tensor([label_dict[filename]]).long().to(device)

        L = wav_tensor.shape[1]
        fake_frames = []

        if L <= wlen:
            # 短音频补零
            padded = torch.zeros(1, wlen, device=device)
            padded[0, :L] = wav_tensor[0, :L]
            with torch.enable_grad():
                if attack_method == "fgsm":
                    fake_frame = fgsm_attack(speaker_model, padded, speaker_id, epsilon=epsilon)
                else:
                    fake_frame = pgd_attack(speaker_model, padded, speaker_id, epsilon=epsilon)
            fake_frames.append(fake_frame)
        else:
            begin_idx = 0
            while begin_idx <= L - wlen:
                frame = wav_tensor[:, begin_idx:begin_idx + wlen]
                with torch.enable_grad():
                    if attack_method == "fgsm":
                        fake_frame = fgsm_attack(speaker_model, frame, speaker_id, epsilon=epsilon)
                    else:
                        fake_frame = pgd_attack(speaker_model, frame, speaker_id, epsilon=epsilon)
                fake_frames.append(fake_frame)
                begin_idx += wshift

        if not fake_frames:
            continue

        with torch.no_grad():
            fake_all = torch.cat(fake_frames, dim=0)
            det_output = detector_model(fake_all)
            det_probs = torch.softmax(det_output, dim=1)
            _, best_class = torch.max(torch.sum(det_probs, dim=0), 0)

        if best_class.item() == 0:   # fake 标签为 0
            correct += 1

    return correct / total
