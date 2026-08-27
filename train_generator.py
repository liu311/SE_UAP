import sys
import os
import time
import argparse
import numpy as np
import random
import tqdm
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from audio_utils import load_audio, repeat_to_length

from common.dataset import TIMITDetectorDataset, LibriSpeechDetectorDataset
from common.trainer import ClassifierTrainer, load_checkpoint, save_checkpoint
from common.utils import read_conf, get_dict_from_args
from generator import Generator1D
from adversarial_utils import (
    AdversarialDetectorCNN, load_speaker_model, load_detector_model,
    sentence_test, preprocess, SNR, PESQ, read_list
)


def project_l2(delta: torch.Tensor, tau: float, eps: float = 1e-12):
    """Project generator output onto the L2 perturbation ball described in the manuscript."""
    flat = delta.view(delta.size(0), -1)
    norm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
    scale = torch.clamp(tau / norm, max=1.0)
    return (flat * scale).view_as(delta)


class RunningAverage:
    def __init__(self):
        self.sums = {}
        self.counts = {}

    def update(self, vals: dict, counts: dict):
        for k, v in vals.items():
            self.sums[k] = self.sums.get(k, 0.0) + v * counts[k]
            self.counts[k] = self.counts.get(k, 0) + counts[k]

    def average(self):
        return {k: self.sums[k] / max(self.counts[k], 1) for k in self.sums}


# =========================================================
# 与第二份代码保持一致的帧级 Preprocess
# =========================================================
def preprocess_batch_maxabs(wav_batch: torch.Tensor, eps: float = 1e-8):
    """
    仿照第二份代码中的 Preprocess：
        wav_data = wav_data / max(abs(wav_data))

    输入:
        wav_batch: (B, T)
    输出:
        wav_norm: (B, T)
        norm_factor: (B, 1)
    """
    norm_factor = wav_batch.abs().amax(dim=1, keepdim=True).clamp_min(eps)
    wav_norm = wav_batch / norm_factor
    return wav_norm, norm_factor


# =========================================================
# 损失函数
# =========================================================
class SpeakerLoss(nn.Module):
    """
    非目标攻击：
    希望真实类 logit 小于某个其他类 logit
    """
    def __init__(self, margin=100.0):
        super().__init__()
        self.margin = margin

    def forward(self, logits, label):
        """
        logits: (B, C)
        label : (B,)
        """
        B = logits.size(0)
        correct = logits[torch.arange(B, device=logits.device), label]

        # 修复原代码 bug：
        # 不能用乘0的方式去掉真实类，否则当所有 logit 都是负数时，max_other 会错误变成 0
        mask = F.one_hot(label, num_classes=logits.size(1)).bool()
        max_other = logits.masked_fill(mask, -1e9).max(dim=1)[0]

        loss = (correct - max_other + self.margin).clamp(min=0.0) / self.margin
        return loss.mean()


class SpeakerLossTarget(nn.Module):
    """
    目标攻击：
    希望 target 类 logit 大于所有其他类
    """
    def __init__(self, target, margin=100.0):
        super().__init__()
        self.target = target
        self.margin = margin

    def forward(self, logits, label=None):
        f_target = logits[:, self.target]

        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[:, self.target] = False
        max_other = logits.masked_fill(~mask, -1e9).max(dim=1)[0]

        loss = (max_other - f_target + self.margin).clamp(min=0.0) / self.margin
        return loss.mean()


class DetectorLoss(nn.Module):
    """
    与原代码一致：
    detector 输出 logits, label=1 表示 benign / clean 类
    """
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, label):
        x = self.ce(logits, label)
        bx = torch.clamp(self.beta * x, max=20.0)
        return (torch.exp(bx) - 1.0).mean()


class MSEWithThreshold(nn.Module):
    def __init__(self, threshold=0.05):
        super().__init__()
        self.threshold = threshold

    def forward(self, pred, label):
        err = pred - label
        err_abs = torch.abs(err) - self.threshold
        err_abs = torch.clamp(err_abs, min=0.0)

        valid = (err_abs > 0)
        if valid.sum() == 0:
            return err_abs.sum()
        else:
            return err_abs[valid].mean()


class UniversalLoss:
    """
    pred 是 generator 输出后的波形 (B, T)
    """
    def __init__(self, loss_all):
        self.loss_all = loss_all

    def __call__(self, pred, labels):
        loss_dict = {}
        pred_dict = {}
        label_dict = {}
        loss_tensor_dict = {}

        for key, cfg in self.loss_all.items():
            model = cfg.get('model', None)

            if key == 'norm':
                pred_this = pred
            elif key == 'speaker':
                pred_this = model(pred)
            elif key == 'detector':
                # detector 分支维持原训练逻辑
                pred_this = model(pred)
            else:
                pred_this = model(pred)

            label_this = labels[key]
            loss = cfg['loss_func'](pred_this, label_this) * cfg['factor']

            loss_tensor_dict[key] = loss
            loss_dict[key] = loss.detach().item()
            pred_dict[key] = pred_this.detach()
            label_dict[key] = label_this.detach() if torch.is_tensor(label_this) else label_this

        loss_total = sum(loss_tensor_dict.values())
        loss_dict['total'] = loss_total.detach().item()

        return loss_total, loss_dict, None, pred_dict, label_dict


# =========================================================
# hook 保存梯度
# =========================================================
grads = {}

def save_grad(v):
    def hook(grad):
        grads[v] = grad
    return hook


# =========================================================
# 训练 / 验证时一个 batch 的处理
# =========================================================
def batch_process_generator(model, data, train_mode=True, **kwargs):
    """
    第一份代码原本就是帧级训练；
    这里修改为：
      1) 生成对抗帧 pout
      2) speaker 分支前先做 max-abs 归一化（与第二份代码 Preprocess 对齐）
    """
    wav_data, speaker_id, norm_factor = data

    device = next(model.parameters()).device
    wav_data = wav_data.float().to(device)            # (B, 3200)
    speaker_id = speaker_id.long().to(device)
    norm_factor = norm_factor.float().to(device)

    batch_size = wav_data.shape[0]
    # SE-UAP samples one latent vector and applies the resulting universal perturbation
    # to every sample in the mini-batch.
    noise = torch.randn(1, model.noise_dim, device=device)

    noise_scale = kwargs.get('noise_scale', 1.0)
    target = kwargs.get('target', -1)
    tau = kwargs.get('tau', 0.1)

    if train_mode:
        model.train()
        optimizer = kwargs['optimizer']
        loss_func = kwargs['loss_func']

        delta = project_l2(model(noise) * noise_scale, tau).expand_as(wav_data)
        pout = (wav_data + delta).clamp(-1, 1)
        pout.register_hook(save_grad('wav_d'))

        labels = {
            'speaker': speaker_id,
            'norm': wav_data,
            'detector': torch.ones(batch_size, dtype=torch.long, device=device)
        }

        loss_total, loss_dict, _, pred_dict, label_dict = loss_func(pout, labels)

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        grad_dict = {}
        if 'wav_d' in grads:
            grad_dict['grad_wav_mean_abs'] = grads['wav_d'].abs().mean().item()

        # pred_dict['speaker'] is computed from the adversarial normalized waveform.
        pred_spk = pred_dict['speaker'].argmax(dim=1)
        if target < 0:
            err_spk = (pred_spk != label_dict['speaker']).float().mean().item()
        else:
            err_spk = (pred_spk == target).float().mean().item()

        pred_det = pred_dict['detector'].argmax(dim=1)
        err_det = (pred_det == label_dict['detector']).float().mean().item()

        noise_stats = pout - wav_data
        noise_dict = {
            'mean': noise_stats.mean().item() * 1e3,
            'std': noise_stats.std().item() * 1e3,
            'm_abs': noise_stats.abs().mean().item() * 1e3,
            'l2': noise_stats.view(batch_size, -1).norm(p=2, dim=1).mean().item()
        }

        # 额外打印一份 speaker logits 统计，方便你判断是否还异常
        spk_logits = pred_dict['speaker']
        spk_stat = {
            'spk_logit_mean': spk_logits.mean().item(),
            'spk_logit_std': spk_logits.std().item(),
            'spk_logit_min': spk_logits.min().item(),
            'spk_logit_max': spk_logits.max().item(),
        }

        loss_dict.update(
            err_spk=err_spk,
            err_det=err_det,
            **noise_dict,
            **grad_dict,
            **spk_stat
        )

        rtn = {
            'output': f"loss_total:{loss_total.item():6.3f} loss:{loss_dict}",
            'vars': loss_dict,
            'count': {k: batch_size for k in loss_dict}
        }
        return rtn

    else:
        model.eval()
        with torch.no_grad():
            delta = project_l2(model(noise) * noise_scale, tau).expand_as(wav_data)
            pout = (wav_data + delta).clamp(-1, 1)

            labels = {
                'speaker': speaker_id,
                'norm': wav_data,
                'detector': torch.ones(batch_size, dtype=torch.long, device=device)
            }

            loss_total, loss_dict, _, pred_dict, label_dict = loss_func(pout, labels)

        pred_spk = pred_dict['speaker'].argmax(dim=1)
        if target < 0:
            err_spk = (pred_spk != label_dict['speaker']).float().mean().item()
        else:
            err_spk = (pred_spk == target).float().mean().item()

        pred_det = pred_dict['detector'].argmax(dim=1)
        err_det = (pred_det == label_dict['detector']).float().mean().item()

        noise_stats = pout - wav_data
        noise_dict = {
            'mean': noise_stats.mean().item() * 1e3,
            'std': noise_stats.std().item() * 1e3,
            'm_abs': noise_stats.abs().mean().item() * 1e3,
            'l2': noise_stats.view(batch_size, -1).norm(p=2, dim=1).mean().item()
        }

        spk_logits = pred_dict['speaker']
        spk_stat = {
            'spk_logit_mean': spk_logits.mean().item(),
            'spk_logit_std': spk_logits.std().item(),
            'spk_logit_min': spk_logits.min().item(),
            'spk_logit_max': spk_logits.max().item(),
        }

        loss_dict.update(
            err_spk=err_spk,
            err_det=err_det,
            **noise_dict,
            **spk_stat
        )

        rtn = {
            'output': f"loss_total:{loss_total.item():6.3f} loss:{loss_dict}",
            'vars': loss_dict,
            'count': {k: batch_size for k in loss_dict}
        }
        return rtn


# =========================================================
# 测试函数（保留句级 sentence_test，和第二份代码一致）
# =========================================================
def test_generator_nonfix_seed(model, detector, file_list, data_folder,
                               speaker_model, label_dict, target,
                               noise_scale, device, tau=0.1, pool_size=5, seed=1234):
    model.eval()
    speaker_model.eval()
    detector.eval()

    noise_dim = model.noise_dim
    generator = torch.Generator(device=device).manual_seed(seed)
    perturbation_pool = []
    with torch.no_grad():
        for _ in range(pool_size):
            noise = torch.randn(1, noise_dim, device=device, generator=generator)
            delta = project_l2(model(noise) * noise_scale, tau)
            perturbation_pool.append(delta.squeeze(0).detach().cpu().numpy())

    bar = tqdm.tqdm(file_list)
    metrics = RunningAverage()

    for fname in bar:
        real_norm, fs, _ = load_audio(os.path.join(data_folder, fname), expected_fs=16000, normalize=True)

        tic = time.perf_counter()
        gen = random.choice(perturbation_pool)
        gen_repeated = repeat_to_length(gen, len(real_norm))
        adv_norm = np.clip(real_norm + gen_repeated, -1, 1)
        elapsed = time.perf_counter() - tic

        snr = SNR(adv_norm, real_norm)
        pesq = PESQ(real_norm, adv_norm, fs)
        metrics.update({'T_per_sample': elapsed}, {'T_per_sample': 1})

        label = label_dict[fname]
        pred_fake = sentence_test(speaker_model, torch.from_numpy(adv_norm).float().to(device))
        pred_fake_det = sentence_test(detector, torch.from_numpy(adv_norm).float().to(device))

        if target != -1:
            success = (pred_fake == target).float().mean().item()
            metrics.update({'success_rate': success}, {'success_rate': 1})

            pred_real = sentence_test(speaker_model, torch.from_numpy(real_norm).float().to(device))
            metrics.update({
                'err_rate_raw': (pred_real != label).float().mean().item(),
                'sr_raw': (pred_real == target).float().mean().item()
            }, {'err_rate_raw': 1, 'sr_raw': 1})
        else:
            fool = (pred_fake != label).float().mean().item()
            metrics.update({'fooling_rate': fool}, {'fooling_rate': 1})

        fool_det = (pred_fake_det != 0).float().mean().item()
        metrics.update({'fooling_rate_of_detector': fool_det}, {'fooling_rate_of_detector': 1})

        bar.set_description(f"SNR:{snr:.2f} PESQ:{pesq:.2f} real/fake:{label}/{pred_fake.item()}")

    avg = metrics.average()
    atc = avg.get('T_per_sample', 0.0)
    aes = avg.get('fooling_rate', 0.0) / (atc + 1e-6)
    print(avg)
    print(f"ATC: {atc*1000:.2f}ms, AES: {aes:.4f}")


# =========================================================
# 主程序
# =========================================================
def get_parser():
    parser = argparse.ArgumentParser("Train/Test Universal Adversarial Perturbation Generator")
    parser.add_argument("--mode", choices=['train', 'test'], default='train')
    parser.add_argument("--dataset", choices=['timit', 'libri'], default='timit')
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--speaker_model", type=str, required=True)
    parser.add_argument("--speaker_cfg", type=str, required=True)
    parser.add_argument("--detector_model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output/generator")
    parser.add_argument("--noise_dim", type=int, default=100)
    parser.add_argument("--frame_dim", type=int, default=3200)
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.1, help="L2 perturbation bound for SE-UAP projection")
    parser.add_argument("--pool_size", type=int, default=5, help="number of pre-generated UAPs for online testing")
    parser.add_argument("--target", type=int, default=-1, help="target speaker id for targeted attack")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--speaker_factor", type=float, default=1.0)
    parser.add_argument("--detector_factor", type=float, default=1.0)
    parser.add_argument("--norm_factor", type=float, default=1000.0)
    parser.add_argument("--norm_clip", type=float, default=0.01)
    parser.add_argument("--margin", type=float, default=100.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--wlen", type=int, default=200, help="frame length in ms")
    parser.add_argument("--frame_mode", choices=['fixed', 'random'], default='random')

    parser.add_argument("--test_ckpt", type=str, help="checkpoint path for testing")
    parser.add_argument("--test_fixed_seed", action='store_true')
    return parser


def main():
    args = get_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 预训练模型
    speaker_model = load_speaker_model(args.speaker_model, args.speaker_cfg, device)
    detector_model = load_detector_model(args.detector_model, device)

    # 数据集
    if args.dataset == 'timit':
        train_dataset = TIMITDetectorDataset(
            args.data_root, train=True, wlen=args.wlen, frame_mode=args.frame_mode
        )
        test_scp = os.path.join(args.data_root, "processed", "test_8_2.scp")
        label_dict_path = os.path.join(args.data_root, "processed", "TIMIT_labels.npy")
        data_folder = args.data_root
    else:
        train_dataset = LibriSpeechDetectorDataset(
            args.data_root, train=True, wlen=args.wlen, frame_mode=args.frame_mode
        )
        test_scp = os.path.join(args.data_root, "Librispeech_spkid_sel/split_spk_list", "libri_te_8_2.scp")
        label_dict_path = os.path.join(args.data_root, "Librispeech_spkid_sel/split_spk_list", "libri_dict.npy")
        data_folder = os.path.join(args.data_root, "Librispeech_spkid_sel")

    test_file_list = read_list(test_scp)
    label_dict = np.load(label_dict_path, allow_pickle=True).item()

    if args.mode == 'train':
        if args.target < 0:
            spk_loss = SpeakerLoss(margin=args.margin)
        else:
            spk_loss = SpeakerLossTarget(args.target, margin=args.margin)

        det_loss = DetectorLoss(beta=args.beta)
        norm_loss = MSEWithThreshold(threshold=args.norm_clip)

        loss_all = {
            'speaker': {
                'model': speaker_model,
                'factor': args.speaker_factor,
                'loss_func': spk_loss
            },
            'detector': {
                'model': detector_model,
                'factor': args.detector_factor,
                'loss_func': det_loss
            },
            'norm': {
                'loss_func': norm_loss,
                'factor': args.norm_factor
            }
        }

        cost = UniversalLoss(loss_all)

        model = Generator1D(args.noise_dim, args.frame_dim).to(device)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            worker_init_fn=partial(lambda w, s: np.random.seed(s + w), seed=args.seed)
        )

        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.95, 0.999))
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=1.0)

        log_dir = os.path.join(args.output_dir, f"run_{time.strftime('%Y%m%d_%H%M%S')}")
        writer = SummaryWriter(log_dir)

        trainer = ClassifierTrainer(
            model,
            train_loader,
            optimizer,
            cost,
            batch_process_generator,
            args.output_dir,
            writer=writer,
            eval_every=1,
            print_every=10,
            lr_scheduler=lr_scheduler,
            batch_param={
                'noise_scale': args.noise_scale,
                'target': args.target,
                'tau': args.tau
            },
            device=device
        )

        trainer.run(args.epochs)

        final_ckpt = os.path.join(args.output_dir, 'generator_final.pth')
        save_checkpoint(model, final_ckpt, optimizer=optimizer)
        print(f"Final generator saved to {final_ckpt}")
        writer.close()

    else:
        if not args.test_ckpt:
            raise ValueError("Need --test_ckpt for testing")

        model = Generator1D(args.noise_dim, args.frame_dim).to(device)
        load_checkpoint(model, args.test_ckpt, map_location=device)

        test_generator_nonfix_seed(
            model, detector_model, test_file_list, data_folder,
            speaker_model, label_dict, args.target, args.noise_scale, device,
            tau=args.tau, pool_size=args.pool_size, seed=args.seed
        )


if __name__ == "__main__":
    main()
