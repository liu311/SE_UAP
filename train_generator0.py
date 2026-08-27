# train_generator.py
import sys
sys.path.append('/home/chenyuying/Projects/me')
import os
import time
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from copy import deepcopy
import tqdm
from functools import partial
from common.dataset import TIMITDetectorDataset, LibriSpeechDetectorDataset
from common.trainer import ClassifierTrainer, load_checkpoint, save_checkpoint
from common.utils import read_conf, get_dict_from_args
from generator import Generator1D
# 文件顶部缺少这些导入
import soundfile as sf
import torch.nn.functional as F
from adversarial_utils import (
    AdversarialDetectorCNN, load_speaker_model, load_detector_model,
    sentence_test, preprocess, SNR, PESQ, read_list
)
class RunningAverage:
    def __init__(self):
        self.sums = {}
        self.counts = {}

    def update(self, vals: dict, counts: dict):
        for k, v in vals.items():
            self.sums[k] = self.sums.get(k, 0) + v * counts[k]
            self.counts[k] = self.counts.get(k, 0) + counts[k]

    def average(self):
        return {k: self.sums[k] / self.counts[k] for k in self.sums}
# -------------------- 损失函数定义（与原代码一致） --------------------
class SpeakerLoss(nn.Module):
    def __init__(self, margin=100):
        super().__init__()
        self.margin = margin

    def forward(self, logits, label):
        # logits: model output (batch, n_speakers)
        B = len(label)
        print("logits------------",logits)
        correct = logits[range(B), label]
        max_other = torch.max(logits * (1 - torch.nn.functional.one_hot(label, logits.size(-1)).float()), dim=1)[0]
        loss = (correct - max_other + self.margin).clamp(min=0) / self.margin
        return loss.mean()

class SpeakerLossTarget(nn.Module):
    def __init__(self, target, margin=100):
        super().__init__()
        self.target = target
        self.margin = margin

    def forward(self, logits, label):
        f_target = logits[:, self.target]
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[:, self.target] = False
        max_other = torch.max(logits.masked_fill(~mask, -1e9), dim=1)[0]
        loss = (max_other - f_target + self.margin).clamp(min=0)
        return loss.mean()

class DetectorLoss(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, label):
        x = self.ce(logits, label)
        bx = torch.clamp(self.beta * x, max=20.0)
        return (torch.exp(bx) - 1).mean()

class MSEWithThreshold(nn.Module):
    def __init__(self, threshold=0.05):
        super().__init__()
        self.threshold = threshold

    def forward(self, pred, label):
        err = pred - label
        err_abs = torch.abs(err)
        err_abs = err_abs - self.threshold
        err_abs[err_abs < 0] = 0
        if (err_abs > 0).sum() == 0:
            return err_abs.sum()
        else:
            return err_abs.sum() / (err_abs > 0).sum()

class UniversalLoss:
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
            else:
                pred_this = model(pred)
            label_this = labels[key]
            loss = cfg['loss_func'](pred_this, label_this) * cfg['factor']
            loss_dict[key] = loss.detach().item()
            pred_dict[key] = pred_this.detach()
            label_dict[key] = label_this.detach()
            loss_tensor_dict[key] = loss  # 保留tensor用于backward
            loss_dict[key] = loss.detach().item()  # 标量用于logging
        loss_total = sum(loss_tensor_dict.values())  # tensor相加，有梯度
        loss_dict['total'] = loss_total.detach().item()
        return loss_total, loss_dict, None, pred_dict, label_dict

# -------------------- 数据处理函数（训练/评估共用） --------------------
grads = {}
def save_grad(v):
    def hook(grad):
        grads[v] = grad
    return hook

def batch_process_generator(model, data, train_mode=True, **kwargs):
    wav_data, speaker_id, norm_factor = data
    device = next(model.parameters()).device          # 从模型获取 device
    wav_data   = wav_data.float().to(device)          # (B, 3200)
    speaker_id = speaker_id.long().to(device)
    norm_factor = norm_factor.float().to(device)

    batch_size = wav_data.shape[0]
    noise = torch.randn(batch_size, model.noise_dim, device=device)  # 改用 device
    noise_scale = kwargs.get('noise_scale', 1)
    target = kwargs.get('target', -1)
    norm_factor = norm_factor.unsqueeze(1).repeat(1, wav_data.shape[1]).to(device)

    if train_mode:
        model.train()
        optimizer = kwargs['optimizer']
        loss_func = kwargs['loss_func']

        pout = model(noise)
        pout = (pout * noise_scale + wav_data * norm_factor).clamp(-1, 1) / norm_factor
        pout.register_hook(save_grad('wav_d'))

        labels = {
            'speaker': speaker_id,
            'norm': wav_data,
            'detector': torch.ones(batch_size, dtype=torch.long, device=wav_data.device)
        }
        loss_total, loss_dict, _, pred_dict, label_dict = loss_func(pout, labels)

        model.zero_grad()
        loss_total.backward()
        grad_dict = {'total': grads['wav_d'].abs().mean().item()}
        optimizer.step()

        pred = pred_dict['speaker'].argmax(1)
        if target < 0:
            err_spk = (pred != label_dict['speaker']).float().mean().item()
        else:
            err_spk = (pred == target).float().mean().item()
        err_det = (pred_dict['detector'].argmax(1) == label_dict['detector']).float().mean().item()

        noise_stats = pout - wav_data
        noise_dict = {
            'mean': noise_stats.mean().item() * 1e3,
            'std': noise_stats.std().item() * 1e3,
            'm_abs': noise_stats.abs().mean().item() * 1e3
        }
        loss_dict.update(err_spk=err_spk, err_det=err_det, **noise_dict, **grad_dict)

        rtn = {
            'output': f"loss_total:{loss_total.item():6.3f} loss:{loss_dict}",
            'vars': loss_dict,
            'count': {k: batch_size for k in loss_dict}
        }
    else:
        model.eval()
        with torch.no_grad():
            pout = model(noise)
            pout = (pout * noise_scale + wav_data * norm_factor).clamp(-1, 1) / norm_factor
            labels = {
                'speaker': speaker_id,
                'norm': wav_data,
                'detector': torch.ones(batch_size, dtype=torch.long, device=wav_data.device)
            }
            loss_total, loss_dict, _, pred_dict, label_dict = loss_func(pout, labels)

        pred = pred_dict['speaker'].argmax(1)
        if target < 0:
            err_spk = (pred != label_dict['speaker']).float().mean().item()
        else:
            err_spk = (pred == target).float().mean().item()
        err_det = (pred_dict['detector'].argmax(1) == label_dict['detector']).float().mean().item()
        noise_stats = pout - wav_data
        noise_dict = {
            'mean': noise_stats.mean().item() * 1e3,
            'std': noise_stats.std().item() * 1e3,
            'm_abs': noise_stats.abs().mean().item() * 1e3
        }
        loss_dict.update(err_spk=err_spk, err_det=err_det, **noise_dict)
        rtn = {
            'output': f"loss_total:{loss_total.item():6.3f} loss:{loss_dict}",
            'vars': loss_dict,
            'count': {k: batch_size for k in loss_dict}
        }
    return rtn

# -------------------- 评估函数（复用 adversarial_utils 中的 sentence_test） --------------------
def test_generator_nonfix_seed(model, detector, file_list, data_folder, speaker_model, label_dict, target, noise_scale, device):
    model.eval()
    speaker_model.eval()
    detector.eval()
    noise_dim = model.noise_dim
    bar = tqdm.tqdm(file_list)
    metrics = RunningAverage()

    for fname in bar:
        noise = torch.randn(1, noise_dim, device=device)
        real, fs = sf.read(os.path.join(data_folder, fname))
        real_norm, _ = preprocess(real)
        gen = model(noise).squeeze().detach().cpu().numpy()
        gen_repeated = np.tile(gen, int(np.ceil(len(real) / len(gen))))[:len(real)]
        adv = np.clip(gen_repeated * noise_scale + real, -1, 1)
        adv_norm, _ = preprocess(adv)

        snr = SNR(adv, real)
        pesq = PESQ(real, adv, fs)
        label = label_dict[fname]
        pred_fake = sentence_test(speaker_model, torch.from_numpy(adv_norm).float().to(device))
        pred_fake_det = sentence_test(detector, torch.from_numpy(adv_norm).float().to(device))

        if target != -1:
            success = (pred_fake == target).float().mean().item()
            metrics.update({'success_rate': success}, {'success_rate': 1})
            pred_real = sentence_test(speaker_model, torch.from_numpy(real_norm).float().to(device))
            metrics.update({'err_rate_raw': (pred_real != label).float().mean().item(),
                            'sr_raw': (pred_real == target).float().mean().item()},
                           {'err_rate_raw': 1, 'sr_raw': 1})
        else:
            fool = (pred_fake != label).float().mean().item()
            metrics.update({'fooling_rate': fool}, {'fooling_rate': 1})

        fool_det = (pred_fake_det != 0).float().mean().item()
        metrics.update({'fooling_rate_of_detector': fool_det}, {'fooling_rate_of_detector': 1})
        bar.set_description(f"SNR:{snr:.2f} PESQ:{pesq:.2f} real/fake:{label}/{pred_fake.item()}")

    avg = metrics.average()
    atc = avg.get('T_per_sample', 0)  # 需要自己计时，此处略
    aes = avg.get('fooling_rate', 0) / (atc + 1e-6)
    print(avg)
    print(f"ATC: {atc*1000:.2f}ms, AES: {aes:.4f}")

# -------------------- 主程序 --------------------
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
    parser.add_argument("--target", type=int, default=-1, help="target speaker id for targeted attack")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--speaker_factor", type=float, default=1.0)
    parser.add_argument("--detector_factor", type=float, default=1.0)
    parser.add_argument("--norm_factor", type=float, default=1000.0)
    parser.add_argument("--norm_clip", type=float, default=0.01)
    parser.add_argument("--margin", type=int, default=100)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--wlen", type=int, default=200,
                        help="frame length in ms (200ms = 3200 samples @ 16kHz)")
    parser.add_argument("--frame_mode", choices=['fixed', 'random'], default='random')
    # 测试专用
    parser.add_argument("--test_ckpt", type=str, help="checkpoint path for testing")
    parser.add_argument("--test_fixed_seed", action='store_true')
    return parser

def main():
    args = get_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 加载预训练模型
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
        # 构建损失函数
        if args.target < 0:
            spk_loss = SpeakerLoss(margin=args.margin)
        else:
            SpeakerLossTarget(args.target, margin=args.margin)
        det_loss = DetectorLoss(beta=args.beta)
        norm_loss = MSEWithThreshold(threshold=args.norm_clip)

        loss_all = {
            'speaker': {'model': speaker_model, 'factor': args.speaker_factor, 'loss_func': spk_loss},
            'detector': {'model': detector_model, 'factor': args.detector_factor, 'loss_func': det_loss},
            'norm': {'loss_func': norm_loss, 'factor': args.norm_factor}
        }
        cost = UniversalLoss(loss_all)

        # 生成器
        model = Generator1D(args.noise_dim, args.frame_dim).to(device)

        # DataLoader
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=0, pin_memory=True,
                                  worker_init_fn=partial(lambda w, s: np.random.seed(s + w), seed=args.seed))

        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.95, 0.999))
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=1.0)

        log_dir = os.path.join(args.output_dir, f"run_{time.strftime('%Y%m%d_%H%M%S')}")
        writer = SummaryWriter(log_dir)

        trainer = ClassifierTrainer(
            model, train_loader, optimizer, cost, batch_process_generator, args.output_dir,
            writer=writer, eval_every=1, print_every=10,
            lr_scheduler=lr_scheduler, batch_param={'noise_scale': args.noise_scale, 'target': args.target},
            device=device
        )
        trainer.run(args.epochs)
        # 训练结束后，显式保存最终模型
        final_ckpt = os.path.join(args.output_dir, 'generator_final.pth')
        save_checkpoint(model, final_ckpt, optimizer=optimizer)
        print(f"Final generator saved to {final_ckpt}")
        writer.close()

    else:  # test mode
        if not args.test_ckpt:
            raise ValueError("Need --test_ckpt for testing")
        model = Generator1D(args.noise_dim, args.frame_dim).to(device)
        load_checkpoint(model, args.test_ckpt, map_location=device)
        test_generator_nonfix_seed(model, detector_model, test_file_list, data_folder,
                                       speaker_model, label_dict, args.target, args.noise_scale, device)
        # if args.test_fixed_seed:
        #     for seed in range(1, 6):
        #         torch.manual_seed(seed)
        #         test_generator_fixed_seed(model, seed, detector_model, test_file_list, data_folder,
        #                                   speaker_model, label_dict, args.target, args.noise_scale, device)
        # else:
        #     test_generator_nonfix_seed(model, detector_model, test_file_list, data_folder,
        #                                speaker_model, label_dict, args.target, args.noise_scale, device)

# 需要补充 test_generator_fixed_seed 和 RunningAverage 等，此处略（原代码已有）
# 实际使用时请将原 test_fixed_seed 函数搬移过来，并导入 tqdm, RunningAverage 等

if __name__ == "__main__":
    main()
