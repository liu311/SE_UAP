import numpy as np
import os

def split_and_save(files, train_path, test_path, ratio=0.8, seed=1234):
    rng = np.random.default_rng(seed)
    files = sorted(files)           # 排序保证可重现
    rng.shuffle(files)
    n_train = int(len(files) * ratio)
    train_files = files[:n_train]
    test_files  = files[n_train:]
    with open(train_path, 'w') as f:
        f.write('\n'.join(train_files) + '\n')
    with open(test_path, 'w') as f:
        f.write('\n'.join(test_files) + '\n')
    print(f"Total: {len(files)} → Train: {len(train_files)}, Test: {len(test_files)}")

# ── TIMIT ──────────────────────────────────────────────
timit_processed = "/Data/chenyuying/TIMIT/processed"

with open(os.path.join(timit_processed, "train.scp")) as f:
    timit_train = [l.strip() for l in f if l.strip()]
with open(os.path.join(timit_processed, "test.scp")) as f:
    timit_test  = [l.strip() for l in f if l.strip()]

timit_all = timit_train + timit_test
print(f"TIMIT total: {len(timit_all)}")

split_and_save(
    timit_all,
    os.path.join(timit_processed, "train_8_2.scp"),
    os.path.join(timit_processed, "test_8_2.scp"),
)

# ── LibriSpeech ────────────────────────────────────────
libri_processed = "/Data/chenyuying/Librispeech_spkid_sel/split_spk_list"

with open(os.path.join(libri_processed, "libri_tr.scp")) as f:
    libri_train = [l.strip() for l in f if l.strip()]
with open(os.path.join(libri_processed, "libri_te.scp")) as f:
    libri_test  = [l.strip() for l in f if l.strip()]

libri_all = libri_train + libri_test
print(f"LibriSpeech total: {len(libri_all)}")

split_and_save(
    libri_all,
    os.path.join(libri_processed, "libri_tr_8_2.scp"),
    os.path.join(libri_processed, "libri_te_8_2.scp"),
)