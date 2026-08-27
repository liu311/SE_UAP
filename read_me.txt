cd
# 针对libri数据集，进行fgsm对抗训练，得到预训练的检测器，用于后续的SE_UAP攻击。
python train_detector.py \
    --dataset libri \
    --data_root /Data/chenyuying \
    --speaker_model ./speaker_model/SincNet_LIBRI/epoch_23.pth \
    --speaker_cfg ./config/libri_speaker_generator.cfg \
    --attack fgsm \
    --epsilon 0.01 \
    --batch_size 16 \
    --epochs 10 \
    --lr 0.001 \
    --wlen 200 \
    --output_dir ./output/detector_libri \
    --seed 1234

# 针对TIMIT数据集，进行fgsm对抗训练，得到预训练的检测器，用于后续的SE_UAP攻击。
python train_detector.py \
    --dataset timit \
    --data_root /Data/chenyuying/TIMIT/ \
    --speaker_model ./output/SincNet_TIMIT/model_raw.pkl \
    --speaker_cfg ./config/timit_speaker_generator.cfg \
    --attack fgsm \
    --epsilon 0.01 \
    --batch_size 16 \
    --epochs 20 \
    --lr 0.001 \
    --wlen 200 \
    --output_dir ./output/detector \
    --seed 1234

# 针对TIMIT数据集，基于我们提出的SE_UAP进行对抗训练，，得到generator

# 训练
CUDA_VISIBLE_DEVICES=0 python train_generator.py \
    --mode train \
    --dataset timit \
    --data_root /Data/chenyuying/TIMIT/ \
    --speaker_model ./speaker_model/SincNet_TIMIT/model_raw.pkl \
    --speaker_cfg ./config/timit_speaker_generator.cfg \
    --detector_model ./output/detector/detector_timit_fgsm_noovs.pth \
    --output_dir ./output/generator_timit \
    --wlen 200 \
    --frame_mode random \
    --noise_dim 100 \
    --frame_dim 3200 \
    --epochs 50 \
    --batch_size 128 \
    --lr 0.001 \
    --speaker_factor 1.0 \
    --detector_factor 0.5 \
    --norm_factor 1000.0 \
    --norm_clip 0.01 \
    --beta 1.0 \
    --seed 1234
# 测试
CUDA_VISIBLE_DEVICES=0 python train_generator.py \
    --mode test \
    --dataset timit \
    --data_root /Data/chenyuying/TIMIT/ \
    --speaker_model ./speaker_model/SincNet_TIMIT/model_raw.pkl \
    --speaker_cfg ./config/timit_speaker_generator.cfg \
    --detector_model ./output/detector/detector_timit_fgsm_noovs.pth \
    --test_ckpt ./output/generator_timit/generator_final.pth \
    --wlen 200 \
    --frame_mode random \
    --noise_dim 100 \
    --frame_dim 3200 \
    --seed 1234
# 针对libri数据集，基于我们提出的SE_UAP进行对抗训练，，得到generator
# 训练
CUDA_VISIBLE_DEVICES=0 python train_generator.py \
    --mode train \
    --dataset libri \
    --data_root /Data/chenyuying \
    --speaker_model ./speaker_model/SincNet_LIBRI/epoch_23.pth \
    --speaker_cfg ./config/libri_speaker_generator.cfg \
    --detector_model ./output/detector_libri/detector_libri_fgsm_noovs.pth \
    --output_dir ./output/generator_libri \
    --wlen 200 \
    --frame_mode random \
    --noise_dim 100 \
    --frame_dim 3200 \
    --epochs 50 \
    --batch_size 128 \
    --lr 0.001 \
    --speaker_factor 1.0 \
    --detector_factor 0.5 \
    --norm_factor 1000.0 \
    --norm_clip 0.01 \
    --beta 1.0 \
    --seed 1234
# 测试
CUDA_VISIBLE_DEVICES=0 python train_generator.py \
    --mode test \
    --dataset libri \
    --data_root /Data/chenyuying \
    --speaker_model ./speaker_model/SincNet_LIBRI/epoch_23.pth \
    --speaker_cfg ./config/libri_speaker_generator.cfg \
    --detector_model ./output/detector_libri/detector_libri_fgsm_noovs.pth \
    --test_ckpt ./output/generator_libri/generator_final.pth \
    --wlen 200 \
    --frame_mode random \
    --noise_dim 100 \
    --frame_dim 3200 \
    --seed 1234

python test_speaker_model.py \
    --speaker_model ./speaker_model/SincNet_LIBRI/epoch_23.pth \
    --speaker_cfg ./config/libri_speaker_generator.cfg \
    --test_scp /Data/chenyuying/TIMIT/processed/test_8_2.scp \
    --label_dict /Data/chenyuying/TIMIT/processed/TIMIT_labels.npy \
    --data_root /Data/chenyuying/TIMIT \
    --device cpu

python test_speaker_model.py \
    --speaker_model /path/to/speaker.pt \
    --speaker_cfg /path/to/config.yaml \
    --test_scp /Data/chenyuying/Librispeech_spkid_sel/split_spk_list/libri_te_8_2.scp \
    --label_dict /Data/chenyuying/Librispeech_spkid_sel/split_spk_list/libri_dict.npy \
    --data_root /Data/chenyuying/Librispeech_spkid_sel \
    --device cuda