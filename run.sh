#!/usr/bin/env
# python3 train.py --resume --dataset celeb_a --filters 16 --checkpoint_freq 1000 --batch_size 128

python main.py --experiment generate --filters 16 --checkpoint_dir ./model/permanent_saved_models
