#!/bin/sh

CUDA_VISIBLE_DEVICES="" python -m maddpg_o.experiments.train_normal \
    --scenario=grassland \
    --sight=100.0 \
    --num-episodes=100000 \
    --num-good=3 \
    --num-adversaries=2 \
    --num-units=32 \
    --checkpoint-rate=2000 \
    --good-share-weights \
    --adv-share-weights \
    --good-policy=att-maddpg \
    --adv-policy=att-maddpg \
    --save-dir="./result/grassland_att_3-2" \
    --save-rate=1000 \
    --train-rate=100 \
    --n-cpu-per-agent=40 \
    --n-envs=100 \
    --timeout=0.03
