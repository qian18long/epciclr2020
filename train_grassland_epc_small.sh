#!/bin/sh

CUDA_VISIBLE_DEVICES="" python -m maddpg_o.experiments.train_epc \
    --sight=100.0 \
    --initial-population=2 \
    --num-selection=2 \
    --num-stages=3 \
    --test-num-episodes=10 \
    --stage-num-episodes=10 \
    --num-good=3 \
    --num-adversaries=2 \
    --num-units=2 \
    --checkpoint-rate=0 \
    --good-policy=att-maddpg \
    --adv-policy=att-maddpg \
    --good-share-weights \
    --adv-share-weights \
    --save-dir="./result/grassland_epc_test" \
    --save-rate=10 \
    --train-rate=10 \
    --n-cpu-per-agent=40 \
    --stage-n-envs=1 \
    --timeout=0.03
