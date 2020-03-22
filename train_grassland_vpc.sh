#!/bin/sh

CUDA_VISIBLE_DEVICES="" python -m maddpg_o.experiments.train_epc \
    --scenario=grassland \
    --sight=100.0 \
    --initial-population=1 \
    --num-selection=1 \
    --num-stages=4 \
    --test-num-episodes=2000 \
    --stage-num-episodes 100000 50000 50000 50000 \
    --num-good=3 \
    --num-adversaries=2 \
    --num-units=32 \
    --checkpoint-rate=0 \
    --good-policy=att-maddpg \
    --adv-policy=att-maddpg \
    --good-share-weights \
    --adv-share-weights \
    --save-dir="./result/grassland_vpc" \
    --save-rate=100 \
    --train-rate=100 \
    --n-cpu-per-agent=40 \
    --stage-n-envs=25 \
    --timeout=0.03
