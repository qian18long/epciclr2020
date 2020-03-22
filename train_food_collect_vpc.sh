#!/bin/sh

CUDA_VISIBLE_DEVICES="" python -m maddpg_o.experiments.train_epc \
    --scenario=food_collect \
    --sight=100.0 \
    --cooperative \
    --initial-population=1 \
    --num-selection=1 \
    --num-stages=4 \
    --test-num-episodes=2000 \
    --stage-num-episodes 50000 20000 20000 20000 \
    --num-good=3 \
    --num-adversaries=0 \
    --num-food=3 \
    --num-units=32 \
    --checkpoint-rate=0 \
    --good-policy=att-maddpg \
    --adv-policy=att-maddpg \
    --good-share-weights \
    --adv-share-weights \
    --save-dir="./result/food_collect_vpc" \
    --save-rate=100 \
    --train-rate=100 \
    --n-cpu-per-agent=40 \
    --stage-n-envs=25 \
    --timeout=0.03
