#!/bin/sh

CUDA_VISIBLE_DEVICES="" python -m maddpg_o.experiments.train_epc \
    --scenario=adversarial \
    --sight=100.0 \
    --alpha=1.0 \
    --initial-population=9 \
    --num-selection=2 \
    --num-stages=3 \
    --test-num-episodes=2000 \
    --stage-num-episodes 50000 20000 20000 \
    --num-good=4 \
    --num-adversaries=4 \
    --num-units=32 \
    --checkpoint-rate=0 \
    --good-policy=att-maddpg \
    --adv-policy=att-maddpg \
    --good-share-weights \
    --adv-share-weights \
    --save-dir="./result/adversarial_epc" \
    --save-rate=100 \
    --train-rate=100 \
    --n-cpu-per-agent=40 \
    --stage-n-envs=25 \
    --timeout=0.03
