#!/bin/bash

gpu_id=0

for source in amazon/apparel amazon/books amazon/dvd amazon/electronics amazon/health amazon/kitchen amazon/music amazon/sports amazon/toys amazon/video
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train_model.py --source-dataset $source --model-type dann_arch --base-dir ./checkpoints/dann_arch_source_models/
done
