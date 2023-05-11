#!/bin/bash

gpu_id=0

for dataset in amazon/apparel amazon/books amazon/dvd amazon/electronics amazon/health amazon/kitchen amazon/music amazon/sports amazon/toys amazon/video
do
  for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
  do
    CUDA_VISIBLE_DEVICES=$gpu_id python train_model.py --source-dataset $dataset --model-type dann_arch --base-dir checkpoints/ensemble_dann_arch_source_models/$i/ --seed $(( 100*(i+1) ))
  done
done
