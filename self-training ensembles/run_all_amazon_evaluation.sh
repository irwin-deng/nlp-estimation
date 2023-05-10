#!/bin/bash

gpu_id=0

for model_type in dann_arch
do
  for method in acc_weighted_ri sim_weighted_ri our_ri
  do
    CUDA_VISIBLE_DEVICES=$gpu_id python eval_pipeline.py --model-type $model_type --method $method \
        --datasets amazon/apparel amazon/books amazon/dvd amazon/electronics amazon/health amazon/kitchen amazon/music amazon/sports amazon/toys amazon/video
  done
done
