#!/bin/bash

gpu_id=0

for model_type in dann_arch
do
  for method in our_ri acc_weighted_ri sim_weighted_ri
  do
    CUDA_VISIBLE_DEVICES=$gpu_id python eval_pipeline.py --model-type $model_type --method $method \
        --source-datasets cifar10 \
        --target-datasets cifar10c/brightness cifar10c/contrast cifar10c/defocus_blur cifar10c/elastic_transform \
            cifar10c/fog cifar10c/frost cifar10c/gaussian_blur cifar10c/gaussian_noise cifar10c/glass_blur \
            cifar10c/impulse_noise cifar10c/jpeg_compression cifar10c/motion_blur cifar10c/pixelate \
            cifar10c/saturate cifar10c/shot_noise cifar10c/snow cifar10c/spatter cifar10c/speckle_noise cifar10c/zoom_blur
  done
done
