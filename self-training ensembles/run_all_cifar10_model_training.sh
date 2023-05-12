#!/bin/bash

gpu_id=0

CUDA_VISIBLE_DEVICES=$gpu_id python train_model.py --source-dataset "cifar10" --model-type dann_arch --base-dir ./checkpoints/dann_arch_source_models/
