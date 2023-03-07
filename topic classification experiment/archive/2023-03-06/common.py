import torch
import transformers

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA device {torch.cuda.get_device_name(device)}")
else:
    device = torch.device("cpu")
    print("Using CPU")
