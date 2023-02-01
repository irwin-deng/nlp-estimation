import torch
import transformers

torch.cuda.empty_cache()
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA device {torch.cuda.get_device_name(device)}")
else:
    device = torch.device("cpu")
    print("Using CPU")


bert_base = transformers.AutoModel.from_pretrained("bert-base-uncased")
bert_base.cpu()
bert_base.eval()
