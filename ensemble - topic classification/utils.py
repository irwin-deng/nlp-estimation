import random
import numpy as np
import torch
import torch.backends.cudnn
from torch.utils.data import Dataset
from typing import Any, Callable, Union


def set_seed(seed: int) -> None:
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class TensorDictDataset(Dataset):
    n_examples: int
    name: str
    data: dict[str, torch.Tensor]

    def __init__(self, name: str, data: dict[str, torch.Tensor]):
        self.name = name
        if len(data) == 0:
            raise AssertionError(f"Dataset '{name}' is missing data")
        self.n_examples = next(iter(data.values())).shape[0]
        for col, values in data.items():
            if isinstance(values, torch.Tensor):
                if self.n_examples != values.shape[0]:
                    raise AssertionError(
                        f"Shape of {col} {values.shape} "
                        f"incompatible with n_examples {self.n_examples}")
            else:
                raise AssertionError(f"Invalid type of {col}: {type(values)}")

        self.data = dict(data.items())

    def __getitem__(self, indx: Union[int, slice]) -> dict[str, torch.Tensor]:
        return {col: values[indx] for (col, values) in self.data.items()}

    def __len__(self) -> int:
        return self.n_examples

    def to(self, device: torch.device):
        for key in self.data:
            print(key)
            self.data[key] = self.data[key].to(device)

    def cpu(self):
        for key in self.data:
            self.data[key] = self.data[key].cpu()

    def get_copy(self) -> "TensorDictDataset":
        return TensorDictDataset(
            self.name,
            data={col: tensor.clone().detach() for (col, tensor) in self.data.items()}
        )


def get_batch_eval_metrics(target: torch.Tensor, output: torch.Tensor,
        loss_func: Callable[[torch.Tensor, torch.Tensor], float] = None) -> dict[str, Any]:
    """
    Returns a dict of evaluation metrics for a classification model given a
    target tensor and the model's output tensor
    """
    metrics = {}

    current_batch_size = len(target)
    metrics["batch_size"] = current_batch_size

    predictions = output.argmax(dim=1, keepdim=True).squeeze()
    n_correct = torch.sum(predictions == target)
    accuracy = n_correct / current_batch_size
    metrics["accuracy"] = accuracy

    if loss_func is not None:
        loss = loss_func(output, target)
        metrics["loss"] = loss

    return metrics
