import torch
from typing import Any, Callable, Optional, Union
from common import device

class TensorDictDataset(dict):
    n_examples: int
    name: str

    def __init__(self, name: str, dataset: dict[str, Union[torch.Tensor, list[str]]]):
        self.name = name
        assert "labels" in dataset
        assert isinstance(dataset["labels"], torch.Tensor)
        self.n_examples = dataset["labels"].shape[0]
        for feature, values in dataset.items():
            if isinstance(values, torch.Tensor):
                if self.n_examples != values.shape[0]:
                    raise AssertionError(f"Shape of {feature} {values.shape} "
                        f"incompatible with n_examples {self.n_examples}")
            elif isinstance(values, list):
                if self.n_examples != len(values):
                    raise AssertionError(f"Length of {feature} {len(values)} "
                        f"does not equal n_examples {self.n_examples}")
            else:
                raise AssertionError(f"Invalid type of {feature}: {type(values)}")
        super().__init__(dataset)

    def __getitem__(self, feature: str) -> Union[torch.Tensor, list[str]]:
        return super().__getitem__(feature)

    def __setitem__(self, feature: str, values: Union[torch.Tensor, list[str]]) -> None:
        if isinstance(values, torch.Tensor):
            if self.n_examples != values.shape[0]:
                raise AssertionError(f"Shape of {feature} {values.shape} "
                    f"incompatible with n_examples {self.n_examples}")
        elif isinstance(values, list):
            if self.n_examples != len(values):
                raise AssertionError(f"Length of {feature} {len(values)} "
                    f"does not equal n_examples {self.n_examples}")
        else:
            raise AssertionError(f"Invalid type of {feature}: {type(values)}")
        super().__setitem__(feature, values)

    def __delitem__(self, feature: str) -> None:
        return super().__delitem__(feature)

    def __len__(self) -> int:
        return self.n_examples


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


def shuffle_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Create a new tensor that contains the randomly shuffled values of the input tensor
    """
    flattened_tensor = tensor.flatten()
    shuffled_indices = torch.randperm(torch.numel(tensor), device=device)
    return flattened_tensor[shuffled_indices].reshape(tensor.size())
