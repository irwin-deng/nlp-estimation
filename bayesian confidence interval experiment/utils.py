import torch
from typing import Any, Callable
from common import device


def get_batch_eval_metrics(target: torch.tensor, output: torch.tensor,
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
