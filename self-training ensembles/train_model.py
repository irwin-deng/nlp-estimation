import argparse
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision import datasets

from utils.data_util import *
from utils.lib import *
from models.model import Model
from models.dann_model import DigitsDANNModel, CifarDANNModel, TextDANNModel

def test(model: nn.Module, dataloader: DataLoader):
    model.eval()
    n_correct, n_total = 0, 0
    for img, label in iter(dataloader):
        batch_size = len(label)
        img, label = img.cuda(), label.cuda()

        class_output, _, _ = model(img)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

    acc = n_correct.double() / n_total
    return acc

def train(model: nn.Module, dataloader_source: DataLoader, alpha: float,
          optimizer: torch.optim.Optimizer, loss_class: torch.nn.modules.loss._Loss) -> None:
    model.train()
    for s_img, s_label in iter(dataloader_source):
        s_img, s_label = s_img.cuda(), s_label.cuda()

        class_output, _, _ = model(s_img, alpha)
        loss_s_label = loss_class(class_output, s_label)

        loss = loss_s_label
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    parser = argparse.ArgumentParser(description='Pretrain model')
    parser.add_argument('--source-dataset', default='mnist', type=str, help='source dataset')
    parser.add_argument('--batch-size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--nepoch', default=20, type=int, help='Number of epochs for training')
    parser.add_argument('--model-type', default="typical_dnn", choices=['typical_dnn', "dann_arch"], type=str, help='given model type')
    parser.add_argument('--base-dir', default='./checkpoints/source_models/', type=str, help='dir to save model')
    parser.add_argument('--seed', type=int, default=0)
    
    # args parse
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    batch_size, nepoch = args.batch_size, args.nepoch
    source_dataset: str = args.source_dataset
    model_type = args.model_type
    
    if not os.path.exists(args.base_dir):
        os.makedirs(args.base_dir)
    save_dir = os.path.join(args.base_dir, source_dataset)

    def img_transform(size: int):
        return transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
        ])

    if source_dataset == "mnist":
        normalizer = transforms.Normalize(mean=(0.1307, 0.1307, 0.1307), std=(0.3015, 0.3015, 0.3015))
        dataset_source = datasets.MNIST(root='dataset/mnist', train=True, transform=img_transform(28), download=True)
        dataloader_source = DataLoader(dataset=dataset_source, batch_size=batch_size, shuffle=True, num_workers=2)
        dataset_source_test = datasets.MNIST(root='dataset/mnist', train=False, transform=img_transform(28), download=True)
        dataloader_source_test = DataLoader(dataset=dataset_source_test, batch_size=batch_size, shuffle=False, num_workers=2)
    elif source_dataset == "mnist-m":
        normalizer = transforms.Normalize(mean=(0.4582, 0.4623, 0.4085), std=(0.1722, 0.1603, 0.1787))
        train_list = os.path.join('dataset/mnist_m/mnist_m_train_labels.txt')
        test_list = os.path.join('dataset/mnist_m/mnist_m_test_labels.txt')
        dataset_source = MNISTM(data_root='dataset/mnist_m/mnist_m_train', data_list=train_list, transform=img_transform(28))
        dataloader_source = DataLoader(dataset=dataset_source, batch_size=batch_size, shuffle=True, num_workers=2)
        dataset_source_test =  MNISTM(data_root='dataset/mnist_m/mnist_m_test', data_list=test_list, transform=img_transform(28))
        dataloader_source_test = DataLoader(dataset=dataset_source_test, batch_size=batch_size, shuffle= False, num_workers=2)
    elif source_dataset == "svhn":
        normalizer = transforms.Normalize(mean=(0.4379, 0.4440, 0.4731), std=(0.1161, 0.1192, 0.1017))
        dataset_source = SVHN('dataset/svhn/', split='train', transform=img_transform(28), download=False)
        dataloader_source = DataLoader(dataset=dataset_source, batch_size=batch_size, shuffle=True, num_workers=2)
        dataset_source_test = SVHN('dataset/svhn/', split='test', transform=img_transform(28), download=False)
        dataloader_source_test = DataLoader(dataset=dataset_source_test, batch_size=batch_size, shuffle=False, num_workers=2)
    elif source_dataset == "usps":
        normalizer = transforms.Normalize(mean=(0.2542, 0.2542, 0.2542), std=(0.3356, 0.3356, 0.3356))
        dataset_source = USPS(split="train", transform=img_transform(28))
        dataloader_source = DataLoader(dataset=dataset_source, batch_size=batch_size, shuffle=True, num_workers=2)
    elif source_dataset.startswith("amazon/"):
        dataset_source = Amazon("dataset/amazon/", split='train', category=source_dataset.removeprefix("amazon/"))
        dataloader_source = DataLoader(dataset=dataset_source, batch_size=batch_size, shuffle=True, num_workers=2)
        dataset_source_test = Amazon("dataset/amazon/", split='test', category=source_dataset.removeprefix("amazon/"))
        dataloader_source_test = DataLoader(dataset=dataset_source_test, batch_size=batch_size, shuffle=True, num_workers=2)
    elif source_dataset == "cifar10":
        normalizer = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))
        dataset = datasets.CIFAR10("dataset/cifar10/", train=True, transform=img_transform(32), download=True)
        # Split into train-validation because cifar10c uses the same test set
        train_ds, valid_ds = random_split(dataset, (0.8, 0.2), torch.Generator().manual_seed(0))
        dataloader_source = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
        dataloader_source_test = DataLoader(dataset=valid_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    else:
        raise NotImplementedError()

    # Model Setup
    if model_type == "typical_dnn":
        model = Model(normalizer=normalizer).cuda()
    elif model_type == "dann_arch":
        if source_dataset.startswith("amazon/"):
            model = TextDANNModel().cuda()
        elif source_dataset.startswith("cifar10"):
            model = CifarDANNModel(normalizer=normalizer).cuda()
        else:
            model = DigitsDANNModel(normalizer=normalizer).cuda()
    else:
        raise NotImplementedError()
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_class = torch.nn.NLLLoss().cuda()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(nepoch):
        alpha = (2 / (1 + np.exp(-10 * (epoch / (nepoch - 1)))) - 1) * 0.1 if nepoch > 1 else 0.1
        train(model, dataloader_source, alpha, optimizer, loss_class)
        acc_s = test(model, dataloader_source_test)
        print('EPOCH {} Acc: {} {:.2f}%'.format(epoch, source_dataset, acc_s*100))

    torch.save(model, os.path.join(save_dir, 'checkpoint.pth'))


if __name__ == '__main__':
    main()
