import argparse
import random
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision import transforms
import torch.nn.functional as F

os.environ["TOKENIZERS_PARALLELISM"] = "true"

from utils.data_util import *
from utils.lib import *
from utils.trust_score import TrustScore
from utils.similarity import get_distance_matrix

    
def get_dataloaders(source_dataset, target_dataset, batch_size, test_time):
    def img_transform(size: int):
        return transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
        ])

    if source_dataset == "mnist":
        dataset_source = datasets.MNIST(root='dataset/mnist', train=True, transform=img_transform(28), download=True)
        dataloader_source = DataLoader(dataset=dataset_source, batch_size=batch_size, shuffle=True, num_workers=2)
        dataset_source_test = datasets.MNIST(root='dataset/mnist', train=False, transform=img_transform(28), download=True)
        dataloader_source_test = DataLoader(dataset=dataset_source_test, batch_size=batch_size, shuffle=False, num_workers=2)
    elif source_dataset == "mnist-m":
        train_list = os.path.join('dataset/mnist_m/mnist_m_train_labels.txt')
        test_list = os.path.join('dataset/mnist_m/mnist_m_test_labels.txt')
        dataset_source = MNISTM(data_root='dataset/mnist_m/mnist_m_train', data_list=train_list, transform=img_transform(28))
        dataloader_source = DataLoader(dataset=dataset_source, batch_size=batch_size, shuffle=True, num_workers=2)
        dataset_source_test =  MNISTM(data_root='dataset/mnist_m/mnist_m_test', data_list=test_list, transform=img_transform(28))
        dataloader_source_test = DataLoader(dataset=dataset_source_test, batch_size=batch_size, shuffle= False, num_workers=2)
    elif source_dataset == "svhn":
        dataset_source = SVHN('dataset/svhn/', split='train', transform=img_transform(28), download=False)
        dataloader_source = DataLoader(dataset=dataset_source, batch_size=batch_size, shuffle=True, num_workers=2)
        dataset_source_test = SVHN('dataset/svhn/', split='test', transform=img_transform(28), download=False)
        dataloader_source_test = DataLoader(dataset=dataset_source_test, batch_size=batch_size, shuffle=False, num_workers=2)
    elif source_dataset == "usps":
        dataset_source = USPS(split="train", transform=img_transform(28))
        dataloader_source = DataLoader(dataset=dataset_source, batch_size=batch_size, shuffle=True, num_workers=2)
        dataset_source_test =  USPS(split="test", transform=img_transform(28))
        dataloader_source_test = DataLoader(dataset=dataset_source_test, batch_size=batch_size, shuffle= False, num_workers=2)
    elif source_dataset.startswith("amazon/"):
        dataset_source = Amazon("dataset/amazon/", split="train", category=source_dataset.removeprefix("amazon/"))
        dataloader_source = DataLoader(dataset=dataset_source, batch_size=batch_size, shuffle=True, num_workers=2)
        dataset_source_test = Amazon("dataset/amazon/", split="test", category=source_dataset.removeprefix("amazon/"))
        dataloader_source_test = DataLoader(dataset=dataset_source_test, batch_size=batch_size, shuffle=True, num_workers=2)
    elif source_dataset == "cifar10":
        dataset = datasets.CIFAR10("dataset/cifar10/", train=True, transform=img_transform(32), download=True)
        # Split into train-validation because cifar10c uses the same test set
        train_ds, valid_ds = random_split(dataset, (0.8, 0.2), torch.Generator().manual_seed(0))
        dataloader_source = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
        dataloader_source_test = DataLoader(dataset=valid_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    else:
        raise NotImplementedError()

    if target_dataset == "mnist":
        dataset_target = datasets.MNIST(root='dataset/mnist', train=True, transform=img_transform(28), download=True)
        dataloader_target = DataLoader(dataset=dataset_target, batch_size=batch_size, shuffle=True, num_workers=2)
        dataset_target_test = datasets.MNIST(root='dataset/mnist', train=False, transform=img_transform(28), download=True)
        dataloader_target_test = DataLoader(dataset=dataset_target_test, batch_size=batch_size, shuffle=False, num_workers=2)
    elif target_dataset == "mnist-m":
        train_list = os.path.join('dataset/mnist_m/mnist_m_train_labels.txt')
        test_list = os.path.join('dataset/mnist_m/mnist_m_test_labels.txt')
        dataset_target = MNISTM(data_root='dataset/mnist_m/mnist_m_train', data_list=train_list, transform=img_transform(28))
        dataloader_target = DataLoader(dataset=dataset_target, batch_size=batch_size, shuffle=True, num_workers=2)
        dataset_target_test =  MNISTM(data_root='dataset/mnist_m/mnist_m_test', data_list=test_list, transform=img_transform(28))
        dataloader_target_test = DataLoader(dataset=dataset_target_test, batch_size=batch_size, shuffle= False, num_workers=2)
    elif target_dataset == "svhn":
        dataset_target = SVHN('dataset/svhn/', split='train', transform=img_transform(28), download=False)
        dataloader_target = DataLoader(dataset=dataset_target, batch_size=batch_size, shuffle=True, num_workers=2)
        dataset_target_test = SVHN('dataset/svhn/', split='test', transform=img_transform(28), download=False)
        dataloader_target_test = DataLoader(dataset=dataset_target_test, batch_size=batch_size, shuffle=False, num_workers=2)
    elif target_dataset == "usps":
        dataset_target = USPS(split="train", transform=img_transform(28))
        dataloader_target = DataLoader(dataset=dataset_target, batch_size=batch_size, shuffle=True, num_workers=2)
        dataset_target_test =  USPS(split="test", transform=img_transform(28))
        dataloader_target_test = DataLoader(dataset=dataset_target_test, batch_size=batch_size, shuffle= False, num_workers=2)
    elif target_dataset.startswith("amazon/"):
        dataloader_target = None
        dataset_target_test = Amazon("dataset/amazon/", split="all", category=target_dataset.removeprefix("amazon/"))
        dataloader_target_test = DataLoader(dataset=dataset_target_test, batch_size=batch_size, shuffle=True, num_workers=2)
    elif target_dataset.startswith("cifar10c/"):
        dataloader_target = None
        dataset_target_test = CIFAR10C("dataset/cifar10c/", split="test", category=target_dataset.removeprefix("cifar10c/"), transform=img_transform(32))
        dataloader_target_test = DataLoader(dataset=dataset_target_test, batch_size=batch_size, shuffle=True, num_workers=2)
    else:
        raise NotImplementedError()
    
    if test_time:
        dataloader_target = DataLoader(dataset=dataset_target_test, batch_size=batch_size, shuffle=True, num_workers=2)
        
    return dataloader_source, dataloader_source_test, dataloader_target, dataloader_target_test


def eval_proxy_risk_method(model, source_dataset, target_dataset, args):

    batch_size = 128
    nepoch = 20
    lam = 50
    test_time = False

    dataloader_source, dataloader_source_test, dataloader_target, dataloader_target_test = get_dataloaders(source_dataset, target_dataset, batch_size, test_time)
    check_model_path = "./checkpoints/dann_models/{}_{}_{}/checkpoint.pth".format(source_dataset, target_dataset, "test_time" if test_time else "training_time")

    acc_t = test(model, dataloader_target_test)

    # load pretrained check model
    model2 = torch.load(check_model_path).cuda()

    # Get prediction on target test set
    target_test_pred_labels, target_test_labels = get_model_pred(model, dataloader_target_test)

    # estimate proxy risk
    acc_s = test(model2, dataloader_source_test)
    loss_s_domain = test_divergence(model2, dataloader_source_test, source=True)
    loss_t_domain = test_divergence(model2, dataloader_target_test, source=False)
    dir_loss = (1. - acc_s) + 0.1*(loss_s_domain + loss_t_domain)
    disagree = test_disagreement(model, model2, dataloader_target_test)
    eps = dir_loss * 1.1
    max_proxy_risk = disagree
    precision, recall, f1 = get_proxy_risk_error_detection_results(model2, dataloader_target_test, target_test_pred_labels)
    for epoch in range(nepoch):
        # maximize disagreement
        proxy_risk_train(model, model2, dataloader_source, dataloader_target, nepoch, lam)

        # check domain-invariant constrain
        acc_s = test(model2, dataloader_source_test)
        dann_acc_t = test(model2, dataloader_target_test)
        loss_s_domain = test_divergence(model2, dataloader_source_test, source=True)
        loss_t_domain = test_divergence(model2, dataloader_target_test, source=False)

        disagree = test_disagreement(model, model2, dataloader_target_test)
        dir_loss = (1. - acc_s) + 0.1*(loss_s_domain + loss_t_domain)
        if dir_loss <= eps and max_proxy_risk < disagree:
            precision, recall, f1 = get_proxy_risk_error_detection_results(model2, dataloader_target_test, target_test_pred_labels)
            max_proxy_risk = disagree
    
    estimated_acc = 1 - max_proxy_risk
    estimated_error = abs(estimated_acc - acc_t)
        
    return acc_t, estimated_acc, estimated_error, precision, recall, f1


def eval_our_ri_method(model, source_dataset, target_dataset, weight_by, args):

    batch_size = 128
    test_time = True
    nround = 5
    nepoch = 5
    if source_dataset.startswith("amazon/"):
        nepoch = 20
    gamma = 0.1
    num_classes = 10
        

    dataloader_source, dataloader_source_test, dataloader_target, dataloader_target_test = get_dataloaders(source_dataset, target_dataset, batch_size, test_time)

    check_model_paths = []
    for i in range(nepoch):
        check_model_path = "./checkpoints/ensemble_dann_arch_source_models/{:d}/{}/checkpoint.pth".format(i, source_dataset)
        check_model_paths.append(check_model_path)

    if weight_by == "similar":
        if source_dataset.startswith("amazon/"):
            distance_matrix = get_distance_matrix(dataloader_target_test.dataset.raw_text,
                                                  dataloader_source_test.dataset.raw_text,
                                                  "cls")
        elif source_dataset.startswith("cifar10"):
            valid_images, _ = get_data(dataloader_source_test)
            test_images, _ = get_data(dataloader_target_test)
            distance_matrix = get_distance_matrix(test_images, valid_images, "resnet")
        nn_weights = torch.softmax(distance_matrix ** -1, dim=1)
        labeled_test_weights = nn_weights.sum(dim=0).cpu().numpy()
        source_test_images, source_test_labels = get_data(dataloader_source_test)
        weighted_labeled_ds = CustomDatasetWithWeight(source_test_images, source_test_labels, labeled_test_weights)

    pseudo_weight = gamma
    if target_dataset == "usps":
        pseudo_weight = pseudo_weight * 10

    t_test_acc = test(model, dataloader_target_test)
    
    source_train_images, source_train_labels = get_data(dataloader_source)
    target_test_images, target_test_labels = get_data(dataloader_target_test)
    target_test_pred_labels, _ = get_model_pred(model, dataloader_target_test)

    pseudo_train_ds = CustomDatasetWithWeight(source_train_images, source_train_labels, np.ones(source_train_labels.shape[0], dtype=np.float32))
    dataloader_source_pseudo = torch.utils.data.DataLoader(pseudo_train_ds, batch_size=batch_size, shuffle=True)

    for i in range(nround):
        pred_record = np.zeros((target_test_pred_labels.shape[0], num_classes), dtype=np.float64)

        for epoch in range(nepoch):
            model2 = torch.load(check_model_paths[epoch]).cuda()
            optimizer = optim.Adam(model2.parameters(), lr=1e-3)
            ensemble_self_training(model2, dataloader_source_pseudo, pseudo_weight, optimizer)
            target_test_pred_labels_2, _ = get_model_pred(model2, dataloader_target_test)
            if weight_by is None:
                model_weight = 1
            elif weight_by == "acc":
                model_weight = test(model2, dataloader_source_test) ** 10
            elif weight_by == "similar":
                model_weight = test(model2, DataLoader(weighted_labeled_ds, batch_size=batch_size)) ** 10
            pred_record[np.arange(target_test_pred_labels_2.shape[0]), target_test_pred_labels_2] += model_weight
        
        target_test_pseudo_labels = np.argmax(pred_record, axis=1)
        disagree_record = (target_test_pseudo_labels!=target_test_pred_labels)
        disagree_indices = np.where(disagree_record)[0]
        
        pseudo_train_images = np.concatenate((source_train_images, target_test_images[disagree_indices]), axis=0)
        pseudo_train_labels = np.concatenate((source_train_labels, target_test_pseudo_labels[disagree_indices]), axis=0)
        pseudo_train_weights = np.concatenate((np.ones(source_train_labels.shape[0], dtype=np.float32), np.zeros(disagree_indices.shape[0], dtype=np.float32)), axis=0)
        pseudo_train_ds = CustomDatasetWithWeight(pseudo_train_images, pseudo_train_labels, pseudo_train_weights)
        dataloader_source_pseudo = torch.utils.data.DataLoader(pseudo_train_ds, batch_size=batch_size, shuffle=True)
    
    estimated_acc = 1.0 - np.mean(disagree_record)
    estimated_error = abs(estimated_acc-t_test_acc)
    precision, recall, f1 = get_error_detection_results(disagree_record, target_test_labels, target_test_pred_labels)
    
    return t_test_acc, estimated_acc, estimated_error, precision, recall, f1


def eval_our_rm_method(model, source_dataset, target_dataset, args):

    batch_size = 128
    test_time = True
    nround = 5
    nepoch = 5
    gamma = 0.1
    num_classes = 10

    dataloader_source, dataloader_source_test, dataloader_target, dataloader_target_test = get_dataloaders(source_dataset, target_dataset, batch_size, test_time)
    check_model_path = "./checkpoints/dann_models/{}_{}_{}/checkpoint.pth".format(source_dataset, target_dataset, "test_time" if test_time else "training_time")

    pseudo_weight = gamma
    if target_dataset == "usps":
        pseudo_weight = pseudo_weight * 10

    t_test_acc = test(model, dataloader_target_test)
    
    source_train_images, source_train_labels = get_data(dataloader_source)
    target_test_images, target_test_labels = get_data(dataloader_target_test)
    target_test_pred_labels, _ = get_model_pred(model, dataloader_target_test)

    pseudo_train_ds = CustomDatasetWithWeight(source_train_images, source_train_labels, np.ones(source_train_labels.shape[0], dtype=np.float32))
    dataloader_source_pseudo = torch.utils.data.DataLoader(pseudo_train_ds, batch_size=batch_size, shuffle=True)

    for i in range(nround):
        model2 = torch.load(check_model_path).cuda()
        optimizer = optim.Adam(model2.parameters(), lr=1e-3)
        pred_record = np.zeros((target_test_pred_labels.shape[0], num_classes), dtype=np.float64)

        for epoch in range(nepoch):
            dann_ensemble_self_training(model2, dataloader_source_pseudo, dataloader_target, pseudo_weight, optimizer)
            target_test_pred_labels_2, _ = get_model_pred(model2, dataloader_target_test)
            pred_record[np.arange(target_test_pred_labels_2.shape[0]), target_test_pred_labels_2] += 1     
        
        target_test_pseudo_labels = np.argmax(pred_record, axis=1)
        disagree_record = (target_test_pseudo_labels!=target_test_pred_labels)
        disagree_indices = np.where(disagree_record)[0]
        
        pseudo_train_images = np.concatenate((source_train_images, target_test_images[disagree_indices]), axis=0)
        pseudo_train_labels = np.concatenate((source_train_labels, target_test_pseudo_labels[disagree_indices]), axis=0)
        pseudo_train_weights = np.concatenate((np.ones(source_train_labels.shape[0], dtype=np.float32), np.zeros(disagree_indices.shape[0], dtype=np.float32)), axis=0)
        pseudo_train_ds = CustomDatasetWithWeight(pseudo_train_images, pseudo_train_labels, pseudo_train_weights)
        dataloader_source_pseudo = torch.utils.data.DataLoader(pseudo_train_ds, batch_size=batch_size, shuffle=True)
    
    estimated_acc = 1.0 - np.mean(disagree_record)
    estimated_error = abs(estimated_acc-t_test_acc)
    precision, recall, f1 = get_error_detection_results(disagree_record, target_test_labels, target_test_pred_labels)

    return t_test_acc, estimated_acc, estimated_error, precision, recall, f1 

def eval_conf_avg_method(model, source_dataset, target_dataset, args):
    batch_size = 128
    test_time = True

    dataloader_source, dataloader_source_test, dataloader_target, dataloader_target_test = get_dataloaders(source_dataset, target_dataset, batch_size, test_time)

    t_test_acc = test(model, dataloader_target_test)
    target_confs = get_model_conf(model, dataloader_target_test)
    estimated_acc = np.mean(target_confs)
    estimated_error = abs(estimated_acc-t_test_acc)

    return t_test_acc, estimated_acc, estimated_error

def eval_ensemble_conf_avg_method(model, source_dataset, target_dataset, args):
    batch_size = 128
    test_time = True

    dataloader_source, dataloader_source_test, dataloader_target, dataloader_target_test = get_dataloaders(source_dataset, target_dataset, batch_size, test_time)

    ensemble_size = 10
    ensemble_model = []
    for i in range(ensemble_size):
        if args.model_type == "typical_dnn":
            model_path = "./checkpoints/ensemble_typical_dnn_source_models/{}/{}/checkpoint.pth".format(i, source_dataset)
        elif args.model_type == "dann_arch":
            model_path = "./checkpoints/ensemble_dann_arch_source_models/{}/{}/checkpoint.pth".format(i, source_dataset)
        
        ensemble_model.append(torch.load(model_path).cuda())
    
    t_test_acc = test(model, dataloader_target_test)
    target_ensemble_conf = get_ensemble_model_confidence(ensemble_model, dataloader_target_test)
    estimated_acc = np.mean(target_ensemble_conf)
    estimated_error = abs(estimated_acc-t_test_acc)

    return t_test_acc, estimated_acc, estimated_error

def eval_conf_method(model, source_dataset, target_dataset, args):
    batch_size = 128
    test_time = True

    dataloader_source, dataloader_source_test, dataloader_target, dataloader_target_test = get_dataloaders(source_dataset, target_dataset, batch_size, test_time)

    source_acc = test(model, dataloader_source_test)
    source_model_outputs = get_model_outputs(model, dataloader_source_test)
    source_test_pred_labels = np.argmax(source_model_outputs, axis=1)
    target_model_outputs, target_test_labels = get_model_outputs(model, dataloader_target_test, True)
    target_test_pred_labels = np.argmax(target_model_outputs, axis=1)
    source_scores = np.max(source_model_outputs, axis=1)
    target_scores = np.max(target_model_outputs, axis=1)
    indices = np.argsort(source_scores)
    sorted_confs = source_scores[indices]
    j = int((1 - source_acc) * indices.shape[0])
    threshold = (sorted_confs[j] + sorted_confs[j+1]) / 2
    error_record = (target_scores < threshold).astype(np.int32)
    
    precision, recall, f1 = get_error_detection_results(error_record, target_test_labels, target_test_pred_labels)
    return precision, recall, f1

def eval_trust_score_method(model, source_dataset, target_dataset, args):
    batch_size = 128
    test_time = True

    dataloader_source, dataloader_source_test, dataloader_target, dataloader_target_test = get_dataloaders(source_dataset, target_dataset, batch_size, test_time)

    trust_score_module = TrustScore()
    source_model_outputs, source_labels = get_model_logit_outputs(model, dataloader_source, True)
    trust_score_module.fit(source_model_outputs, source_labels)

    source_acc = test(model, dataloader_source_test)
    source_model_outputs = get_model_logit_outputs(model, dataloader_source_test)
    source_test_pred_labels = np.argmax(source_model_outputs, axis=1)
    target_model_outputs, target_test_labels = get_model_logit_outputs(model, dataloader_target_test, True)
    target_test_pred_labels = np.argmax(target_model_outputs, axis=1)
    source_scores = trust_score_module.get_score(source_model_outputs, source_test_pred_labels)
    target_scores = trust_score_module.get_score(target_model_outputs, target_test_pred_labels)
    indices = np.argsort(source_scores)
    sorted_confs = source_scores[indices]
    j = int((1 - source_acc) * indices.shape[0])
    threshold = (sorted_confs[j] + sorted_confs[j+1]) / 2
    error_record = (target_scores < threshold).astype(np.int32)

    precision, recall, f1 = get_error_detection_results(error_record, target_test_labels, target_test_pred_labels)
    return precision, recall, f1

def main():
    parser = argparse.ArgumentParser(description='Evaluate unsupervised accuracy estimation and error detection tasks')
    parser.add_argument('--model-type', default="typical_dnn", choices=['typical_dnn', "dann_arch"], type=str, help='given model type')
    parser.add_argument('--method', default="conf_avg", type=str, help='method')
    parser.add_argument('--datasets', nargs='+', required=False)
    parser.add_argument('--source-datasets', nargs='+', required=False)
    parser.add_argument('--target-datasets', nargs='+', required=False)
    parser.add_argument('--seed', type=int, default=0)

    # args parse
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    all_error = []
    all_f1 = []

    if args.datasets is not None:
        source_datasets = args.datasets
        target_datasets = args.datasets
    else:
        source_datasets = args.source_datasets
        target_datasets = args.target_datasets

    for source_dataset in source_datasets:
        for target_dataset in target_datasets:
            if source_dataset == target_dataset:
                continue

            if args.model_type == "typical_dnn":
                model_path = "./checkpoints/typical_dnn_source_models/{}/checkpoint.pth".format(source_dataset)
            elif args.model_type == "dann_arch":
                model_path = "./checkpoints/dann_arch_source_models/{}/checkpoint.pth".format(source_dataset)

            # any candidate model 
            model = torch.load(model_path).cuda()
            
            if args.method == 'our_ri':
                t_test_acc, estimated_acc, estimated_error, precision, recall, f1 = eval_our_ri_method(model, source_dataset, target_dataset, None, args)
            elif args.method == 'acc_weighted_ri':
                t_test_acc, estimated_acc, estimated_error, precision, recall, f1 = eval_our_ri_method(model, source_dataset, target_dataset, "acc", args)
            elif args.method == 'sim_weighted_ri':
                t_test_acc, estimated_acc, estimated_error, precision, recall, f1 = eval_our_ri_method(model, source_dataset, target_dataset, "similar", args)
            elif args.method == 'our_rm':
                t_test_acc, estimated_acc, estimated_error, precision, recall, f1 = eval_our_rm_method(model, source_dataset, target_dataset, args)
            elif args.method == 'proxy_risk':
                t_test_acc, estimated_acc, estimated_error, precision, recall, f1 = eval_proxy_risk_method(model, source_dataset, target_dataset, args)
            elif args.method == 'conf_avg':
                t_test_acc, estimated_acc, estimated_error = eval_conf_avg_method(model, source_dataset, target_dataset, args)
            elif args.method == 'ensemble_conf_avg':
                t_test_acc, estimated_acc, estimated_error = eval_ensemble_conf_avg_method(model, source_dataset, target_dataset, args)
            elif args.method == 'conf':
                precision, recall, f1 = eval_conf_method(model, source_dataset, target_dataset, args)
            elif args.method == 'trust_score':
                precision, recall, f1 = eval_trust_score_method(model, source_dataset, target_dataset, args)
            else:
                assert False, "Not supported method: {}".format(args.method)
            
            print("Source Dataset: {}, Target Dataset: {}".format(source_dataset, target_dataset))
            if args.method in ['our_ri', 'acc_weighted_ri', 'sim_weighted_ri', 'our_rm', 'proxy_risk']:
                print("Target Accuracy: {:.4f}, Estimated Accuracy: {:.4f}, Estimation Error: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f}".format(t_test_acc, estimated_acc, estimated_error, precision, recall, f1))
                all_error.append(estimated_error)
                all_f1.append(f1)
            elif args.method in ['conf_avg', 'ensemble_conf_avg']:
                print("Target Accuracy: {:.4f}, Estimated Accuracy: {:.4f}, Estimation Error: {:.4f}".format(t_test_acc, estimated_acc, estimated_error))
                all_error.append(estimated_error)
            elif args.method in ['conf', 'trust_score']:
                print("Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f}".format(precision, recall, f1))
                all_f1.append(f1)

    print("Summary Results")
    print("Method: {}".format(args.method))
    if args.method in ['our_ri', 'acc_weighted_ri', 'sim_weighted_ri', 'our_rm', 'proxy_risk']:
        print("Accuracy Estimation")
        avg_error = np.mean(all_error)
        print("Average Estimation Error: {:.4f}".format(avg_error))
        print("Error Detection")
        avg_f1 = np.mean(all_f1)
        print("Average F1 Score: {:.4f}".format(avg_f1))
    elif args.method in ['conf_avg', 'ensemble_conf_avg']:
        print("Accuracy Estimation")
        avg_error = np.mean(all_error)
        print("Average Estimation Error: {:.4f}".format(avg_error))
    elif args.method in ['conf', 'trust_score']:
        print("Error Detection")
        avg_f1 = np.mean(all_f1)
        print("Average F1 Score: {:.4f}".format(avg_f1))


if __name__ == "__main__":
    main()
