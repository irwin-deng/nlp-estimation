import torch
from torchvision import transforms, datasets
from PIL import Image
import os
import h5py
import os.path
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer

class Amazon(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapper for the Amazon reviews dataset.
    Download "unprocessed.tar.gz" from https://www.cs.jhu.edu/~mdredze/datasets/sentiment/
    """
    data: np.ndarray      # Contains inputs encoded with tf-idf
    raw_text: np.ndarray  # Contains inputs as strings
    targets: np.ndarray   # Labels of each example
    tf_idf_transform: TfidfVectorizer

    category_dirs = {  # Mapping from category names to their folders in the dataset
        "apparel": "apparel",
        "books": "books",
        "dvd": "dvd",
        "electronics": "electronics",
        "health": "health_&_personal_care",
        "kitchen": "kitchen_&_housewares",
        "music": "music",
        "sports": "sports_&_outdoors",
        "toys": "toys_&_games",
        "video": "video"}

    def __init__(self, root, category, split="train", transform=None):
        """
        param root: root directory
        param category: category name (should be a key in self.category_dirs)
        "{root}/{self.category_dirs[category]}/positive.review" should contain the positive reviews
        """
        self.root = root
        self.category = category
        self.split = split
        self.tf_idf_transform = transform

        if self.category not in self.category_dirs:
            raise ValueError('Wrong category entered!')
        if self.split not in ["train", "test", "all"]:
            raise ValueError('Wrong split entered! Please use split="train" '
                             'or split="extra" or split="test" '
                             'or split="train_and_extra" ')

        def clean_invalid_chars(input: str) -> str:
            to_replace = {"&": "&amp;", "\x1a": "", "<P>": ""}
            cleaned_input = input
            for old, new in to_replace.items():
                cleaned_input = cleaned_input.replace(old, new)
            return cleaned_input

        # read reviews from file. Store the title + review body in a list of strings
        positive_reviews = []
        with open(os.path.join(root, self.category_dirs[category], "positive.review"), encoding='cp1252') as file:
            parsed_dataset = ET.fromstring("<root>" + clean_invalid_chars(file.read()) + "</root>")
            for item in parsed_dataset:
                positive_reviews.append(item.find("title").text + item.find("review_text").text)
            file.close()
        negative_reviews = []
        with open(os.path.join(root, self.category_dirs[category], "negative.review"), encoding='cp1252') as file:
            parsed_dataset = ET.fromstring("<root>" + clean_invalid_chars(file.read()) + "</root>")
            for item in parsed_dataset:
                negative_reviews.append(item.find("title").text + item.find("review_text").text)
            file.close()

        # Split into train/test
        if split == "train":
            positive_reviews = positive_reviews[:int(len(positive_reviews) * 0.75)]
            negative_reviews = negative_reviews[:int(len(negative_reviews) * 0.75)]
        elif split == "test":
            positive_reviews = positive_reviews[int(len(positive_reviews) * 0.75):]
            negative_reviews = negative_reviews[int(len(negative_reviews) * 0.75):]
        elif split == "all":
            pass
        else:
            raise NotImplementedError()

        # do tf-idf transform
        if self.tf_idf_transform is None:
            corpus = []
            for category_dir in self.category_dirs.values():
                for file_name in ["positive.review", "negative.review"]:
                    with open(os.path.join(root, category_dir, file_name), encoding='cp1252') as file:
                        ds_split = ET.fromstring("<root>" + clean_invalid_chars(file.read()) + "</root>")
                        for item in ds_split:
                            corpus.append(item.find("title").text + item.find("review_text").text)
                        file.close()
            self.tf_idf_transform = TfidfVectorizer(max_features=5000).fit(corpus)

        self.raw_text = np.array(positive_reviews + negative_reviews)
        self.data = np.array(self.tf_idf_transform.transform(positive_reviews + negative_reviews).todense())
        self.targets = np.concatenate((np.ones(len(positive_reviews)), np.zeros(len(negative_reviews))))

    def __getitem__(self, index):
        features, target = self.data[index], self.targets[index]
        return features.astype(np.float32), target.astype(np.int64)

    def __len__(self):
        return len(self.data)


class SVHN(torch.utils.data.Dataset):
    url = ""
    filename = ""
    file_md5 = ""

    split_list = {
        'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
        'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                 "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"],
        'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                  "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"],
        'train_and_extra': [
                ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                 "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
                ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                 "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"]]}

    def __init__(self, root, split='train',
                 transform=None, target_transform=None, download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # training set or test set or extra set

        if self.split not in self.split_list:
            raise ValueError('Wrong split entered! Please use split="train" '
                             'or split="extra" or split="test" '
                             'or split="train_and_extra" ')

        if self.split == "train_and_extra":
            self.url = self.split_list[split][0][0]
            self.filename = self.split_list[split][0][1]
            self.file_md5 = self.split_list[split][0][2]
        else:
            self.url = self.split_list[split][0]
            self.filename = self.split_list[split][1]
            self.file_md5 = self.split_list[split][2]

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(root, self.filename))

        if self.split == "test":
            self.data = loaded_mat['X']
            self.targets = loaded_mat['y']
            # Note label 10 == 0 so modulo operator required
            self.targets = (self.targets % 10).squeeze()    # convert to zero-based indexing
            self.data = np.transpose(self.data, (3, 2, 0, 1))
        else:
            self.data = loaded_mat['X']
            self.targets = loaded_mat['y']

            if self.split == "train_and_extra":
                extra_filename = self.split_list[split][1][1]
                loaded_mat = sio.loadmat(os.path.join(root, extra_filename))
                self.data = np.concatenate([self.data,
                                                  loaded_mat['X']], axis=3)
                self.targets = np.vstack((self.targets,
                                               loaded_mat['y']))
            # Note label 10 == 0 so modulo operator required
            self.targets = (self.targets % 10).squeeze()    # convert to zero-based indexing
            self.data = np.transpose(self.data, (3, 2, 0, 1))

    def __getitem__(self, index):
        if self.split == "test":
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target.astype(np.int64)

    def __len__(self):
        if self.split == "test":
            return len(self.data)
        else:
            return len(self.data)

    def _check_integrity(self):
        root = self.root
        if self.split == "train_and_extra":
            md5 = self.split_list[self.split][0][2]
            fpath = os.path.join(root, self.filename)
            train_integrity = check_integrity(fpath, md5)
            extra_filename = self.split_list[self.split][1][1]
            md5 = self.split_list[self.split][1][2]
            fpath = os.path.join(root, extra_filename)
            return check_integrity(fpath, md5) and train_integrity
        else:
            md5 = self.split_list[self.split][2]
            fpath = os.path.join(root, self.filename)
            return check_integrity(fpath, md5)

    def download(self):
        if self.split == "train_and_extra":
            md5 = self.split_list[self.split][0][2]
            download_url(self.url, self.root, self.filename, md5)
            extra_filename = self.split_list[self.split][1][1]
            md5 = self.split_list[self.split][1][2]
            download_url(self.url, self.root, extra_filename, md5)
        else:
            md5 = self.split_list[self.split][2]
            download_url(self.url, self.root, self.filename, md5)


class USPS(torch.utils.data.Dataset):
    def __init__(self,
                split="train",
                transform=None,
                data_path="./dataset/usps/usps.h5"):
        self.data_path = data_path
        self.transform = transform

        with h5py.File(self.data_path, 'r') as hf:
            if split == "train":
                train = hf.get('train')
                X = train.get('data')[:] * 255
                X = X.reshape((-1, 16, 16)).astype(np.uint8)
                y = train.get('target')[:]
            elif split == "test":
                test = hf.get('test')
                X = test.get('data')[:] * 255
                X = X.reshape((-1, 16, 16)).astype(np.uint8)
                y = test.get('target')[:]
            else:
                assert False, "invalid split!"
        self.X = X
        # print(X[0]*255)
        self.y = y
        self.n_data = self.X.shape[0]

    def __getitem__(self, item):
        img, label = self.X[item], self.y[item]
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)
            label = int(label)

        return img, label

    def __len__(self):
        return self.n_data


class MNISTM(torch.utils.data.Dataset):
    def __init__(self, data_root, data_list, transform=None):
        self.root = data_root
        self.transform = transform

        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)

        self.img_paths = []
        self.img_labels = []

        for data in data_list:
            self.img_paths.append(data[:-3])
            self.img_labels.append(data[-2])

    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')

        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels

    def __len__(self):
        return self.n_data


class CIFAR10C(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapper for the corrupted CIFAR-10 dataset.
    Download "CIFAR-10-C.tar" from https://zenodo.org/record/2535967#.ZF_PxhHMJD9
    """
    images: list[Image.Image]
    labels: np.ndarray
    root: str  # root directory of dataset containing image/label files

    def __init__(self, root, split, category, transform=None):
        self.root = root
        self.transform = transform

        self.n = 10000
        # Read images from files
        self.images = np.load(os.path.join(root, f"{category}.npy"))[-10000:]  # Get all with corruption severity 5
        # Convert numpy arrays to PIL images
        self.images = [Image.fromarray(self.images[i], mode="RGB") for i in range(self.n)]

        self.labels = np.load(os.path.join(root, "labels.npy"))[-10000:]

        if split != "test":
            raise NotImplementedError()

    def __getitem__(self, item):
        image = self.images[item]
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[item]

    def __len__(self):
        return self.n


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.labels = labels
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Load data and get label
        X = self.images[index]
        y = self.labels[index]

        return X, y

class CustomDatasetWithWeight(torch.utils.data.Dataset):
    def __init__(self, images, labels, weights):
        self.labels = labels
        self.images = images
        self.weights = weights

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Load data and get label
        X = self.images[index]
        y = self.labels[index]
        w = self.weights[index]

        return X, y, w