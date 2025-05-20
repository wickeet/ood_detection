import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import CIFAR10, MNIST, SVHN, FashionMNIST, KMNIST


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root", type=str, default="dataset", help="Directory data will be stored."
    )
    args = parser.parse_args()
    return args


def download_datasets(data_root):
    # dataset MNIST
    MNIST(data_root, download=True)
    for set in ["train", "test"]:
        dataset = MNIST(root=data_root, train=True if set == "train" else False)
        out_dir = Path(dataset.raw_folder).parent / "numpy" / set
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(len(dataset)):
            img, _ = dataset[i]
            img_np = np.array(img)
            np.save(out_dir / f"{dataset.__class__.__name__}_{i}.npy", img_np)

    # dataset FashionMNIST
    FashionMNIST(data_root, download=True)
    for set in ["train", "test"]:
        dataset = FashionMNIST(root=data_root, train=True if set == "train" else False)
        out_dir = Path(dataset.raw_folder).parent / "numpy" / set
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(len(dataset)):
            img, _ = dataset[i]
            img_np = np.array(img)
            np.save(out_dir / f"{dataset.__class__.__name__}_{i}.npy", img_np)

    # dataset KMNIST
    KMNIST(data_root, download=True)
    for set in ["train", "test"]:
        dataset = KMNIST(root=data_root, train=True if set == "train" else False)
        out_dir = Path(dataset.raw_folder).parent / "numpy" / set
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(len(dataset)):
            img, _ = dataset[i]
            img_np = np.array(img)
            np.save(out_dir / f"{dataset.__class__.__name__}_{i}.npy", img_np)

    # dataset CIFAR10 (downloaded in a different location than MNIST and FashionMNIST)
    root = Path(data_root) / "CIFAR10" / "raw"
    CIFAR10(root, download=True)
    for set in ["train", "test"]:
        dataset = CIFAR10(root=root, train=True if set == "train" else False)
        out_dir = root.parent / "numpy" / set
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(len(dataset)):
            img, _ = dataset[i]
            img_np = np.array(img).transpose((2, 0, 1))
            np.save(out_dir / f"{dataset.__class__.__name__}_{i}.npy", img_np)

    # dataset SVHN (downloaded in a different location than MNIST and FashionMNIST)
    root = Path(data_root) / "SVHN" / "raw"
    for set in ["train", "test"]:
        dataset = SVHN(root=root, split=set, download=True)
        out_dir = root.parent / "numpy" / set
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(len(dataset)):
            img, _ = dataset[i]
            img_np = np.array(img).transpose((2, 0, 1))
            np.save(out_dir / f"{dataset.__class__.__name__}_{i}.npy", img_np)


def save_list_as_csv(list, output_path):
    # Convert Path objects to strings
    with open(output_path, "w", newline="") as f:
        tsv_output = csv.writer(f, delimiter=",")
        tsv_output.writerow(list)


def create_train_test_splits(data_root):
    splits_dir = Path(data_root) / "data_splits"
    splits_dir.mkdir(exist_ok=True, parents=True)

    # Creating splits between train and val for datasets
    for dataset in ["FashionMNIST", "MNIST", "CIFAR10", "SVHN", "CAMELYON16"]:
        numpy_data_root = Path(data_root) / dataset / "numpy"
        train_and_val_list = list((numpy_data_root / "train").glob("*"))
        train_list, val_list = train_test_split(train_and_val_list, test_size=0.05, random_state=42)
        test_list = list((numpy_data_root / "test").glob("*"))
        for split_name, data_split in zip(
            ["train", "val", "test"], [train_list, val_list, test_list]
        ):
            save_list_as_csv(data_split, splits_dir / f"{dataset}_{split_name}.csv")


if __name__ == "__main__":
    args = parse_args()
    download_datasets(data_root=args.data_root)
    create_train_test_splits(data_root=args.data_root)
