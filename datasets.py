from torchvision import datasets
import torchvision.transforms as transforms
import torch

def get_dataset(name, folder, transform=None):
    supported_datasets = ["vanilla", "flip_augment"]
    if not (name in supported_datasets):
        raise NotImplementedError("Dataset not implemented")
    if name == "vanilla":
        return datasets.ImageFolder(folder, transform=transform),
    if name == "flip_augment":
        vanilla = datasets.ImageFolder(folder, transform=transform)
        flip_trans = transforms.Compose([transform, transforms.RandomHorizontalFlip(p=1)])
        flipped = datasets.ImageFolder(folder, transform=flip_trans)
        return torch.utils.data.ConcatDataset([vanilla, flipped])
