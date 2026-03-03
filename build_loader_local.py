# build_loader_local.py
from torch.utils.data import DataLoader
from class_LocalColorizationDataset import LocalColorizationDataset

def build_local_loader(
    root_dir,
    batch_size=32,
    image_size=256,
    num_workers=4,
    shuffle=True,
):
    dataset = LocalColorizationDataset(root_dir=root_dir, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader
