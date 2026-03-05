# build_loader_local.py
from torch.utils.data import DataLoader, random_split
from class_LocalColorizationDataset import LocalColorizationDataset

def build_train_val_loaders(
    root_dir,
    batch_size=32,
    image_size=256,
    num_workers=4,
    val_fraction=0.1,
):
    full_dataset = LocalColorizationDataset(root_dir=root_dir, image_size=image_size)

    num_total = len(full_dataset)
    num_val = int(num_total * val_fraction)
    num_train = num_total - num_val

    train_dataset, val_dataset = random_split(full_dataset, [num_train, num_val])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
