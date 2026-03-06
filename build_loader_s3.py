# build_loader_s3.py
from torch.utils.data import DataLoader, IterableDataset
from class_S3ColorizationDataset import S3ColorizationDataset

import random

def build_train_val_s3_loaders(
    s3_prefix,
    region,
    endpoint,
    batch_size=32,
    image_size=256,
    num_workers=0,
    val_fraction=0.1,
    use_s3torchconnector=True,
):
    """
    Builds train/val loaders from S3 prefix.
    For IterableDataset, we approximate val split by taking first N% as val.
    """
    full_dataset = S3ColorizationDataset(
        s3_prefix_or_uris=s3_prefix,
        region=region,
        endpoint=endpoint,
        image_size=image_size,
        use_s3torchconnector=use_s3torchconnector,
    )

    all_keys = full_dataset._list_s3_keys()
    # total_images = len(all_keys)
    # print(f"Total images found in S3: {total_images}")
    # print(all_keys[:5])  # Print first 5 keys for verification

    train_val_folder_check = full_dataset.s3_train_val_folders_exists(bucket_name=s3_prefix)

    train_dataset, val_dataset = None, None
    if train_val_folder_check:
        # For simplicity with IterableDataset: use prefix split or manual sharding
        # Option 1: separate S3 prefixes for train/val
        train_dataset = S3ColorizationDataset(
            s3_prefix_or_uris=f"{s3_prefix}/train/",
            region=region,
            endpoint=endpoint,
            image_size=image_size,
            use_s3torchconnector=use_s3torchconnector,
            split="train",
        )
        val_dataset = S3ColorizationDataset(
            s3_prefix_or_uris=f"{s3_prefix}/val/",
            region=region,
            endpoint=endpoint,
            image_size=image_size,
            use_s3torchconnector=use_s3torchconnector,
            split="val",
        )
    else:
        print("No 'train'/'val' folders found in S3. Using single dataset and splitting by first N% of keys for val.")
        if s3_prefix.endswith("/") != True:
            s3_prefix += "/"  # Ensure prefix ends with slash for correct key construction

        all_keys = [f"{s3_prefix}{key}" for key in all_keys]  # Prepend prefix to keys
        random.shuffle(all_keys)  # Shuffle keys to randomize train/val split
        # Split into two lists based on val_fraction (default 0.1 means 10% val, 90% train)
        split_point = int(len(all_keys) * (1 - val_fraction))
        train_set_uris = all_keys[:split_point]
        val_set_uris = all_keys[split_point:]

        train_dataset = S3ColorizationDataset(
            s3_prefix_or_uris=train_set_uris,
            region=region,
            endpoint=endpoint,
            image_size=image_size,
            use_s3torchconnector=use_s3torchconnector,
            split="train",
        )
        val_dataset = S3ColorizationDataset(
            s3_prefix_or_uris=val_set_uris,
            region=region,
            endpoint=endpoint,
            image_size=image_size,
            use_s3torchconnector=use_s3torchconnector,
            split="val",
        )

        print(f"Train/Val split: {len(train_set_uris)} train images, {len(val_set_uris)} val images.")
        print(f"Train dataset length: {len(train_dataset)}, Val dataset length: {len(val_dataset)}")

    pin_memory = True if num_workers > 0 else False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        # persistent_workers=num_workers > 0,  # Keep workers alive across epochs for IterableDataset        
    )
    print(f"Built train loader with {len(train_loader)} batches.")
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        # persistent_workers=num_workers > 0,  # Keep workers alive across epochs for IterableDataset
    )
    print(f"Built val loader with {len(val_loader)} batches.")
    return train_loader, val_loader
