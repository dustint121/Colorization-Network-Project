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
    use_persistent_workers=False,
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
    train_split_keys, val_split_keys = S3ColorizationDataset._find_train_val_split_from_keys(all_keys)
    print(f"Split discovery: {len(train_split_keys)} train keys, {len(val_split_keys)} val keys")
    if train_split_keys:
        print(f"  sample train key: {train_split_keys[0]}")
    if val_split_keys:
        print(f"  sample val key:   {val_split_keys[0]}")

    train_val_folder_check = bool(train_split_keys) and bool(val_split_keys)
    train_dataset, val_dataset = None, None

    if train_val_folder_check:
        # Feed the discovered keys straight to the child datasets so they
        # don't re-list S3 under a wrong prefix. Bucket is taken from the
        # parent s3_prefix; keys are already relative to the bucket root.
        train_dataset = S3ColorizationDataset(
            s3_prefix_or_uris=s3_prefix,
            region=region,
            endpoint=endpoint,
            image_size=image_size,
            use_s3torchconnector=use_s3torchconnector,
            split="train",
            _keys_override=train_split_keys,
        )
        val_dataset = S3ColorizationDataset(
            s3_prefix_or_uris=s3_prefix,
            region=region,
            endpoint=endpoint,
            image_size=image_size,
            use_s3torchconnector=use_s3torchconnector,
            split="val",
            _keys_override=val_split_keys,
        )
        print(f"Built train dataset ({len(train_dataset)}) and val dataset ({len(val_dataset)}) from discovered split.")

    else:
        print("No 'train'/'val' folders found in S3. Using single dataset and splitting by first N% of keys for val.")
        if s3_prefix.endswith("/") != True:
            s3_prefix += "/"  # Ensure prefix ends with slash for correct key construction

        all_keys = [f"{s3_prefix}{key}" for key in all_keys]  # Prepend prefix to keys
        random.Random(42).shuffle(all_keys) # Shuffle keys to randomize train/val split; set seed for reproducibility
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
        persistent_workers=use_persistent_workers,  # Keep workers alive across epochs for IterableDataset        
    )

    print(f"Built train loader with {len(train_loader)} batches.")
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent_workers,  # Keep workers alive across epochs for IterableDataset
    )

    print(f"Built val loader with {len(val_loader)} batches.")


    # Fail fast rather than silently running empty epochs (was: 16 epochs of
    # val_loss=inf before the user hit Ctrl-C).
    n_train_batches = len(train_loader)
    n_val_batches   = len(val_loader)
    print(f"Built train loader with {n_train_batches} batches ({len(train_dataset)} samples).")
    print(f"Built val   loader with {n_val_batches} batches ({len(val_dataset)} samples).")

    if n_train_batches == 0 or n_val_batches == 0:
        raise RuntimeError(
            f"Empty DataLoader detected (train batches={n_train_batches}, "
            f"val batches={n_val_batches}). This usually means S3 key discovery "
            f"picked up 0 images. Check the 'Split discovery' log above and "
            f"verify that s3_prefix ({s3_prefix!r}) actually contains .jpg/.jpeg/.png "
            f"objects with a 'train' or 'val' component in their key path."
        )

    return train_loader, val_loader
