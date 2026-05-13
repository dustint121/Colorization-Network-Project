# train_s3_colorization.py
import os
import re
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast 
from torch.optim import Adam
from tqdm.auto import tqdm

from class_ColorizationNet import ColorizationNet
from build_loader_s3 import build_train_val_s3_loaders

import dotenv
dotenv.load_dotenv()   


def train_s3(
    s3_prefix,
    region,
    endpoint,
    num_epochs=5,
    num_workers=os.cpu_count(),
    batch_size=32,
    image_size=256,
    lr=1e-4,
    device="cuda",
    checkpoint_dir="checkpoints_s3",
    use_s3torchconnector=True,
    use_persistent_workers=False,
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_loader, val_loader = build_train_val_s3_loaders(
        s3_prefix=s3_prefix,
        region=region,
        endpoint=endpoint,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        use_s3torchconnector=use_s3torchconnector,
        use_persistent_workers=use_persistent_workers,
    )

    model = ColorizationNet().to(device)
    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    scaler = GradScaler(enabled=(device == "cuda"))

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(num_epochs):
        # reset global step here; stream iterator messes up calculations
        global_step = epoch * len(train_loader)
        
        # ----- TRAIN -----
        model.train()
        train_epoch_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", unit="batch")
        for L_batch, ab_batch in train_epoch_bar:
            L_batch = L_batch.to(device).float()
            ab_batch = ab_batch.to(device).float()

            optimizer.zero_grad()

            # ---- MIXED PRECISION FORWARD + LOSS ----
            with autocast(device_type="cuda" if device == "cuda" else "cpu",
                                    dtype=torch.float16 if device == "cuda" else torch.float32,
                                    enabled=(device == "cuda")):
                ab_pred = model(L_batch)
                loss = criterion(ab_pred, ab_batch)

            # ---- BACKWARD WITH SCALER ----
            scaler.scale(loss).backward()

            # gradient clipping on scaled grads
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()


            global_step += 1
            loss_value = loss.item()
            train_epoch_bar.set_postfix({"loss": f"{loss_value:.4f}", "step": global_step})

        train_epoch_bar.close()

        # ----- VALIDATION AFTER EPOCH -----
        print(f"Epoch {epoch+1} completed. Starting validation...")
        val_epoch_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}", unit="batch")
        model.eval()
        val_loss_sum = 0.0
        val_count = 0 
        with torch.no_grad():
            for L_val, ab_val in val_epoch_bar:
                L_val = L_val.to(device).float()
                ab_val = ab_val.to(device).float()
                ab_pred_val = model(L_val)
                loss_val = criterion(ab_pred_val, ab_val)
                val_loss_sum += loss_val.item() * L_val.size(0)
                val_count += L_val.size(0)
            val_epoch_bar.close()

        # Guard against the "validated on 0 samples" footgun.
        if val_count == 0:
            print(f"WARNING: Epoch {epoch+1} validated on 0 samples — "
                  f"check that workers are yielding data (look for "
                  f"'[S3ColorizationDataset] dropped sample' messages above).")
            mean_val_loss = float("inf")
        else:
            mean_val_loss = val_loss_sum / val_count
        print(f"Epoch {epoch+1}: val_loss = {mean_val_loss:.4f} (over {val_count} samples)")


        scheduler.step(mean_val_loss)

        # ----- END-OF-EPOCH CHECKPOINT (new) -----
        bucket_name = s3_prefix[5:].split("/")[0]  # Extract bucket name from s3://bucket/prefix
        epoch_ckpt_name = f"{bucket_name}_colorization_epoch_{epoch+1}.pt"
        epoch_ckpt_path = os.path.join(checkpoint_dir, epoch_ckpt_name)
        torch.save(
            {
                "step": global_step,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": mean_val_loss,
            },
            epoch_ckpt_path,
        )
        print(f"Saved end-of-epoch checkpoint to {epoch_ckpt_path}")

        # ----- BEST CHECKPOINT -----
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            bucket_name = s3_prefix[5:].split("/")[0]  # Extract bucket name from s3://bucket/prefix
            best_path = os.path.join(checkpoint_dir, f"{bucket_name}_colorization_best.pt")
            torch.save(
                {
                    "step": global_step,
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": mean_val_loss,
                },
                best_path,
            )
            print(
                f"New best checkpoint (val_loss={best_val_loss:.4f}) saved to {best_path}"
            )



if __name__ == "__main__":
    image_root = "c:/Users/dusti/Downloads/landscape-images"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # need to set AWS credentials in environment variables for s3torchconnector to work  
    os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")

    s3_uri="s3://landscape-images/"
    region="us-west-2"
    endpoint="https://s3.us-west-2.wasabisys.com"


    train_s3(
        s3_prefix=s3_uri,
        region=region,
        endpoint=endpoint,
        num_epochs=10,
        batch_size=32,
        image_size=256,
        lr=1e-4,
        device=device,
        checkpoint_dir="checkpoints_s3",
        use_s3torchconnector=False,  # set to False to use boto3-based loader instead
    )

# check size of batches 
    # landscape dataset
        #train_loader = (90000 x 0.9) / 32 = 2531 batches per epoch
                        # 90000 / 32 = 2812 batches per epoch if no val split 
