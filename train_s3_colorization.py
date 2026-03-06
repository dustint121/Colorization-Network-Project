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
    save_every_steps=10000,
    use_s3torchconnector=True,
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

    # if 'save_every_step' is between 75% and 125% of the number of batches in an epoch:
        #  skip since it will be saved at epoch end and step-based saving would be redundant
    check = (1.25 * len(train_loader) > save_every_steps) and (0.75 * len(train_loader) < save_every_steps)
    for epoch in range(num_epochs):
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


            if not check and global_step % save_every_steps == 0:
                # step-based checkpoint (unchanged)
                step_name = f"Landscape Dataset 90K_colorization_step_{global_step}.pt"
                ckpt_path = os.path.join(checkpoint_dir, step_name)
                state = {
                    "step": global_step,
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "loss": loss_value,
                }
                torch.save(state, ckpt_path)
                print(f"\nSaved checkpoint to {ckpt_path}")


                # ----- QUICK VALIDATION & BEST CHECKPOINT AT THIS STEP -----
                model.eval()
                val_loss_sum = 0.0
                val_count = 0
                check_point_val_epoch_bar = tqdm(val_loader, desc=f"Checkpoint Validation Check: step {global_step}", unit="batch")
                with torch.no_grad():
                    for L_val, ab_val in check_point_val_epoch_bar:
                        L_val = L_val.to(device).float()
                        ab_val = ab_val.to(device).float()
                        ab_pred_val = model(L_val)
                        loss_val = criterion(ab_pred_val, ab_val)
                        val_loss_sum += loss_val.item() * L_val.size(0)
                        val_count += L_val.size(0)
                step_val_loss = val_loss_sum / max(1, val_count)
                print(f"Step {global_step}: val_loss = {step_val_loss:.4f}")

                if step_val_loss < best_val_loss:
                    best_val_loss = step_val_loss
                    bucket_name = s3_prefix[5:].split("/")[0]  # Extract bucket name from s3://bucket/prefix
                    best_path = os.path.join(
                        checkpoint_dir, f"{bucket_name}_colorization_best.pt"
                    )
                    torch.save(
                        {
                            "step": global_step,
                            "epoch": epoch,
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "val_loss": step_val_loss,
                        },
                        best_path,
                    )
                    print(
                        f"New best checkpoint (val_loss={best_val_loss:.4f}) saved to {best_path}"
                    )
                model.train()  # back to training mode


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

        mean_val_loss = val_loss_sum / max(1, val_count)
        print(f"Epoch {epoch+1}: val_loss = {mean_val_loss:.4f}")

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
        save_every_steps=1_000,
        use_s3torchconnector=False,  # set to False to use boto3-based loader instead
    )

# check size of batches 
    # landscape dataset
        #train_loader = (90000 x 0.9) / 32 = 2531 batches per epoch
                        # 90000 / 32 = 2812 batches per epoch if no val split 
