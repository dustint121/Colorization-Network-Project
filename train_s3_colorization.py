# train_s3_colorization.py
import os
import re
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from class_ColorizationNet import ColorizationNet
from build_loader_s3 import build_train_val_s3_loaders


import dotenv
dotenv.load_dotenv()   



# Copy your existing find_latest_checkpoint and train_local functions here
# Just change the loader call:
def train_s3(
    s3_prefix,
    region,
    endpoint,
    num_epochs=5,
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
        num_workers=4,
        use_s3torchconnector=use_s3torchconnector,
    )


    model = ColorizationNet().to(device)
    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", patience=3, factor=0.5
                    )  # optional: reduce LR if val loss plateaus


    # Track best validation loss for "best" checkpoint saving
    best_val_loss = float("inf")

    global_step = 0
    # for epoch in range(start_epoch, num_epochs):
    for epoch in range(num_epochs):
        # ----- TRAIN -----
        model.train()
        epoch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch")

        for L_batch, ab_batch in epoch_bar:
            L_batch = L_batch.to(device).float()
            ab_batch = ab_batch.to(device).float()

            optimizer.zero_grad()
            ab_pred = model(L_batch)
            loss = criterion(ab_pred, ab_batch)
            loss.backward()

            # GRADIENT CLIPPING: clip gradients to max norm of 1.0 to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            optimizer.step()

            global_step += 1
            loss_value = loss.item()
            epoch_bar.set_postfix({"loss": f"{loss_value:.4f}", "step": global_step})

            if global_step % save_every_steps == 0:
                # Regular step-based checkpoint
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

        # ----- VALIDATION -----
        model.eval() # inference mode (no dropout/batchnorm updates)
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for L_val, ab_val in val_loader:
                L_val = L_val.to(device).float()
                ab_val = ab_val.to(device).float()
                ab_pred_val = model(L_val)
                loss_val = criterion(ab_pred_val, ab_val)
                val_loss_sum += loss_val.item() * L_val.size(0)
                val_count += L_val.size(0)

        mean_val_loss = val_loss_sum / max(1, val_count)
        print(f"Epoch {epoch+1}: val_loss = {mean_val_loss:.4f}")

        # LR SCHEDULER: step the ReduceLROnPlateau scheduler with the mean validation loss
        scheduler.step(mean_val_loss)

        # after computing mean_val_loss:
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_path = os.path.join(checkpoint_dir, "Landscape Dataset 90K_best.pt")
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
            print(f"New best checkpoint (val_loss={best_val_loss:.4f}) saved to {best_path}")


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
