import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import DisasterDataset
from model import UNet
import os

# --- Hyperparameters ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
EPOCHS = 5
IMAGE_DIR = "data/images" # Satellite images here
MASK_DIR = "data/masks"   # Black/white masks here

def train():
    # --- Hardware Acceleration ---
    # Detect Apple Silicon (MPS), NVIDIA (CUDA), or fallback to CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[System] Apple M-Series GPU detected. Engaging MPS acceleration.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # --- Initialize Model, Loss, and Optimizer ---
    model = UNet(in_channels=3, out_channels=1).to(device)
    
    # BCEWithLogitsLoss is the mathematical standard for binary (0 or 1) pixel classification
    loss_fn = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Load Data ---
    if not os.path.exists(IMAGE_DIR):
        print(f"Waiting for data! Please create '{IMAGE_DIR}' and '{MASK_DIR}' directories.")
        return

    dataset = DisasterDataset(image_dir=IMAGE_DIR, mask_dir=MASK_DIR)
    
    if len(dataset) == 0:
        print("Directories found, but no images inside. Add some data!")
        return
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("[System] Initializing Training Loop...")
    
    # --- THE ENGINE ROOM ---
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for batch_idx, (data, targets) in enumerate(dataloader):
            # Move the tensors onto Mac's GPU
            data = data.to(device)
            targets = targets.to(device)

            # A. The Forward Pass (Make a guess)
            predictions = model(data)

            # B. Calculate the Error (How far off was the guess from the actual mask?)
            loss = loss_fn(predictions, targets)

            # C. The Backward Pass (Calculus: compute the gradients for the weights)
            optimizer.zero_grad() # Clear out the math from the last loop
            loss.backward()       # Compute derivatives

            # D. The Optimizer Step (Actually tweak the numbers in the U-Net to be smarter)
            optimizer.step()

            running_loss += loss.item()

        # Print progress at the end of the epoch
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Average Error (Loss): {avg_loss:.4f}")

    # --- Save the Brain ---
    torch.save(model.state_dict(), "disaster_unet.pth")
    print("\n[System] Training complete! Neural Network weights saved to 'disaster_unet.pth'")

if __name__ == "__main__":
    train()
