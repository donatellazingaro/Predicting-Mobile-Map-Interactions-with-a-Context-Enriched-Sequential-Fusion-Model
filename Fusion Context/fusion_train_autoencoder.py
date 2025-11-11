#!filepath projects/computational/'Fusion Context'/fusion_train_autoencoder.py
"""
fusion_train_autoencoder.py
---------------------------------------
Trains the Context Autoencoder (AE) used in the Fusion model.

Input:
  - fusion_context_train.pkl
  - fusion_context_dev.pkl
  - fusion_context_test.pkl

Each pickle entry: (session_id, context_vector, label)

Outputs:
  - models/context_encoder.pt              (encoder weights)
  - models/context_autoencoder_full.pt     (full AE weights)
  - outputs/context_autoencoder_loss.csv   (train/dev/test loss log)
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import DataLoader, TensorDataset # type: ignore

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 1000
PATIENCE = 20
LATENT_DIM = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= UTILITIES =================
def load_context_vectors(pkl_path):
    """Load context vectors from pickle (session_id, context_vec, label)."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return np.array([x[1] for x in data], dtype=np.float32)

def make_loader(array, batch_size, shuffle):
    return DataLoader(TensorDataset(torch.tensor(array, dtype=torch.float32)),
                      batch_size=batch_size, shuffle=shuffle)

# ================= MODEL =================
class ContextAutoencoder(nn.Module):
    def __init__(self, input_dim=8, latent_dim=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon

# ================= MAIN =================
if __name__ == "__main__":
    # --- Load data ---
    context_train = load_context_vectors(os.path.join(DATA_DIR, "fusion_context_train.pkl"))
    context_dev   = load_context_vectors(os.path.join(DATA_DIR, "fusion_context_dev.pkl"))
    context_test  = load_context_vectors(os.path.join(DATA_DIR, "fusion_context_test.pkl"))

    print("Loaded context data:")
    print(f"Train: {context_train.shape} | Dev: {context_dev.shape} | Test: {context_test.shape}")

    train_loader = make_loader(context_train, BATCH_SIZE, shuffle=True)
    dev_loader   = make_loader(context_dev, BATCH_SIZE, shuffle=False)
    test_loader  = make_loader(context_test, BATCH_SIZE, shuffle=False)

    # --- Initialize model ---
    model = ContextAutoencoder(input_dim=context_train.shape[1], latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_dev_loss = float("inf")
    patience_left = PATIENCE
    loss_log = []

    # --- Train loop ---
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train = 0
        for (batch,) in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            _, recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_train += loss.item() * batch.size(0)
        train_loss = total_train / len(train_loader.dataset)

        # --- Dev evaluation ---
        model.eval()
        total_dev = 0
        with torch.no_grad():
            for (batch,) in dev_loader:
                batch = batch.to(DEVICE)
                _, recon = model(batch)
                loss = criterion(recon, batch)
                total_dev += loss.item() * batch.size(0)
        dev_loss = total_dev / len(dev_loader.dataset)
        loss_log.append((train_loss, dev_loss))
        print(f"Epoch {epoch:03d} | Train {train_loss:.6f} | Dev {dev_loss:.6f}")

        if dev_loss < best_dev_loss - 1e-8:
            best_dev_loss = dev_loss
            patience_left = PATIENCE
            torch.save(model.encoder.state_dict(), os.path.join(MODEL_DIR, "context_encoder.pt"))
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "context_autoencoder_full.pt"))
            print("Saved best AE checkpoints.")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(" Early stopping.")
                break

    # --- Test evaluation ---
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "context_autoencoder_full.pt")))
    model.eval()
    total_test = 0
    with torch.no_grad():
        for (batch,) in test_loader:
            batch = batch.to(DEVICE)
            _, recon = model(batch)
            total_test += criterion(recon, batch).item() * batch.size(0)
    test_loss = total_test / len(test_loader.dataset)
    print(f"Final Test Reconstruction Loss: {test_loss:.6f}")

    # --- Save log ---
    df = pd.DataFrame(loss_log, columns=["train_loss", "dev_loss"])
    df["test_loss"] = test_loss
    df.to_csv(os.path.join(OUTPUT_DIR, "context_autoencoder_loss.csv"), index=False)
    print(f"Training complete. Loss log saved → {OUTPUT_DIR}/context_autoencoder_loss.csv")
