import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import os
import shutil
import matplotlib.pyplot as plt

torch.set_float32_matmul_precision("high")
import dataset  # your existing dataset module


def conv_block(
    in_channels,
    out_channels,
    kernel_size=3,
    stride=1,
    padding=1,
    activation=nn.Identity(),
    pooling=nn.Identity(),
):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        activation,
        pooling,
    )


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_fft, self.hop = 1024, 24
        window = torch.hamming_window(self.n_fft)
        self.register_buffer("window", window)

        self.block1 = conv_block(1, 64, kernel_size=7, activation=nn.ReLU(), pooling=nn.MaxPool2d(2))
        self.block2 = conv_block(64, 128, kernel_size=7, activation=nn.ReLU(), pooling=nn.MaxPool2d(2))
        self.block3 = conv_block(128, 128, kernel_size=3, activation=nn.ReLU(), pooling=nn.MaxPool2d(2))
        self.block4 = conv_block(128, 256, kernel_size=3, activation=nn.ReLU(), pooling=nn.MaxPool2d(2))
        self.block5 = conv_block(
            256,
            256,
            kernel_size=3,
            activation=nn.ReLU(),
            pooling=nn.AdaptiveMaxPool2d(1),
        )

        self.fc_ts = nn.Sequential(nn.Linear(256, 128), nn.ReLU())

        self.feats_encoder = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )

        self.fc1 = nn.Sequential(nn.Linear(256, 128), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
        self.fc_regression = nn.Linear(64, 1)

    def forward(self, ts, feats):
        # Process time series STFT -> conv blocks
        sp = (
            torch.stft(
                ts,
                n_fft=self.n_fft,
                hop_length=self.hop,
                window=self.window,
                return_complex=True,
            )
            .abs()
            .unsqueeze(1)
        )
        sp = sp[:, :, :126]  # Crop frequencies ~0-12 Hz
        x = self.block1(sp)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x).squeeze()
        ts_emb = self.fc_ts(x)

        # Process extra features
        feats_emb = self.feats_encoder(feats)

        # Combine and regress
        shared = torch.cat((ts_emb, feats_emb), dim=1)
        shared = self.fc1(shared)
        shared = self.fc2(shared)
        out = self.fc_regression(shared)
        return out[:, 0]


# Device
DEVICE = "cuda"
BATCH_SIZE = 32
# Datasets and loaders
train_ds = dataset.BaseDataset(train=True, device=DEVICE)
val_ds = dataset.BaseDataset(train=False, device=DEVICE)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# Model, loss, optimizer
model = torch.compile(Model().to(DEVICE))
criterion = nn.SmoothL1Loss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Scheduler: halve LR every epoch
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

# Training directory
train_dir = "training"
if os.path.exists(train_dir):
    shutil.rmtree(train_dir)
os.makedirs(train_dir, exist_ok=True)

best_val_loss = float("inf")
train_losses = []
val_losses = []

num_epochs = 10
for epoch in range(1, num_epochs + 1):
    model.train()
    for i, (ts, feats, gt) in enumerate(train_loader, 1):
        preds = model(ts, feats)
        loss = criterion(preds, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        if i % 100 == 0:
            print(f"Epoch {epoch} Iter {i}/{len(train_loader)} - Train Loss: {loss.item():.3f}")

    # Step the scheduler at epoch end
    scheduler.step()
    current_lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch} - learning rate now {current_lr:.6f}")

    # Validation
    model.eval()
    val_running = 0.0
    val_unnormalized = 0.0
    with torch.no_grad():
        for i, (ts, feats, gt) in enumerate(val_loader, 1):
            ts, feats, gt = ts.to(DEVICE), feats.to(DEVICE), gt.to(DEVICE)
            preds = model(ts, feats)
            v_loss = criterion(preds, gt)
            val_running += v_loss.item()
            # un-normalize to original scale
            preds = preds * train_ds.std_labels + train_ds.mean_labels
            gt = gt * train_ds.std_labels + train_ds.mean_labels

            v_loss_un = (preds - gt).abs().mean()
            val_unnormalized += v_loss_un.item()
            if i % 25 == 0:
                print(
                    f"Epoch {epoch} Val Iter {i}/{len(val_loader)} - Val L1: {v_loss.item():.3f}, Val Unnormalized L1: {v_loss_un.item():.3f}"
                )

    avg_val_loss = val_running / len(val_loader)
    val_losses.append(avg_val_loss)
    print(
        f"Epoch {epoch} - Avg Val L1: {avg_val_loss:.3f}, Avg Val Unnormalized L1: {val_unnormalized / len(val_loader):.3f}"
    )

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(train_dir, "best_model.pth"))

# Plot loss curves
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(
    [i * len(train_loader) for i in range(1, num_epochs + 1)],
    val_losses,
    label="Val Loss",
    marker="o",
)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curves")
plt.savefig(os.path.join(train_dir, "loss_curve.png"))
plt.close()

print(
    f"Training complete. Best Val Loss: {best_val_loss:.3f}, Best Unnormalized Val Loss: {val_unnormalized / len(val_loader):.3f}"
)
