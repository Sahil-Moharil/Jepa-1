import torch
from torch.utils.data import DataLoader

from models.ajepa import AJEPA
from data.dataset import RealAudioDataset
from training.trainer import train_one_epoch
from utils.config import CONFIG
from utils.helpers import get_device


def main():
    device = get_device()
    print(f"Using device: {device}")

    dataset = RealAudioDataset(
    audio_dir="data/audio",        # path to your folder with .wav files
    frames=CONFIG["target_frames"],
    feature_dim=CONFIG["input_dim"]
    )


    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"]
    )

    model = AJEPA(
        input_dim=CONFIG["input_dim"],
        embed_dim=CONFIG["embed_dim"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])

    for epoch in range(CONFIG["epochs"]):
        loss = train_one_epoch(model, dataloader, optimizer, device)
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Loss: {loss:.4f}")

    torch.save(model.state_dict(), "ajepa_model.pth")
    print("Model saved as ajepa_model.pth")


if __name__ == "__main__":
    main()
