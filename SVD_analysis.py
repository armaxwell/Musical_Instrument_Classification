from pathlib import Path
from PIL import Image

TRAIN_SPECTROGRAMS_DIR = Path(__file__).resolve().parent / "Train_submission" / "Train_Spectrograms"
SAMPLE_IMAGE = "029500_morning-rain-piano-65875.png"


def get_spectrogram_path(filename: str = SAMPLE_IMAGE) -> Path:
    return TRAIN_SPECTROGRAMS_DIR / filename


def open_spectrogram_image(filename: str = SAMPLE_IMAGE):
    image_path = get_spectrogram_path(filename)
    if not image_path.exists():
        raise FileNotFoundError(f"Spectrogram not found: {image_path}")

    image = Image.open(image_path)
    image.show()
    return image


if __name__ == "__main__":
    print(f"Loading image from: {TRAIN_SPECTROGRAMS_DIR}")
    image = open_spectrogram_image()
    print(f"Opened image: {image.filename}")
    print(f"Image size: {image.size}, mode: {image.mode}")

