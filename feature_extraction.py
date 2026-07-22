import os
import cv2
import torch
import torchaudio
import numpy as np

from pathlib import Path
from PIL import Image
from tqdm import tqdm

from transformers import Wav2Vec2Processor, Wav2Vec2Model
from torchvision import models, transforms


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "dataset"
OUTPUT_DIR = "features"

AUDIO_DIR = os.path.join(DATASET_DIR, "audio")
VIDEO_DIR = os.path.join(DATASET_DIR, "video")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Extract one frame every N seconds
FRAME_INTERVAL = 1.0

# Audio sample rate expected by Wav2Vec2
AUDIO_SAMPLE_RATE = 16000


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

AUDIO_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "audio")
VIDEO_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "video")

os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)


print("Using device:", DEVICE)


# ============================================================
# LOAD AUDIO MODEL
# ============================================================

print("Loading audio model...")

audio_processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base"
)

audio_model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base"
)

audio_model.to(DEVICE)
audio_model.eval()


# ============================================================
# LOAD VIDEO MODEL
# ============================================================

print("Loading video model...")

video_model = models.resnet50(
    weights=models.ResNet50_Weights.DEFAULT
)

# Remove final classification layer
video_model.fc = torch.nn.Identity()

video_model.to(DEVICE)
video_model.eval()


video_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# AUDIO FEATURE EXTRACTION
# ============================================================

def extract_audio_features(audio_path):

    try:

        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert stereo to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample audio
        if sample_rate != AUDIO_SAMPLE_RATE:

            resampler = torchaudio.transforms.Resample(
                sample_rate,
                AUDIO_SAMPLE_RATE
            )

            waveform = resampler(waveform)

        waveform = waveform.squeeze()

        # Convert to numpy
        audio = waveform.numpy()

        # Process audio
        inputs = audio_processor(
            audio,
            sampling_rate=AUDIO_SAMPLE_RATE,
            return_tensors="pt"
        )

        input_values = inputs.input_values.to(DEVICE)

        # Extract embedding
        with torch.no_grad():

            outputs = audio_model(
                input_values
            )

            # Average over time
            embedding = outputs.last_hidden_state.mean(
                dim=1
            )

        embedding = embedding.squeeze().cpu().numpy()

        return embedding

    except Exception as e:

        print(
            f"Error processing audio {audio_path}: {e}"
        )

        return None


# ============================================================
# VIDEO FEATURE EXTRACTION
# ============================================================

def extract_video_features(video_path):

    try:

        cap = cv2.VideoCapture(video_path)

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            fps = 30

        frame_interval = int(
            fps * FRAME_INTERVAL
        )

        frame_count = 0

        embeddings = []

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            # Sample frames
            if frame_count % frame_interval == 0:

                # OpenCV BGR -> RGB
                frame_rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                image = Image.fromarray(
                    frame_rgb
                )

                image = video_transform(
                    image
                )

                image = image.unsqueeze(
                    0
                ).to(DEVICE)

                # Extract visual embedding
                with torch.no_grad():

                    embedding = video_model(
                        image
                    )

                embedding = embedding.squeeze(
                    0
                ).cpu().numpy()

                embeddings.append(
                    embedding
                )

            frame_count += 1

        cap.release()

        if len(embeddings) == 0:

            return None

        # Average all frame embeddings
        video_embedding = np.mean(
            embeddings,
            axis=0
        )

        return video_embedding

    except Exception as e:

        print(
            f"Error processing video {video_path}: {e}"
        )

        return None


# ============================================================
# PROCESS AUDIO DATASET
# ============================================================

audio_extensions = [
    ".wav",
    ".mp3",
    ".flac",
    ".m4a"
]

audio_files = []

for root, dirs, files in os.walk(
    AUDIO_DIR
):

    for file in files:

        if Path(file).suffix.lower() in audio_extensions:

            audio_files.append(
                os.path.join(
                    root,
                    file
                )
            )


print(
    f"Found {len(audio_files)} audio files"
)


for audio_path in tqdm(
    audio_files,
    desc="Extracting audio features"
):

    filename = Path(
        audio_path
    ).stem

    output_path = os.path.join(
        AUDIO_OUTPUT_DIR,
        filename + ".npy"
    )

    # Skip already processed files
    if os.path.exists(output_path):
        continue

    embedding = extract_audio_features(
        audio_path
    )

    if embedding is not None:

        np.save(
            output_path,
            embedding
        )


# ============================================================
# PROCESS VIDEO DATASET
# ============================================================

video_extensions = [
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm"
]

video_files = []

for root, dirs, files in os.walk(
    VIDEO_DIR
):

    for file in files:

        if Path(file).suffix.lower() in video_extensions:

            video_files.append(
                os.path.join(
                    root,
                    file
                )
            )


print(
    f"Found {len(video_files)} video files"
)


for video_path in tqdm(
    video_files,
    desc="Extracting video features"
):

    filename = Path(
        video_path
    ).stem

    output_path = os.path.join(
        VIDEO_OUTPUT_DIR,
        filename + ".npy"
    )

    # Skip already processed files
    if os.path.exists(output_path):
        continue

    embedding = extract_video_features(
        video_path
    )

    if embedding is not None:

        np.save(
            output_path,
            embedding
        )


print(
    "Feature extraction complete!"
)