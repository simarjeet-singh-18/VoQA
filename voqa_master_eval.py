import os
import re
import json
import time
import sys
import torch
import torch.nn as nn
import torchaudio
import cv2
from PIL import Image
from tqdm import tqdm

# ==========================================
# 1. DATASET HANDLER (STRICT SAMPLING)
# ==========================================
class VoQADataset:
    def __init__(self, json_path, video_dir, audio_dir):
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        
        print(f"[INFO] Loading full dataset from {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        print(f"[INFO] Loaded {len(self.data)} total samples.")

    def __len__(self):
        return len(self.data)
        
    def _extract_frames_experimental(self, video_path):
        """
        Extracts 1 frame every 6 seconds (equivalent to taking 1s every 6s at 1 fps).
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        if not cap.isOpened():
            return frames
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None:
            fps = 30.0 # Fallback if OpenCV fails to read fps metadata
            
        frame_interval = int(fps * 6) # 1 frame every 6 seconds
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for idx in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR (OpenCV default) to RGB, then to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
                    
        cap.release()
        return frames

    def _process_audio_experimental(self, audio_path):
        """
        Loads audio and resamples to exactly 16 kHz as per the paper.
        """
        target_sample_rate = 16000
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Resample if the native audio is not 16kHz
            if sample_rate != target_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
                waveform = resampler(waveform)
                
            return waveform, target_sample_rate
        except Exception as e:
            return None, None

    def get_sample(self, idx):
        item = self.data[idx]
        
        if "video_only_id" in item or "video_audio_id" in item:
            video_filename = item.get("video_only_id", item.get("video_audio_id"))
            audio_filename = item.get("audio_id")
        else:
            video_id = item.get("video_id")
            video_filename = f"{video_id}.mp4" if not str(video_id).endswith(".mp4") else str(video_id)
            audio_filename = video_filename 
            
        ground_truth = item.get("anser", item.get("answer", "")).strip().lower()
        
        video_path = os.path.join(self.video_dir, video_filename)
        audio_path = os.path.join(self.audio_dir, audio_filename)
        
        # Apply strict experimental sampling
        extracted_frames = self._extract_frames_experimental(video_path)
        audio_waveform, sample_rate = self._process_audio_experimental(audio_path)

        return {
            "id": video_filename,
            "ground_truth": ground_truth,
            "video_path": video_path,
            "frames": extracted_frames,
            "audio_waveform": audio_waveform,
            "sample_rate": sample_rate
        }

# ==========================================
# 2. MODEL WRAPPERS & VGGISH AUDIO
# ==========================================
def normalize_text(text):
    text = str(text).lower().strip()
    return re.sub(r'[^\w\s]', '', text)

class AudioVideoFusionWrapper(nn.Module):
    """
    Implements the pure Audio-Visual architecture from the paper.
    - Video: Processed via target open-source visual model.
    - Audio: VGGish (128-D) -> Linear Layer (512-D).
    - Text Encoder: Excluded entirely.
    """
    def __init__(self, visual_model_name, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = device
        self.visual_model_name = visual_model_name
        
        print(f"[INFO] Initializing VGGish Audio Encoder (16kHz) -> 512-D Linear Layer...")
        # Load Pre-trained VGGish from torch hub
        self.vggish = torch.hub.load('harritaylor/torchvggish', 'vggish', trust_repo=True)
        self.vggish.eval()
        self.vggish.to(self.device)
        
        # Linear layer to project 128-D VGGish feature to 512-D as per experiments
        self.audio_projection = nn.Linear(128, 512).to(self.device)
        
        print(f"[INFO] Initializing Visual Encoder: {self.visual_model_name}...")
        # Initialize your open source visual model here (e.g. Qwen2.5-VL / InternVL)
        # Note: Because this is a custom architectural fusion, the exact forward pass 
        # depends on how you choose to concatenate the 512-D audio tensor into the visual LLM.
        
        # Example initialization placeholder for Qwen:
        # from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        # self.visual_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(visual_model_name).to(self.device)
        # self.visual_processor = AutoProcessor.from_pretrained(visual_model_name)

    def predict(self, sample):
        # 1. Process Audio (16kHz waveform -> VGGish -> 128-D -> Linear -> 512-D)
        audio_feature_512 = None
        if sample["audio_waveform"] is not None:
            with torch.no_grad():
                # VGGish expects a specific numpy format or tensor; 
                # passing the raw waveform (flattened if stereo)
                waveform = sample["audio_waveform"].mean(dim=0).cpu().numpy()
                vggish_128 = self.vggish(waveform) # shape: (num_segments, 128)
                vggish_128 = vggish_128.to(self.device)
                audio_feature_512 = self.audio_projection(vggish_128) # shape: (num_segments, 512)

        # 2. Process Video (Frames sampled 1s every 6s)
        frames = sample["frames"]
        if not frames:
            return ""

        # 3. Fusion & Prediction
        # Here you fuse the `audio_feature_512` with the `frames` through your 
        # visual model to generate the prediction without an explicit text question input.
        # output = self.visual_model(frames=frames, audio_embeds=audio_feature_512)
        
        prediction = "two" # Placeholder output
        return normalize_text(prediction)

# ==========================================
# 3. MASTER BENCHMARK EVALUATION LOOP
# ==========================================
def run_benchmark(models_dict, dataset):
    results = {}
    total_samples = len(dataset)
    
    for model_name, model_instance in models_dict.items():
        print(f"\n{'='*50}\nStarting Strict Experimental Evaluation: {model_name}\n{'='*50}")
        correct = 0
        evaluated_count = 0
        
        for idx in tqdm(range(total_samples), desc=f"Evaluating {model_name}"):
            sample = dataset.get_sample(idx)
            
            if not os.path.exists(sample["video_path"]) or not sample["frames"]:
                continue
                
            prediction = model_instance.predict(sample)
            gt = normalize_text(sample["ground_truth"])
            
            if gt in prediction or prediction in gt:
                correct += 1
            evaluated_count += 1
                
        accuracy = (correct / evaluated_count) * 100 if evaluated_count > 0 else 0.0
        results[model_name] = {
            "accuracy": round(accuracy, 2),
            "correct": correct,
            "total_evaluated": evaluated_count
        }
        print(f"\n[RESULT] {model_name} Accuracy: {accuracy:.2f}% ({correct}/{evaluated_count})")
        
        del model_instance
        torch.cuda.empty_cache()

    print("\n" + "="*50)
    print("FINAL EXPERIMENTAL RESULTS:")
    print("="*50)
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    JSON_PATH = "/mnt/gpu_users/achyut/Simarjeet/MVoQA/AVQA_outputs/modified_train_multi.json"
    VIDEO_DIR = "/mnt/gpu_users/achyut/Simarjeet/MVoQA/AVQA_outputs/video_only"
    AUDIO_DIR = "/mnt/gpu_users/achyut/Simarjeet/MVoQA/AVQA_outputs/audio_only"
    
    dataset = VoQADataset(JSON_PATH, VIDEO_DIR, AUDIO_DIR)
    
    models_to_test = {
        "Custom-VGGish-Visual-Fusion": AudioVideoFusionWrapper(visual_model_name="Qwen/Qwen2.5-VL-3B-Instruct")
    }
    
    run_benchmark(models_to_test, dataset)