"""
VoQA Zero-Shot Multi-Model Benchmark
======================================
Video-only Question Answering where the "question" is burned in as a
watermark on the video frames (no separate text question input), paired
with an audio track (MUSIC-AVQA-style segment sampling). Evaluates a
REGISTRY of Hugging Face vision-language models sequentially on one GPU.

HONESTY NOTE ON SCOPE:
  "All possible Hugging Face VLMs" isn't literally achievable in one script
  -- there are thousands of checkpoints and no single shared API (some use
  `.chat(...)`, some take a raw text prompt with no chat template, some
  ship fully custom `trust_remote_code` inference code). What this script
  actually does: it covers the major open-source VLM families through FOUR
  general inference strategies (see STRATEGY REGISTRY below), and gives you
  a one-line pattern to register any additional model that fits one of
  those strategies. Anything requiring genuinely bespoke code (e.g. a
  model with a non-standard multi-stage pipeline) will need its own
  wrapper -- the registry makes that a ~15-line addition, not a rewrite.

Design constraints (unchanged from before):
  - Zero-shot only: no fine-tuning, no gradient updates, pretrained weights only.
  - Default mode "pure_zero_shot": no instruction text where the model's API
    allows it (VoQA paper Sec 3.3.1 -- image tokens only, model must notice,
    read, and answer the watermark completely unprompted).
  - Audio branch: VGGish (torchvggish), forward pass only. Segmented into
    ~1s chunks and subsampled at the same 6s stride as the video frames
    (MUSIC-AVQA Sec 5.1), logged per model run. NOT fused into the answer,
    and deliberately has NO trainable projection layer -- an untrained
    nn.Linear(128,512) is random noise, not a feature transform, so it
    cannot legitimately contribute to a "zero-shot, no fine-tuning" answer.
    If you need real audio-conditioned answers, that requires a training
    stage (a frozen-backbone linear probe is the minimal honest option --
    ask and I'll add it as a separate opt-in stage).
"""

import os
import re
import gc
import json
import torch
import torch.nn as nn
import torchaudio
import cv2
from PIL import Image
from tqdm import tqdm

# ==========================================
# 1. DATASET HANDLER (segment-aligned audio + video)
# ==========================================
class VoQADataset:
    def __init__(self, json_path, video_dir, audio_dir, seconds_per_sample=6):
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.seconds_per_sample = seconds_per_sample

        print(f"[INFO] Loading dataset index from {json_path}...")
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        print(f"[INFO] Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def _extract_frames(self, video_path):
        """1 frame sampled every `seconds_per_sample` seconds."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        if not cap.isOpened():
            return frames

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0

        frame_interval = max(1, int(fps * self.seconds_per_sample))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for idx in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

        cap.release()
        return frames

    def _load_audio(self, audio_path, target_sr=16000):
        try:
            waveform, sr = torchaudio.load(audio_path)
            if sr != target_sr:
                waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
            return waveform, target_sr
        except Exception:
            return None, None

    def get_sample(self, idx):
        item = self.data[idx]

        if "video_only_id" in item or "video_audio_id" in item:
            video_filename = item.get("video_only_id", item.get("video_audio_id"))
            audio_filename = item.get("audio_id", video_filename)
        else:
            video_id = item.get("video_id")
            video_filename = f"{video_id}.mp4" if not str(video_id).endswith(".mp4") else str(video_id)
            audio_filename = video_filename

        ground_truth = item.get("answer", item.get("anser", "")).strip().lower()

        video_path = os.path.join(self.video_dir, video_filename)
        audio_path = os.path.join(self.audio_dir, audio_filename)

        frames = self._extract_frames(video_path)
        waveform, sr = self._load_audio(audio_path)

        return {
            "id": video_filename,
            "ground_truth": ground_truth,
            "video_path": video_path,
            "frames": frames,
            "audio_waveform": waveform,
            "sample_rate": sr,
        }


def normalize_text(text):
    text = str(text).lower().strip()
    return re.sub(r"[^\w\s]", "", text)


def filter_response(raw_text, max_answer_words=6):
    """Trims free-form zero-shot generations down to a short answer span."""
    text = str(raw_text).strip()
    text = text.split("\n")[0]
    text = re.split(r"[.!?]", text)[0]
    words = text.split()
    if len(words) > max_answer_words:
        text = " ".join(words[:max_answer_words])
    return normalize_text(text)


# ==========================================
# 2. AUDIO BRANCH — VGGish, forward pass only, logged not fused
# ==========================================
class VGGishAudioEncoder:
    def __init__(self, device="cuda"):
        self.device = device
        print("[INFO] Loading pretrained VGGish (frozen, forward-pass only)...")
        self.model = torch.hub.load("harritaylor/torchvggish", "vggish", trust_repo=True)
        self.model.eval()
        self.model.to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_segments(self, waveform, sample_rate, seconds_per_sample=6):
        """Returns (T,128) segment embeddings subsampled to the video's stride, or None."""
        if waveform is None:
            return None
        mono = waveform.mean(dim=0).cpu().numpy()
        try:
            full_embeddings = self.model(mono, sample_rate)
        except Exception as e:
            print(f"[WARN] VGGish forward failed: {e}")
            return None
        full_embeddings = full_embeddings.detach().cpu().numpy()
        if full_embeddings.ndim == 1:
            full_embeddings = full_embeddings[None, :]
        return full_embeddings[::seconds_per_sample]


# ==========================================
# 3. STRATEGY REGISTRY — general inference patterns covering most HF VLMs
# ==========================================
PURE_ZERO_SHOT_TEXT = None   # no instruction text at all (default, paper-faithful)
LIGHT_PROMPT_TEXT = (
    "There is a question embedded as text in this image. "
    "Locate it, then answer it in one short word or phrase."
)


class BaseVLMWrapper:
    """Common no-op interface; subclasses implement `_generate`."""

    def __init__(self, model_path, device="cuda", instruction=PURE_ZERO_SHOT_TEXT):
        self.model_path = model_path
        self.device = device
        self.instruction = instruction

    def predict(self, sample):
        frames = sample.get("frames")
        if not frames:
            return ""
        try:
            raw = self._generate(frames)
        except Exception as e:
            print(f"[WARN] Generation failed for {self.model_path}: {e}")
            return ""
        return filter_response(raw)

    def _generate(self, frames):
        raise NotImplementedError

    def unload(self):
        if hasattr(self, "model"):
            del self.model
        gc.collect()
        torch.cuda.empty_cache()


class ChatTemplateMultiImageWrapper(BaseVLMWrapper):
    """
    Strategy for models supporting the unified `AutoModelForImageTextToText`
    + `AutoProcessor.apply_chat_template` API with multiple image inputs.
    Covers: Qwen2-VL / Qwen2.5-VL, LLaVA-NeXT, LLaVA-OneVision, Idefics2/3,
    Pixtral, Llama-3.2-Vision, SmolVLM, and most current transformers-native
    multi-image chat VLMs.
    """

    def __init__(self, model_path, device="cuda", instruction=PURE_ZERO_SHOT_TEXT,
                 max_new_tokens=20, torch_dtype=torch.float16, trust_remote_code=False):
        super().__init__(model_path, device, instruction)
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.max_new_tokens = max_new_tokens
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)

    @torch.no_grad()
    def _generate(self, frames):
        content = [{"type": "image", "image": f} for f in frames]
        if self.instruction:
            content.append({"type": "text", "text": self.instruction})
        messages = [{"role": "user", "content": content}]

        try:
            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            content.append({"type": "text", "text": ""})
            messages = [{"role": "user", "content": content}]
            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        inputs = self.processor(
            text=[text_prompt], images=frames, return_tensors="pt", padding=True
        ).to(self.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        trimmed = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]


class ChatTemplateSingleImageWrapper(ChatTemplateMultiImageWrapper):
    """
    Same API family as above, but for models that only reliably support a
    single image per turn (LLaVA-1.5 being the classic case). Uses the
    middle sampled frame as the representative frame.
    """

    @torch.no_grad()
    def _generate(self, frames):
        middle_frame = frames[len(frames) // 2]
        content = [{"type": "image", "image": middle_frame}]
        if self.instruction:
            content.append({"type": "text", "text": self.instruction})
        messages = [{"role": "user", "content": content}]

        try:
            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            content.append({"type": "text", "text": ""})
            messages = [{"role": "user", "content": content}]
            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        inputs = self.processor(
            text=[text_prompt], images=[middle_frame], return_tensors="pt", padding=True
        ).to(self.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        trimmed = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]


class TrustRemoteCodeChatWrapper(BaseVLMWrapper):
    """
    Strategy for models that ship a custom `.chat(tokenizer_or_processor,
    image, question, ...)` method via trust_remote_code, instead of the
    standard generate() + chat-template pattern.
    Covers: InternVL (older releases), MiniCPM-V, and similar.
    """

    def __init__(self, model_path, device="cuda", instruction=PURE_ZERO_SHOT_TEXT,
                 max_new_tokens=20, torch_dtype=torch.bfloat16):
        super().__init__(model_path, device, instruction)
        from transformers import AutoModel, AutoTokenizer

        self.max_new_tokens = max_new_tokens
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, trust_remote_code=True
        ).eval().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    def _generate(self, frames):
        middle_frame = frames[len(frames) // 2]
        # instruction may be None for pure zero-shot; some .chat() implementations
        # require a non-empty question string, so fall back to "" if so.
        question = self.instruction if self.instruction else ""
        response, _ = self.model.chat(
            self.tokenizer, middle_frame, question,
            generation_config=dict(max_new_tokens=self.max_new_tokens),
        )
        return response


class PromptOnlyWrapper(BaseVLMWrapper):
    """
    Strategy for older / non-chat-templated models that just take a raw
    text prompt + image, no conversation structure.
    Covers: BLIP-2, InstructBLIP, GIT, Kosmos-2, Fuyu, and similar.
    """

    def __init__(self, model_path, device="cuda", instruction=PURE_ZERO_SHOT_TEXT,
                 max_new_tokens=20, torch_dtype=torch.float16, model_cls="vision2seq",
                 trust_remote_code=False):
        super().__init__(model_path, device, instruction)
        from transformers import AutoProcessor
        if model_cls == "vision2seq":
            from transformers import AutoModelForVision2Seq as ModelCls
        elif model_cls == "causal_lm":
            from transformers import AutoModelForCausalLM as ModelCls
        else:
            raise ValueError(f"Unknown model_cls: {model_cls}")

        self.max_new_tokens = max_new_tokens
        self.model = ModelCls.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)

    @torch.no_grad()
    def _generate(self, frames):
        middle_frame = frames[len(frames) // 2]
        prompt_text = self.instruction if self.instruction else ""
        inputs = self.processor(text=prompt_text, images=middle_frame, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]


# Map registry string -> wrapper class, for a config-driven model list.
STRATEGY_MAP = {
    "chat_multi_image": ChatTemplateMultiImageWrapper,
    "chat_single_image": ChatTemplateSingleImageWrapper,
    "trust_remote_code_chat": TrustRemoteCodeChatWrapper,
    "prompt_only": PromptOnlyWrapper,
}


# ==========================================
# 4. MODEL REGISTRY — extend this list to add more models
# ==========================================
# Each entry: (display_name, hf_path_or_local_path, strategy, extra_kwargs)
# `strategy` must be a key in STRATEGY_MAP.
# To add a new model: pick the strategy matching its API, add one line here.
MODEL_REGISTRY = [
    ("Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-3B-Instruct", "chat_multi_image", {}),
    ("Qwen2-VL-7B-Instruct", "Qwen/Qwen2-VL-7B-Instruct", "chat_multi_image", {}),
    ("LLaVA-1.5-7B", "llava-hf/llava-1.5-7b-hf", "chat_single_image", {}),
    ("LLaVA-NeXT-Mistral-7B", "llava-hf/llava-v1.6-mistral-7b-hf", "chat_multi_image", {}),
    ("LLaVA-OneVision-Qwen2-7B", "llava-hf/llava-onevision-qwen2-7b-ov-hf", "chat_multi_image", {}),
    ("Idefics3-8B", "HuggingFaceM4/Idefics3-8B-Llama3", "chat_multi_image", {}),
    ("SmolVLM-Instruct", "HuggingFaceTB/SmolVLM-Instruct", "chat_multi_image", {}),
    ("InternVL2-2B", "OpenGVLab/InternVL2-2B", "trust_remote_code_chat", {"torch_dtype": torch.bfloat16}),
    ("MiniCPM-V-2_6", "openbmb/MiniCPM-V-2_6", "trust_remote_code_chat", {"torch_dtype": torch.bfloat16}),
    ("InstructBLIP-Vicuna-7B", "Salesforce/instructblip-vicuna-7b", "prompt_only", {"model_cls": "vision2seq"}),
    ("BLIP2-OPT-2.7B", "Salesforce/blip2-opt-2.7b", "prompt_only", {"model_cls": "vision2seq"}),
]


# ==========================================
# 5. BENCHMARK LOOP
# ==========================================
def run_benchmark(dataset, registry=MODEL_REGISTRY, device="cuda",
                   mode="pure_zero_shot", seconds_per_sample=6,
                   save_audio_embeddings_dir=None):
    assert mode in ("pure_zero_shot", "light_prompt")
    instruction = PURE_ZERO_SHOT_TEXT if mode == "pure_zero_shot" else LIGHT_PROMPT_TEXT

    audio_encoder = VGGishAudioEncoder(device=device)
    all_results = {}

    for name, model_path, strategy, extra_kwargs in registry:
        print(f"\n{'='*60}\nLoading: {name}  ({strategy})\n{'='*60}")
        wrapper_cls = STRATEGY_MAP[strategy]
        try:
            wrapper = wrapper_cls(model_path, device=device, instruction=instruction, **extra_kwargs)
        except Exception as e:
            print(f"[ERROR] Could not load {name}: {e}")
            all_results[name] = {"error": str(e)}
            continue

        correct, evaluated = 0, 0
        audio_log = {}

        for idx in tqdm(range(len(dataset)), desc=f"Evaluating {name}"):
            sample = dataset.get_sample(idx)
            if not os.path.exists(sample["video_path"]) or not sample["frames"]:
                continue

            prediction = wrapper.predict(sample)
            gt = normalize_text(sample["ground_truth"])

            audio_segments = audio_encoder.encode_segments(
                sample["audio_waveform"], sample["sample_rate"],
                seconds_per_sample=seconds_per_sample,
            )
            if audio_segments is not None:
                audio_log[sample["id"]] = audio_segments.tolist()

            if gt and (gt in prediction or prediction in gt):
                correct += 1
            evaluated += 1

        accuracy = (correct / evaluated) * 100 if evaluated > 0 else 0.0
        all_results[name] = {"accuracy": round(accuracy, 2), "correct": correct, "total_evaluated": evaluated}
        print(f"[RESULT] {name}: {accuracy:.2f}% ({correct}/{evaluated})")

        if save_audio_embeddings_dir and audio_log:
            os.makedirs(save_audio_embeddings_dir, exist_ok=True)
            out_path = os.path.join(save_audio_embeddings_dir, f"{name}_{mode}_audio.json")
            with open(out_path, "w") as f:
                json.dump(audio_log, f)

        wrapper.unload()

    print("\n" + "=" * 60)
    print(f"FINAL RESULTS ({mode})")
    print("=" * 60)
    print(json.dumps(all_results, indent=4))
    return all_results


if __name__ == "__main__":
    JSON_PATH = "./MVoQA/AVQA_outputs/modified_train_multi.json"
    VIDEO_DIR = "./MVoQA/AVQA_outputs/video_only"
    AUDIO_DIR = "./MVoQA/AVQA_outputs/audio_only"

    dataset = VoQADataset(JSON_PATH, VIDEO_DIR, AUDIO_DIR)

    run_benchmark(
        dataset,
        registry=MODEL_REGISTRY,
        device="cuda" if torch.cuda.is_available() else "cpu",
        mode="pure_zero_shot",
        save_audio_embeddings_dir="./voqa_audio_logs",
    )
