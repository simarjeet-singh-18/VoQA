"""
Wrapper around Qwen2.5-Omni (Alibaba/Qwen team) — an open-source
end-to-end model that natively accepts a video file WITH its audio
track and answers questions about it. This is a good first model
for this benchmark because many of the questions require both
modalities at once ("question_relation": "Both").

Setup (run on a machine with a CUDA GPU and internet access to
huggingface.co — NOT available inside this chat's sandbox):

    pip install "transformers>=4.52.3" accelerate qwen-omni-utils torch

Model card / usage reference:
    https://huggingface.co/Qwen/Qwen2.5-Omni-7B
    https://github.com/QwenLM/Qwen2.5-Omni

If you're VRAM-constrained, swap model_id for a quantized checkpoint,
e.g. "Qwen/Qwen2.5-Omni-7B-GPTQ-Int4" or "Qwen/Qwen2.5-Omni-7B-AWQ".
Qwen3-Omni (newer, larger) works as a drop-in alternative with the
same processor/generate pattern if you want to benchmark it too.
"""

from .base_model import BenchmarkModel, PredictionResult


class QwenOmniModel(BenchmarkModel):
    name = "Qwen2.5-Omni-7B"

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-Omni-7B",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        use_audio_in_video: bool = True,
        max_new_tokens: int = 32,
    ):
        import torch
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        self.use_audio_in_video = use_audio_in_video
        self.max_new_tokens = max_new_tokens

        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=getattr(torch, torch_dtype),
            device_map=device,
        )
        self.model.eval()

    def predict(self, sample) -> PredictionResult:
        import torch
        from qwen_omni_utils import process_mm_info

        prompt = self.build_prompt(sample)
        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a helpful assistant that answers multiple-choice "
                            "questions about a video's visual and audio content."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    # video_audio_path is the video file that already contains
                    # the audio track, so a single "video" input covers both
                    # modalities for this model.
                    {"type": "video", "video": sample.video_audio_path},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        try:
            text = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            audios, images, videos = process_mm_info(
                conversation, use_audio_in_video=self.use_audio_in_video
            )
            inputs = self.processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=self.use_audio_in_video,
            ).to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    use_audio_in_video=self.use_audio_in_video,
                    max_new_tokens=self.max_new_tokens,
                    return_audio=False,
                )

            gen_text = self.processor.batch_decode(
                output_ids[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )[0]
        except Exception as e:  # noqa: BLE001 - we want to log & continue benchmarking
            return PredictionResult(predicted_index=None, raw_response="", error=str(e))

        idx = self.parse_letter_answer(gen_text, len(sample.choices))
        return PredictionResult(predicted_index=idx, raw_response=gen_text)
