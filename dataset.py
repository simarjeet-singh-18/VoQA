"""
Dataset loader for the audio-visual multiple-choice QA benchmark.

Expected layout (matches the *_path fields already present in the JSON):

    <base_dir>/video_audio/<id>.mp4     # video WITH audio track
    <base_dir>/video_muted/<id>.mp4     # video, no audio
    <base_dir>/audio_mp3/<id>.mp3       # audio only
    <base_dir>/metadata/<id>.json       # per-sample metadata (if any)

The JSON file itself is a list of records shaped like:

    {
        "id": 183,
        "video_name": "-HG3Omg_89c_000030",
        "video_id": 341,
        "question_text": "What happened in the video?",
        "multi_choice": ["motorboat", "Yacht consignment", ...],
        "answer": 1,
        "question_relation": "View",
        "question_type": "Happening",
        "video_audio_path": "video_audio/183.mp4",
        "video_muted_path": "video_muted/183.mp4",
        "audio_mp3_path": "audio_mp3/183.mp3",
        "metadata_path": "metadata/183.json"
    }
"""

import json
import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AVQASample:
    id: int
    video_name: str
    video_id: int
    question_text: str
    choices: List[str]
    answer: int
    question_relation: str
    question_type: str
    video_audio_path: str
    video_muted_path: str
    audio_mp3_path: str
    metadata_path: str


class AVQADataset:
    def __init__(self, json_path: str, base_dir: str):
        self.json_path = json_path
        self.base_dir = base_dir
        with open(json_path, "r") as f:
            raw = json.load(f)
        self.samples: List[AVQASample] = [self._parse(r) for r in raw]

    def _parse(self, r: dict) -> AVQASample:
        return AVQASample(
            id=r["id"],
            video_name=r["video_name"],
            video_id=r["video_id"],
            question_text=r["question_text"],
            choices=r["multi_choice"],
            answer=r["answer"],
            question_relation=r.get("question_relation", ""),
            question_type=r.get("question_type", ""),
            video_audio_path=os.path.join(self.base_dir, r["video_audio_path"]),
            video_muted_path=os.path.join(self.base_dir, r["video_muted_path"]),
            audio_mp3_path=os.path.join(self.base_dir, r["audio_mp3_path"]),
            metadata_path=os.path.join(self.base_dir, r["metadata_path"]),
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def filter(self, question_type: Optional[str] = None, question_relation: Optional[str] = None):
        out = self.samples
        if question_type:
            out = [s for s in out if s.question_type == question_type]
        if question_relation:
            out = [s for s in out if s.question_relation == question_relation]
        return out

    def missing_files(self):
        """Sanity check: which referenced media files are absent on disk."""
        missing = []
        for s in self.samples:
            for p in (s.video_audio_path, s.audio_mp3_path):
                if not os.path.exists(p):
                    missing.append((s.id, p))
        return missing

    def summary(self):
        from collections import Counter
        return {
            "num_samples": len(self.samples),
            "question_types": dict(Counter(s.question_type for s in self.samples)),
            "question_relations": dict(Counter(s.question_relation for s in self.samples)),
            "num_choices_distribution": dict(Counter(len(s.choices) for s in self.samples)),
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inspect the dataset")
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--base_dir", required=True)
    args = parser.parse_args()

    ds = AVQADataset(args.json_path, args.base_dir)
    print(json.dumps(ds.summary(), indent=2))
    missing = ds.missing_files()
    print(f"\nMissing media files: {len(missing)}")
    for id_, p in missing[:10]:
        print(f"  sample {id_}: {p}")
