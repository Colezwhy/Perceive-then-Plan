# Adapted from the Qwen3-VL fine-tuning code (https://github.com/QwenLM/Qwen3-VL, Apache-2.0).
# Changes: image-only Omni3D dataset, VGGT geometry inputs, no sequence packing or video support.
"""Perceiver training dataset and collator."""

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import PIL.Image as Image
import torch
import transformers
from torch.utils.data import Dataset
from torchvision import transforms as TF

from .rope2d import get_rope_index_3

IGNORE_INDEX = -100
# Qwen3-VL token ids used to find assistant turns.
ASSISTANT_TOKEN_ID = 77091  # "assistant"
IM_END_TOKEN_ID = 151645  # "<|im_end|>"
VGGT_PATCH_SIZE = 14

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def update_processor_pixels(processor, data_args):
    """Set the min/max number of pixels per image used by the Qwen image processor."""
    ip = processor.image_processor
    if hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
        ip.min_pixels = data_args.min_pixels
        ip.max_pixels = data_args.max_pixels
    if hasattr(ip, "size") and isinstance(ip.size, dict):
        ip.size["shortest_edge"] = data_args.min_pixels
        ip.size["longest_edge"] = data_args.max_pixels
    rank0_print(f"Image processor: min_pixels={data_args.min_pixels}, max_pixels={data_args.max_pixels}")
    return processor


def resolve_image_path(image: str, image_root: str) -> str:
    return str((Path(image_root) / image).resolve()) if image_root else image


def _build_messages(item: Dict[str, Any], image_root: str) -> List[Dict[str, Any]]:
    images = item.get("image") or []
    if isinstance(images, str):
        images = [images]
    image_pool = [{"type": "image", "image": resolve_image_path(img, image_root)} for img in images]

    messages = []
    for turn in item["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        text = turn["value"]
        if role == "user":
            content = []
            for seg in re.split(r"(<image>)", text):
                if seg == "<image>":
                    if not image_pool:
                        raise ValueError("Number of <image> placeholders exceeds the number of provided images")
                    content.append(image_pool.pop(0))
                elif seg.strip():
                    content.append({"type": "text", "text": seg.strip()})
            messages.append({"role": role, "content": content})
        else:
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    if image_pool:
        raise ValueError(f"{len(image_pool)} image(s) remain unused (not consumed by placeholders)")
    return messages


def preprocess_qwen_visual(sources, processor, image_root: str) -> Dict:
    """Tokenize one conversation; only assistant answers (and their <|im_end|>\\n) are supervised."""
    if len(sources) != 1:
        raise ValueError(f"Expected 1 source, got {len(sources)}")
    messages = _build_messages(sources[0], image_root)

    full_result = processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt")
    input_ids = full_result["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)

    labels = torch.full_like(input_ids, IGNORE_INDEX)
    input_ids_flat = input_ids[0].tolist()
    L = len(input_ids_flat)
    pos = 0
    while pos < L:
        if input_ids_flat[pos] == ASSISTANT_TOKEN_ID:
            ans_start = pos + 2
            ans_end = ans_start
            while ans_end < L and input_ids_flat[ans_end] != IM_END_TOKEN_ID:
                ans_end += 1
            if ans_end < L:
                labels[0, ans_start : ans_end + 2] = input_ids[0, ans_start : ans_end + 2]
                pos = ans_end
        pos += 1

    full_result["labels"] = labels
    full_result["input_ids"] = input_ids
    return full_result


def prepare_geometry_inputs(image_paths: Sequence[str], image_grid_thw) -> List[torch.Tensor]:
    """Resize each image so its VGGT patch grid (14 px) matches the Qwen patch grid.

    Returns one [3, grid_h * 14, grid_w * 14] tensor in [0, 1] per image.
    """
    to_tensor = TF.ToTensor()
    images = []
    for i, img in enumerate(image_paths):
        if isinstance(img, str):
            img = Image.open(img)
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        img = img.convert("RGB")
        _, grid_h, grid_w = image_grid_thw[i]
        img = img.resize((int(grid_w) * VGGT_PATCH_SIZE, int(grid_h) * VGGT_PATCH_SIZE), Image.Resampling.BICUBIC)
        images.append(to_tensor(img))
    return images


class LazySupervisedDataset(Dataset):
    """Single-image Perceiver training data, tokenized lazily."""

    def __init__(self, processor, data_args):
        super().__init__()
        with open(data_args.train_json, "r") as f:
            self.list_data_dict = json.load(f)
        rank0_print(f"Loaded {len(self.list_data_dict)} training samples from {data_args.train_json}")

        self.image_root = data_args.image_root
        self.processor = update_processor_pixels(processor, data_args)
        self.tokenizer = processor.tokenizer
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        # Try the current sample first (sleep in case it is a transient file-system issue) ...
        for attempt_idx in range(num_base_retries):
            try:
                return self._get_item([self.list_data_dict[i]])
            except Exception as e:
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)
        # ... then the next one, in case this sample is corrupted.
        next_index = min(i + 1, len(self.list_data_dict) - 1)
        for attempt_idx in range(num_base_retries):
            try:
                return self._get_item([self.list_data_dict[next_index]])
            except Exception as e:
                print(f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:", e)
        return self._get_item([self.list_data_dict[i]])

    def _get_item(self, sources) -> Dict[str, torch.Tensor]:
        data_dict = preprocess_qwen_visual(sources, self.processor, self.image_root)

        image_paths = [resolve_image_path(source["image"], self.image_root) for source in sources]
        geometry_inputs = prepare_geometry_inputs(image_paths, data_dict["image_grid_thw"])
        data_dict["geometry_encoder_inputs"] = [geometry_inputs[0]]  # single image per sample

        grid_thw = data_dict.get("image_grid_thw")
        if not isinstance(grid_thw, Sequence):
            grid_thw = [grid_thw]
        position_ids, _ = get_rope_index_3(
            self.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.cat(grid_thw, dim=0),
            video_grid_thw=None,
            second_per_grid_ts=None,
        )
        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [data_dict["input_ids"][0].size(0)]
        return data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)
    padded = [
        torch.nn.functional.pad(tensor, (0, max_length - tensor.shape[2]), "constant", 1) for tensor in tensor_list
    ]
    return torch.cat(padded, dim=1)


@dataclass
class DataCollatorForSupervisedDataset:
    """Pads a batch and stacks the per-sample geometry inputs."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"].squeeze(0) for instance in instances]
        labels = [instance["labels"].squeeze(0) for instance in instances]
        position_ids = [instance["position_ids"] for instance in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            position_ids=position_ids,
        )
        images = [instance["pixel_values"] for instance in instances if "pixel_values" in instance]
        if images:
            batch["pixel_values"] = torch.cat(images, dim=0)
            batch["image_grid_thw"] = torch.cat(
                [instance["image_grid_thw"] for instance in instances if "image_grid_thw" in instance], dim=0
            )
        else:
            batch["pixel_values"] = None
            batch["image_grid_thw"] = None
        batch["pixel_values_videos"] = None
        batch["video_grid_thw"] = None
        batch["geometry_encoder_inputs"] = [torch.stack(instance["geometry_encoder_inputs"]) for instance in instances]
        return batch


def make_supervised_data_module(processor, data_args) -> Dict:
    train_dataset = LazySupervisedDataset(processor, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(processor.tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
