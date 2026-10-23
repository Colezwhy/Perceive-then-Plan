# Adapted from the Qwen3-VL fine-tuning code (https://github.com/QwenLM/Qwen3-VL, Apache-2.0).
from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen3-VL-2B-Instruct")
    tune_mm_llm: bool = field(default=True, metadata={"help": "Train the language model and lm_head."})
    tune_mm_mlp: bool = field(default=False, metadata={"help": "Train the Qwen vision merger (projector)."})
    tune_mm_vision: bool = field(default=False, metadata={"help": "Train the Qwen vision encoder."})

    # Geometry branch. The geometry merger and the FiLM fusion are always trained; VGGT is always frozen.
    geometry_encoder_type: str = field(default="vggt")
    geometry_encoder_path: str = field(default="facebook/VGGT-1B")
    geometry_merger_type: str = field(default="mlp")
    feature_fusion_method: str = field(default="film")
    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "or 'sdpa'"})


@dataclass
class DataArguments:
    train_json: str = field(default="", metadata={"help": "Training records from tools/prepare_omni3d.py"})
    image_root: str = field(default="", metadata={"help": "Directory the record image paths are relative to"})
    max_pixels: int = field(default=576 * 32 * 32)
    min_pixels: int = field(default=784)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
