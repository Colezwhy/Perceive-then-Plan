# Adapted from the Qwen3-VL fine-tuning code (https://github.com/QwenLM/Qwen3-VL, Apache-2.0).
"""Fine-tune Qwen3-VL into the Perceiver (Stage 1). Launch with scripts/train_perceiver.sh."""

import logging
import os
import pathlib

import torch
import transformers
from transformers import AutoConfig, AutoProcessor, Trainer

from ptp_perceiver.data import data_processor
from ptp_perceiver.data.data_processor import make_supervised_data_module
from ptp_perceiver.model import Qwen3VLForConditionalGenerationLayout
from ptp_perceiver.train.argument import DataArguments, ModelArguments, TrainingArguments

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dumps it to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_trainable(model_args, model):
    """Freeze/unfreeze parameter groups. The geometry merger and fusion module stay trainable."""
    for p in model.visual.parameters():
        p.requires_grad = model_args.tune_mm_vision
    for p in model.visual.merger.parameters():
        p.requires_grad = model_args.tune_mm_mlp
    for p in model.language_model.parameters():
        p.requires_grad = model_args.tune_mm_llm
    model.lm_head.requires_grad = model_args.tune_mm_llm
    for p in model.geometry_encoder.parameters():
        p.requires_grad = False


def count_trainable(model):
    groups = {"visual": model.visual, "language_model": model.language_model,
              "geometry_merger": model.geometry_merger, "feature_fusion": model.feature_fusion,
              "geometry_encoder": model.geometry_encoder}
    for name, module in groups.items():
        params = list(module.parameters())
        trainable = sum(p.numel() for p in params if p.requires_grad)
        total = sum(p.numel() for p in params)
        rank0_print(f"  {name:18s} trainable {trainable:>14,d} / {total:>14,d}")


def train():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    data_processor.local_rank = local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.geometry_encoder = True
    config.geometry_encoder_type = model_args.geometry_encoder_type
    config.geometry_merger_type = model_args.geometry_merger_type
    config.feature_fusion_method = model_args.feature_fusion_method

    model = Qwen3VLForConditionalGenerationLayout.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        attn_implementation=model_args.attn_implementation,
        dtype=(torch.bfloat16 if training_args.bf16 else None),
        geometry_encoder_path=model_args.geometry_encoder_path,
    )
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)

    model.config.use_cache = False
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    set_trainable(model_args, model)
    rank0_print("Trainable parameters:")
    count_trainable(model)

    data_module = make_supervised_data_module(processor, data_args=data_args)
    trainer = Trainer(model=model, processing_class=tokenizer, args=training_args, **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
