"""Run a trained Perceiver.

    python tools/infer.py --eval-json data/omni3d/eval.json --image-root $OMNI3D/datasets \
        --output results/predictions.json --limit 1000
    python tools/infer.py --model output/perceiver-Qwen3-VL-2B-Instruct --eval-json data/omni3d/eval.json \
        --image-root $OMNI3D/datasets --output results/predictions.json --limit 1000
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from transformers import AutoProcessor  # noqa: E402

from ptp_perceiver.data.data_processor import prepare_geometry_inputs, resolve_image_path  # noqa: E402
from ptp_perceiver.model import PerceiverForConditionalGeneration  # noqa: E402

RELEASED_MODEL = "Colezwhy/Perceive-then-Plan"
RELEASED_SUBFOLDER = "perceiver-2b"


def load_model(model_path, subfolder, geometry_encoder_path, attn_implementation, device):
    model = PerceiverForConditionalGeneration.from_pretrained(
        model_path,
        subfolder=subfolder,
        device_map=device,
        dtype="auto",
        attn_implementation=attn_implementation,
        geometry_encoder_path=geometry_encoder_path,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path, subfolder=subfolder)
    return model, processor


@torch.no_grad()
def predict(model, processor, image_path, prompt, max_new_tokens):
    messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    geometry_inputs = prepare_geometry_inputs([image_path], inputs["image_grid_thw"])
    inputs["geometry_encoder_inputs"] = [geometry_inputs[0].unsqueeze(0).to(model.device)]
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=RELEASED_MODEL, help="Hugging Face repo id or local checkpoint directory")
    p.add_argument("--subfolder", default=None, help=f"Default: {RELEASED_SUBFOLDER} for {RELEASED_MODEL}")
    p.add_argument("--eval-json", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="Number of images from --start")
    p.add_argument("--geometry-encoder-path", default="facebook/VGGT-1B")
    p.add_argument("--attn-implementation", default="flash_attention_2", help="or 'sdpa'")
    p.add_argument("--max-new-tokens", type=int, default=16384)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    with open(args.eval_json, "r") as f:
        records = json.load(f)
    end = None if args.limit is None else args.start + args.limit
    records = records[args.start:end]

    outputs = []
    if os.path.exists(args.output):
        with open(args.output, "r") as f:
            outputs = json.load(f)
        print(f"Resuming: {len(outputs)} predictions already in {args.output}")
    done = {o["id"] for o in outputs}

    subfolder = args.subfolder
    if subfolder is None:
        subfolder = RELEASED_SUBFOLDER if args.model == RELEASED_MODEL else ""
    model, processor = load_model(args.model, subfolder, args.geometry_encoder_path, args.attn_implementation,
                                  args.device)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    def save():
        with open(args.output, "w") as f:
            json.dump(outputs, f, indent=1)

    for n, record in enumerate(records, start=1):
        if record["id"] in done:
            continue
        image_path = resolve_image_path(record["image"], args.image_root)
        prediction = predict(model, processor, image_path, record["messages"][0]["content"], args.max_new_tokens)
        outputs.append({"id": record["id"], "image": record["image"], "prediction": prediction})
        print(f"[{n}/{len(records)}] image {record['id']} done")
        if len(outputs) % args.save_every == 0:
            save()
    save()
    print(f"Wrote {len(outputs)} predictions to {args.output}")


if __name__ == "__main__":
    main()
