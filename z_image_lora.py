#!/usr/bin/env python3
import argparse
import datetime as dt
import re
from pathlib import Path
from typing import Optional, List

import torch
from diffusers import ZImagePipeline

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"


# ------------------------- Device helpers ------------------------- #

def choose_device(device_arg: str) -> str:
    """
    Choose device with preference: user > MPS > CUDA > CPU.
    """
    if device_arg != "auto":
        return device_arg

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def choose_dtype(device: str) -> torch.dtype:
    """
    Use bfloat16/float16 where it makes sense, otherwise float32.
    """
    if device in ("cuda", "mps"):
        # Z-Image is happy in BF16 on modern hardware
        if hasattr(torch, "bfloat16"):
            return torch.bfloat16
        return torch.float16
    return torch.float32


# ------------------------- Pipeline loading ------------------------- #

def load_pipeline(
    model_id: str,
    device: str,
    lora_path: Optional[Path] = None,
    lora_scale: float = 1.0,
    lora_weight_name: Optional[str] = None,
) -> ZImagePipeline:
    dtype = choose_dtype(device)

    print(f"[INFO] Loading base model: {model_id} (dtype={dtype}, device={device})")
    pipe = ZImagePipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )

    pipe.set_progress_bar_config(disable=False)

    if device == "mps":
        pipe = pipe.to("mps")
        pipe.enable_attention_slicing()
    elif device == "cuda":
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
    else:
        pipe = pipe.to("cpu")

    if lora_path is not None:
        print(f"[INFO] Loading LoRA from: {lora_path}")
        load_lora_into_pipeline(
            pipe,
            lora_path=lora_path,
            lora_scale=lora_scale,
            adapter_name="character",
            weight_name=lora_weight_name,
        )
    else:
        print("[INFO] No LoRA specified; using base Z-Image Turbo.")

    return pipe


def load_lora_into_pipeline(
    pipe: ZImagePipeline,
    lora_path: Path,
    lora_scale: float,
    adapter_name: str = "character",
    weight_name: Optional[str] = None,
) -> None:
    """
    Load a LoRA into the pipeline using diffusers' adapter API.

    lora_path can be:
      - a directory with *.safetensors
      - a single .safetensors file
      - a HF repo id (if you ever upload it)
    """
    kw = {
        "adapter_name": adapter_name,
    }

    if lora_path.is_file():
        # Use weight_name for a single file, if diffusers expects that
        kw["weights"] = str(lora_path)
    else:
        # Directory / repo case
        kw["pretrained_model_name_or_path"] = str(lora_path)

        if weight_name:
            kw["weight_name"] = weight_name

    # Try both modern and older signatures, to be robust
    try:
        pipe.load_lora_weights(str(lora_path), adapter_name=adapter_name, weight_name=weight_name)
    except TypeError:
        # Fallback for older / slightly different API
        pipe.load_lora_weights(str(lora_path))

    try:
        pipe.set_adapters([adapter_name], adapter_weights=[lora_scale])
    except AttributeError:
        # Older diffusers: some pipelines expose a simpler method
        if hasattr(pipe, "fuse_lora"):
            pipe.fuse_lora()
        print("[WARN] Could not set adapter weight explicitly; "
              "if results look too strong/weak, adjust the LoRA itself.")


# ------------------------- Generation ------------------------- #

def generate_images(
    pipe: ZImagePipeline,
    prompts: List[str],
    out_dir: Path,
    seed: Optional[int],
    steps: int,
    guidance_scale: float,
    width: int,
    height: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if seed is None:
        generator = None
        print("[INFO] Using random seeds (no fixed generator).")
    else:
        device = pipe._execution_device if hasattr(pipe, "_execution_device") else pipe.device
        generator = torch.Generator(device=device).manual_seed(seed)
        print(f"[INFO] Using seed={seed}")

    print(f"[INFO] Sampling {len(prompts)} prompt(s), steps={steps}, CFG={guidance_scale}, size={width}x{height}")

    result = pipe(
        prompt=prompts if len(prompts) > 1 else prompts[0],
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
    )

    images = result.images
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    for idx, (prompt, image) in enumerate(zip(prompts, images), start=1):
        safe_slug = re.sub(r"[^a-zA-Z0-9]+", "_", prompt)[:60].strip("_")
        filename = f"{timestamp}_s{seed if seed is not None else 'rand'}_{idx:02d}_{safe_slug}.png"
        path = out_dir / filename
        image.save(path)
        print(f"[SAVE] {path}")


# ------------------------- REPL ------------------------- #

def repl(
    pipe: ZImagePipeline,
    out_dir: Path,
    init_seed: Optional[int],
    steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    n_images: int,
) -> None:
    print(
        "\n[REPL] Enter prompts below.\n"
        "Commands:\n"
        "  /q                -> quit\n"
        "  /seed <int>       -> set base seed\n"
        "  /seed random      -> random seeds every time\n"
        "  /steps <int>      -> change number of steps\n"
        "  /n <int>          -> change images per prompt\n"
        "  /size WxH         -> change size (e.g. /size 768x1024)\n"
        "  /cfg <float>      -> change guidance scale (0.0 for Turbo)\n"
        "  [empty line]      -> reuse last prompt\n"
    )

    current_seed = init_seed
    current_steps = steps
    current_width = width
    current_height = height
    current_cfg = guidance_scale
    current_n = n_images

    last_prompt: Optional[str] = None

    while True:
        try:
            line = input("prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Exiting REPL.")
            break

        if not line:
            if last_prompt is None:
                print("[WARN] No previous prompt to reuse.")
                continue
            prompt = last_prompt
        elif line.startswith("/"):
            cmd, *rest = line[1:].split()
            cmd = cmd.lower()

            if cmd in {"q", "quit", "exit"}:
                print("[INFO] Bye.")
                break

            elif cmd == "seed":
                if not rest:
                    print(f"[INFO] Current seed: {current_seed}")
                    continue
                if rest[0].lower() == "random":
                    current_seed = None
                    print("[INFO] Seeds will now be random each run.")
                else:
                    try:
                        current_seed = int(rest[0])
                        print(f"[INFO] Seed set to {current_seed}")
                    except ValueError:
                        print("[ERROR] /seed expects an integer or 'random'.")

            elif cmd == "steps":
                if not rest:
                    print(f"[INFO] Current steps: {current_steps}")
                    continue
                try:
                    current_steps = int(rest[0])
                    print(f"[INFO] Steps set to {current_steps}")
                except ValueError:
                    print("[ERROR] /steps expects an integer.")

            elif cmd == "n":
                if not rest:
                    print(f"[INFO] Current images per prompt: {current_n}")
                    continue
                try:
                    current_n = max(1, int(rest[0]))
                    print(f"[INFO] Images per prompt set to {current_n}")
                except ValueError:
                    print("[ERROR] /n expects an integer.")

            elif cmd == "size":
                if not rest:
                    print(f"[INFO] Current size: {current_width}x{current_height}")
                    continue
                m = re.match(r"^(\d+)x(\d+)$", rest[0])
                if not m:
                    print("[ERROR] /size expects WxH, e.g. /size 768x1024")
                else:
                    current_width = int(m.group(1))
                    current_height = int(m.group(2))
                    print(f"[INFO] Size set to {current_width}x{current_height}")

            elif cmd == "cfg":
                if not rest:
                    print(f"[INFO] Current guidance scale: {current_cfg}")
                    continue
                try:
                    current_cfg = float(rest[0])
                    print(f"[INFO] Guidance scale set to {current_cfg}")
                except ValueError:
                    print("[ERROR] /cfg expects a float.")

            else:
                print(f"[WARN] Unknown command: /{cmd}")
            continue

        else:
            prompt = line
            last_prompt = prompt

        prompts = [prompt] * current_n
        generate_images(
            pipe=pipe,
            prompts=prompts,
            out_dir=out_dir,
            seed=current_seed,
            steps=current_steps,
            guidance_scale=current_cfg,
            width=current_width,
            height=current_height,
        )

        # If a fixed seed is used, bump it so you get new results next time
        if current_seed is not None:
            current_seed += current_n


# ------------------------- Main ------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive Z-Image Turbo sampler with LoRA (MPS-friendly)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device hint (auto → prefer MPS, then CUDA, then CPU).",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=MODEL_ID,
        help=f"HuggingFace model id (default: {MODEL_ID}).",
    )
    parser.add_argument(
        "--lora-path",
        type=Path,
        default=None,
        help="Path to LoRA file or directory (optional).",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
        help="LoRA strength (default: 1.0).",
    )
    parser.add_argument(
        "--lora-weight-name",
        type=str,
        default=None,
        help="Optional specific safetensors filename inside lora-path, if needed.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=768,
        help="Image height (default: 768).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help="Image width (default: 768).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=8,
        help="Number of inference steps (Turbo usually 6–8).",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=0.0,
        help="CFG guidance scale (Z-Image Turbo expects 0.0 by default).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_interactive"),
        help="Directory to save images.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Initial seed (use --seed -1 for random every time).",
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=1,
        help="Number of images per prompt in REPL.",
    )

    args = parser.parse_args()

    device = choose_device(args.device)
    seed: Optional[int] = args.seed if args.seed >= 0 else None

    pipe = load_pipeline(
        model_id=args.model_id,
        device=device,
        lora_path=args.lora_path,
        lora_scale=args.lora_scale,
        lora_weight_name=args.lora_weight_name,
    )

    repl(
        pipe=pipe,
        out_dir=args.output_dir,
        init_seed=seed,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        n_images=args.n_images,
    )


if __name__ == "__main__":
    main()
