#!/usr/bin/env python3
import argparse
import datetime as dt
from pathlib import Path
from typing import Optional, List, Tuple

import multiprocessing
import re
import torch
from diffusers import ZImagePipeline  # make sure diffusers is recent

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"


def choose_device(device_arg: str) -> str:
    """Choose device with preference: user > MPS > CUDA > CPU."""
    if device_arg != "auto":
        return device_arg
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_full_pipeline(
    device: str,
    use_attn_slicing: bool = True,
) -> ZImagePipeline:
    """
    Build the full Z-Image-Turbo pipeline in BF16, tuned for MPS, without torch.compile.
    """
    # BF16 on GPU, FP32 on CPU
    dtype = torch.bfloat16 if device in ("mps", "cuda") else torch.float32

    print(f"[info] Loading {MODEL_ID} with dtype={dtype}…")
    pipe = ZImagePipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,  # may warn that torch_dtype is deprecated; harmless for now
        low_cpu_mem_usage=False,
    )

    if device in ("mps", "cuda"):
        print(f"[info] Moving pipeline to {device}…")
        pipe.to(device)

        if use_attn_slicing and hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
            print("[opt] attention slicing enabled")

        # Debug: confirm transformer device
        if hasattr(pipe, "transformer"):
            try:
                tr_dev = next(pipe.transformer.parameters()).device
                print(f"[info] transformer device: {tr_dev}")
            except Exception:
                pass

    else:
        print("[info] No GPU device selected; using CPU offload.")
        pipe.enable_model_cpu_offload()
        print("[info] CPU offload enabled.")

    return pipe


def save_image(image, out_dir: Path, prefix: str = "zimage_full", ext: str = "png") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = out_dir / f"{prefix}_{ts}.{ext}"
    image.save(path)
    return path


def parse_prompt_tag(prompt: str, default_n: int) -> Tuple[str, int]:
    """
    Look for a trailing '[xN]' tag and return (clean_prompt, n_images).

    Example:
      "a cat [x3]" -> ("a cat", 3)
      "a cat"      -> ("a cat", default_n)
    """
    m = re.search(r"\[x(\d+)\]\s*$", prompt)
    if m:
        try:
            n = int(m.group(1))
            n = max(1, n)
        except ValueError:
            n = default_n
        clean = re.sub(r"\[x\d+\]\s*$", "", prompt).rstrip()
        return clean, n
    return prompt, default_n


def generate_images(
    pipe: ZImagePipeline,
    prompt: str,
    device: str,
    height: int,
    width: int,
    steps: int,
    guidance_scale: float,
    negative_prompt: Optional[str],
    seed: Optional[int],
    n_images: int,
    out_dir: Path,
) -> List[Path]:
    """
    Generate n_images for a given prompt, using different seeds if n_images > 1.
    If seed is not None, uses seed, seed+1, seed+2, ...
    If seed is None, uses fresh random seeds for each image.
    """
    paths: List[Path] = []

    # Base device for the generator
    gen_device = device if device in ("mps", "cuda") else "cpu"

    for i in range(n_images):
        if seed is None:
            # Random seed per image
            this_seed = torch.randint(0, 2**31 - 1, (1,)).item()
        else:
            this_seed = seed + i

        gen = torch.Generator(gen_device).manual_seed(this_seed)

        print(f"[gen] Prompt='{prompt}' | image {i+1}/{n_images} | seed={this_seed}")
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=gen,
        )
        image = result.images[0]
        out_path = save_image(image, out_dir)
        print(f"[done] Saved → {out_path}")
        paths.append(out_path)

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Z-Image-Turbo full model (BF16) tuned for MPS (no torch.compile)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device hint (auto → prefer MPS, then CUDA, then CPU).",
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
        default=9,
        help="Number of diffusion steps (default: 9; gives 8 DiT forwards for Z-Image-Turbo).",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=0.0,
        help="Guidance scale (Turbo expects 0.0; CFG/negative prompts are effectively ignored).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed (default: random per generation).",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Number of images per prompt (default: 1). In REPL, can be changed with /n.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("zimage_full_outputs"),
        help="Directory to save images.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="One-shot prompt. If omitted, enter interactive REPL. "
             "You can append '[xN]' to the prompt to override --num-images.",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Optional negative prompt (note: Turbo ignores negatives).",
    )
    parser.add_argument(
        "--no-attn-slicing",
        action="store_true",
        help="Disable attention slicing.",
    )

    args = parser.parse_args()

    # Use all CPU cores where CPU work happens
    n_cores = multiprocessing.cpu_count()
    torch.set_num_threads(n_cores)
    torch.set_num_interop_threads(1)
    print(f"[info] PyTorch threads: {torch.get_num_threads()} (data), {torch.get_num_interop_threads()} (interop)")

    device = choose_device(args.device)
    print(f"[info] Device hint: {device}")

    pipe = build_full_pipeline(
        device=device,
        use_attn_slicing=not args.no_attn_slicing,
    )
    pipe.set_progress_bar_config(disable=False)
    print("[info] Pipeline ready. Recommended: guidance_scale=0.0, steps=9 by default.")
    print("[info] You can use '[xN]' at the end of a prompt to generate N images (e.g. 'a cat [x3]').")

    # ---------- One-shot mode ---------- #
    if args.prompt is not None:
        prompt, n_images = parse_prompt_tag(args.prompt, args.num_images)
        generate_images(
            pipe=pipe,
            prompt=prompt,
            device=device,
            height=args.height,
            width=args.width,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            n_images=n_images,
            out_dir=args.output_dir,
        )
        return

    # ---------- Interactive REPL mode ---------- #

    print(
        "\n[REPL] Enter prompts below.\n"
        "Commands:\n"
        "  /q                -> quit\n"
        "  /seed <int>       -> set base seed\n"
        "  /seed random      -> random seeds every time\n"
        "  /height <int>     -> change image height\n"
        "  /width <int>      -> change image width\n"
        "  /steps <int>      -> change number of steps\n"
        "  /n <int>          -> change images per prompt\n"
        "You can also append '[xN]' to a prompt to override /n for that prompt.\n"
    )

    current_seed: Optional[int] = args.seed
    current_height = args.height
    current_width = args.width
    current_steps = args.steps
    current_n_images = max(1, args.num_images)

    while True:
        try:
            line = input("\nPrompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[info] Bye.")
            break

        if not line:
            continue

        if line.lower().startswith("/q"):
            print("[info] Bye.")
            break

        if line.lower().startswith("/seed"):
            parts = line.split()
            if len(parts) == 1 or parts[1].lower() == "random":
                current_seed = None
                print("[seed] random per generation")
            else:
                try:
                    current_seed = int(parts[1])
                    print(f"[seed] fixed → {current_seed}")
                except ValueError:
                    print("Usage: /seed <int> or /seed random")
            continue

        if line.lower().startswith("/height"):
            parts = line.split()
            if len(parts) != 2:
                print("Usage: /height <int>")
                continue
            try:
                current_height = int(parts[1])
                print(f"[cfg] height → {current_height}")
            except ValueError:
                print("Usage: /height <int>")
            continue

        if line.lower().startswith("/width"):
            parts = line.split()
            if len(parts) != 2:
                print("Usage: /width <int>")
                continue
            try:
                current_width = int(parts[1])
                print(f"[cfg] width → {current_width}")
            except ValueError:
                print("Usage: /width <int>")
            continue

        if line.lower().startswith("/steps"):
            parts = line.split()
            if len(parts) != 2:
                print("Usage: /steps <int>")
                continue
            try:
                current_steps = int(parts[1])
                print(f"[cfg] steps → {current_steps}")
            except ValueError:
                print("Usage: /steps <int>")
            continue

        if line.lower().startswith("/n"):
            parts = line.split()
            if len(parts) != 2:
                print("Usage: /n <int>")
                continue
            try:
                current_n_images = max(1, int(parts[1]))
                print(f"[cfg] images per prompt → {current_n_images}")
            except ValueError:
                print("Usage: /n <int>")
            continue

        # Normal prompt
        prompt, n_images = parse_prompt_tag(line, current_n_images)
        generate_images(
            pipe=pipe,
            prompt=prompt,
            device=device,
            height=current_height,
            width=current_width,
            steps=current_steps,
            guidance_scale=args.guidance_scale,
            negative_prompt=args.negative_prompt,
            seed=current_seed,
            n_images=n_images,
            out_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
