#!/usr/bin/env python3
import argparse
import base64
import os
import shutil
import sys
import textwrap
from pathlib import Path

import requests

DEFAULT_BASE_URL = "http://localhost:1234/v1"
CHAT_ENDPOINT = "/chat/completions"


def build_system_prompt(token: str, class_word: str) -> str:
    return textwrap.dedent(f"""
    You are an expert vision-language assistant that writes image captions
    for training a character LoRA for a diffusion model (Z-Image Turbo).

    Goal:
    - For each image, produce a single compact caption that is ideal for LoRA training.

    Rules:
    - ALWAYS include the exact trigger token "{token}" and the class word "{class_word}" exactly once,
      near the start, in the pattern:
      "{token} {class_word}, ..."
    - The trigger token MUST remain all lowercase.
    - Focus ONLY on:
      - pose (standing, sitting, walking, hands crossed, etc.)
      - position (side-view, full body picture, etc.)
      - facial expression (smiling, sad, etc.)
      - clothing (type, color, style)
      - environment / background (room, café, street, park, etc.)
      - lighting (daylight, warm indoor light, evening, etc.)
      - photo type (selfie, portrait etc.)
      - note other people and their position relative to {token}



    - DO NOT mention:
      - facial features (eyes, nose, mouth, jaw, face shape)
      - hair color or hair length or hairstyle
      - skin color or skin tone
      - age, race, ethnicity, body shape, or weight

    - Keep the caption:
      - One single line
      - Natural language (not tag lists)
      - Roughly 15–40 tokens
    - No markdown, no quotes, no explanations, no prefixes like "Caption:" –
      output ONLY the caption text.

Examples:
1) {token} {class_word} sitting at a table, face and upper body image, smiling and looking forward, wearing long-sleeved shirt, in a dark-lit restaurant, with other people in the background, portrait
2) {token} {class_word} smiling, side view of face only, close-up, wearing red t-shirt with a background of a forest, portrait
3) {token} {class_word} standing in a formal suit, full body photo, serious facial expression, not looking at the camera, in a crowded room, with people around him
    """)


def encode_image_to_data_url(image_path: Path) -> str:
    """Encode image as base64 data URL suitable for OpenAI-style image_url."""
    mime = "image/jpeg"
    if image_path.suffix.lower() == ".png":
        mime = "image/png"
    elif image_path.suffix.lower() == ".webp":
        mime = "image/webp"

    with image_path.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def call_lm_studio_vlm(
    base_url: str,
    model: str,
    system_prompt: str,
    original_caption: str,
    image_path: Path,
    temperature: float = 0.2,
    max_tokens: int = 96,
) -> str:
    """
    Call LM Studio's OpenAI-compatible /v1/chat/completions endpoint
    with a VLM model and a single image.
    """
    url = base_url.rstrip("/") + CHAT_ENDPOINT

    image_url = encode_image_to_data_url(image_path)

    user_text = (
        "Look at the image and write a single compact caption that follows the rules in the system prompt.\n"
        "If there is an original caption provided, you may use it as a hint, but trust the visual information "
        "more than the text if they disagree.\n\n"
        f"Original caption (may be empty or noisy):\n{original_caption.strip()}"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        resp = requests.post(url, json=payload, timeout=120)
    except requests.RequestException as e:
        print(f"[ERROR] Request to LM Studio failed: {e}", file=sys.stderr)
        raise

    if resp.status_code != 200:
        print(f"[ERROR] LM Studio returned HTTP {resp.status_code}: {resp.text}", file=sys.stderr)
        raise RuntimeError(f"Bad status code: {resp.status_code}")

    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        print(f"[ERROR] Unexpected response structure from LM Studio: {data}", file=sys.stderr)
        raise RuntimeError("Invalid response structure") from e

    return content.strip()


def recaption_directory(
    input_dir: Path,
    output_dir: Path,
    base_url: str,
    model: str,
    token: str,
    class_word: str,
    overwrite: bool = False,
    dry_run: bool = False,
):
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist or is not a directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    system_prompt = build_system_prompt(token, class_word)

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])

    if not images:
        print(f"[WARN] No images found in {input_dir}")
        return

    print(f"[INFO] Found {len(images)} images in {input_dir}")
    print(f"[INFO] Writing renamed images and captions to {output_dir}")
    print(f"[INFO] Using LM Studio at {base_url}, model: {model}")
    print(f"[INFO] Token: {token}, class word: {class_word}")
    if dry_run:
        print("[INFO] DRY RUN: will NOT write files or call LM Studio.")

    for idx, img_path in enumerate(images, start=1):
        new_basename = f"image{idx:04d}"
        new_img_name = new_basename + img_path.suffix.lower()
        new_img_path = output_dir / new_img_name
        new_txt_path = output_dir / f"{new_basename}.txt"

        print(f"\n[INFO] Processing {img_path.name} -> {new_img_name}")

        # Copy image
        if not dry_run:
            if new_img_path.exists() and not overwrite:
                print(f"[WARN] {new_img_path} exists and overwrite=False; skipping image copy.")
            else:
                shutil.copy2(img_path, new_img_path)

        # Optional original caption (same stem, .txt)
        original_txt_path = img_path.with_suffix(".txt")
        if original_txt_path.is_file():
            original_caption = original_txt_path.read_text(encoding="utf-8")
            print(f"[INFO] Found existing caption: {original_txt_path.name}")
        else:
            original_caption = ""
            print(f"[INFO] No existing caption; will caption from image only.")

        if dry_run:
            print("[DRY RUN] Would call LM Studio VLM with this image and caption:")
            print("---------- ORIGINAL CAPTION ----------")
            print(original_caption.strip())
            print("---------- (END) ---------------------")
            continue

        try:
            new_caption = call_lm_studio_vlm(
                base_url=base_url,
                model=model,
                system_prompt=system_prompt,
                original_caption=original_caption,
                image_path=img_path,
            )
        except Exception as e:
            print(f"[ERROR] Failed to generate caption for {img_path.name}: {e}", file=sys.stderr)
            # Fallback: very basic caption with just token
            new_caption = f"photo of {token} {class_word}"

        # Write new caption
        if new_txt_path.exists() and not overwrite:
            print(f"[WARN] Caption {new_txt_path} exists and overwrite=False; skipping write.")
        else:
            new_txt_path.write_text(new_caption + "\n", encoding="utf-8")
            print(f"[INFO] Wrote caption -> {new_txt_path.name}")
            print(f"[CAPTION] {new_caption}")


def main():
    parser = argparse.ArgumentParser(
        description="Recaption LoRA dataset using a local VLM model in LM Studio."
    )
    parser.add_argument("input_dir", type=Path, help="Directory with input images (+ optional .txt captions).")
    parser.add_argument("output_dir", type=Path, help="Directory to write renamed images + new captions.")
    parser.add_argument(
        "--model",
        required=True,
        help="Model identifier as shown in LM Studio.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Base URL for LM Studio server (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--token",
        required=True,
        help="Unique trigger token to insert into captions.",
    )
    parser.add_argument(
        "--class-word",
        default="person",
        help="Class word for the subject (default: 'person'; e.g. 'man', 'woman').",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in output_dir.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write files or call LM Studio; just print what would happen.",
    )

    args = parser.parse_args()

    try:
        recaption_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            base_url=args.base_url,
            model=args.model,
            token=args.token,
            class_word=args.class_word,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
