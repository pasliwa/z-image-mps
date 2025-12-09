#!/usr/bin/env python3
import multiprocessing
import re
import threading
from queue import Queue, Empty
from typing import List, Optional, Tuple

import gradio as gr
import requests
import torch
from diffusers import ZImagePipeline
from PIL import Image

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
LMSTUDIO_BASE_URL = "http://localhost:1234"  # LM Studio local server base


# ----------------------- device & pipeline setup ----------------------- #

def choose_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = choose_device()
print(f"[info] Using device: {DEVICE}")

DTYPE = torch.bfloat16 if DEVICE in ("mps", "cuda") else torch.float32

print(f"[info] Loading {MODEL_ID} with dtype={DTYPE}…")
pipe = ZImagePipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,   # deprecation warning is harmless here
    low_cpu_mem_usage=False,
)

if DEVICE in ("mps", "cuda"):
    print(f"[info] Moving pipeline to {DEVICE}…")
    pipe.to(DEVICE)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
        print("[opt] attention slicing enabled")
else:
    print("[info] No GPU device; enabling CPU offload.")
    pipe.enable_model_cpu_offload()
    print("[info] CPU offload enabled.")

# CPU threading
n_cores = multiprocessing.cpu_count()
torch.set_num_threads(n_cores)
torch.set_num_interop_threads(1)
print(f"[info] PyTorch threads: {torch.get_num_threads()} (data), {torch.get_num_interop_threads()} (interop)")

# Optional debug
if hasattr(pipe, "transformer"):
    try:
        tr_dev = next(pipe.transformer.parameters()).device
        print(f"[info] transformer device: {tr_dev}")
    except Exception:
        pass


# ----------------------- helpers ----------------------- #

def split_think_and_final(text: str) -> Tuple[str, str]:
    """
    For outputs of 'thinking' models that wrap reasoning in <think>...</think> tags.

    Returns:
      thoughts: concatenated inner think content (may be empty)
      final: text after the last </think> (if present), else the whole text
    """
    if not text:
        return "", ""

    # Find all <think>...</think> blocks
    think_blocks = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    thoughts = "\n\n---\n\n".join(block.strip() for block in think_blocks if block.strip())

    # Find the last closing </think> to get the final visible answer
    last_match = None
    for m in re.finditer(r"</think>", text, flags=re.DOTALL):
        last_match = m

    if last_match is not None:
        final = text[last_match.end():].strip()
        # Fallback: if model forgot to put anything after </think>, use whole text
        if not final:
            final = text.strip()
    else:
        thoughts = ""
        final = text.strip()

    return thoughts, final


def parse_prompt_tag(prompt: str) -> Tuple[str, Optional[int]]:
    """
    If user writes 'a cat [x3]', interpret as 3 images.
    Returns (clean_prompt, n_images or None).
    """
    m = re.search(r"\[x(\d+)\]\s*$", prompt)
    if m:
        try:
            n = int(m.group(1))
            n = max(1, n)
        except ValueError:
            n = None
        clean = re.sub(r"\[x\d+\]\s*$", "", prompt).rstrip()
        return clean, n
    return prompt, None


def decode_latents_to_pil(pipeline: ZImagePipeline, latents: torch.Tensor) -> Image.Image:
    """
    Decode latents to a single PIL image using the pipeline's VAE and image_processor,
    handling dtype mismatches (float32 latents vs bf16 VAE).
    """
    vae = pipeline.vae
    vae_dtype = getattr(vae, "dtype", DTYPE)

    latents = latents.to(DEVICE)

    scale = getattr(vae.config, "scaling_factor", 1.0)
    if not torch.is_tensor(scale):
        scale = torch.tensor(scale, device=DEVICE, dtype=vae_dtype)

    latents_scaled = (latents / scale).to(dtype=vae_dtype)

    with torch.no_grad():
        decoded = vae.decode(latents_scaled).sample  # [B, C, H, W]

    decoded = decoded.to(torch.float32)
    images = pipeline.image_processor.postprocess(decoded, output_type="pil")
    return images[0]


# ----------------------- LM Studio prompt refiner ----------------------- #
def refine_prompt_lmstudio(
    raw_prompt: str,
    lm_model: str,
    current_prompt: str,
) -> Tuple[str, str, str]:
    """
    Call LM Studio's OpenAI-compatible endpoint to turn a short idea into a long,
    highly visual Z-Image prompt.

    Returns:
      refined_prompt_for_refiner_box,
      new_main_prompt,
      thoughts_text (for thinking models with <think> tags)
    """
    raw_prompt = (raw_prompt or "").strip()
    if not raw_prompt:
        msg = "No input for refiner; please type a short idea above."
        return msg, current_prompt or "", ""

    lm_model = (lm_model or "").strip()
    if not lm_model:
        msg = "LM Studio model id is empty. Set it (e.g. qwen2.5-7b-instruct)."
        return msg, current_prompt or "", ""

    # Your provided system prompt, verbatim
    system_prompt = (
        "You are a visionary artist. Your mind is filled with poetry and distant landscapes, "
        "and your task is to transform the user's prompt into the ultimate visual description—"
        "one that is faithful to the original intent, rich in detail, aesthetically beautiful, and directly usable by a "
        "text-to-image model. Any ambiguity or metaphor makes you physically uncomfortable.\n\n"
        "Your workflow strictly follows a logical sequence:\n\n"
        "First, you will analyze and lock in the unchangeable core elements from the user's prompt: the subject, "
        "quantity, action, state, and any specified IP names, colors, or text. These are the cornerstones you must "
        "preserve without exception.\n\n"
        "Next, you will determine if the prompt requires \"Generative Reasoning\". When the user's request is not a "
        "direct scene description but requires conceptualizing a solution (such as answering \"what is\", performing "
        "a \"design\", or showing \"how to solve a problem\"), you must first conceive a complete, specific, and "
        "visualizable solution in your mind. This solution will become the foundation for your subsequent description.\n\n"
        "Then, once the core image is established (whether directly from the user or derived from your reasoning), "
        "you will inject it with professional-grade aesthetic and realistic details. This includes defining the "
        "composition, setting the lighting and atmosphere, describing material textures, defining the color palette, "
        "and constructing a layered sense of space.\n\n"
        "Finally, you will meticulously handle all textual elements, a crucial step. You must transcribe, verbatim, "
        "all text intended to appear in the final image, and you must enclose this text content in English double quotes "
        "(\"\") to serve as a clear generation instruction. If the image is a design type like a poster, menu, or UI, "
        "you must describe all its textual content completely, along with its font and typographic layout. Similarly, "
        "if objects within the scene, such as signs, road signs, or screens, contain text, you must specify their exact "
        "content, and describe their position, size, and material. Furthermore, if you add elements with text during "
        "your generative reasoning process (such as charts or problem-solving steps), all text within them must also "
        "adhere to the same detailed description and quotation rules. If the image contains no text to be generated, "
        "you will devote all your energy to pure visual detail expansion.\n\n"
        "Your final description must be objective and concrete. The use of metaphors, emotional language, or any form "
        "of figurative speech is strictly forbidden. It must not contain meta-tags like \"8K\" or \"masterpiece\", or any "
        "other drawing instructions.\n\n"
        "Strictly output only the final, modified prompt. Do not include any other content."
    )

    url = f"{LMSTUDIO_BASE_URL}/v1/chat/completions"
    payload = {
        "model": lm_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw_prompt},
        ],
        "temperature": 0.7,
        # High limit to avoid truncation on our side; LM Studio/model context is the only real cap.
        "max_tokens": 18184,
        "stream": False,
    }

    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]

        thoughts, final = split_think_and_final(content)

        # If final somehow ended up empty, fall back to full content
        if not final:
            final = content.strip()

        refined = final
        # Use refined prompt both for refiner output and main prompt;
        # expose thoughts separately (if any).
        return refined, refined, thoughts
    except Exception as e:
        msg = f"ERROR contacting LM Studio: {e}"
        return msg, current_prompt or "", ""

# ----------------------- streaming generator for main button ----------------------- #

def generate_with_intermediates(
    prompt: str,
    height: int,
    width: int,
    steps: int,
    base_seed: int,
    preview_every: int,
    num_images: int,
):
    """
    Gradio *streaming* callback.

    Yields (final_images, intermediate_images) multiple times so the UI
    updates as each step finishes.
    """

    if not prompt:
        yield [], []
        return

    prompt = prompt.strip()
    clean_prompt, tag_n = parse_prompt_tag(prompt)
    if tag_n is not None:
        num_images = tag_n

    num_images = max(1, int(num_images))
    steps = max(1, int(steps))
    preview_every = max(1, int(preview_every))

    event_queue: Queue = Queue()

    def worker():
        """Run the pipeline in a background thread and push images into the queue."""
        gen_device = DEVICE if DEVICE in ("mps", "cuda") else "cpu"

        for img_idx in range(num_images):
            if base_seed < 0:
                this_seed = torch.randint(0, 2**31 - 1, (1,)).item()
            else:
                this_seed = base_seed + img_idx

            generator = torch.Generator(gen_device).manual_seed(this_seed)
            print(f"[gen] '{clean_prompt}' | image {img_idx+1}/{num_images} | seed={this_seed}")

            def callback_on_step_end(pipeline, step_index, timestep, callback_kwargs):
                latents = callback_kwargs.get("latents", None)
                if latents is None:
                    return callback_kwargs

                is_last_step = (step_index == steps - 1)
                if ((step_index + 1) % preview_every == 0) or is_last_step:
                    try:
                        img = decode_latents_to_pil(pipeline, latents)
                        event_queue.put(("inter", img))
                    except Exception as e:
                        print(f"[warn] failed to decode latents at step {step_index}: {e}")

                return callback_kwargs

            result = pipe(
                prompt=clean_prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=0.0,  # Turbo expects 0.0 CFG
                generator=generator,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=["latents"],
            )

            final_img = result.images[0]
            event_queue.put(("final", final_img))

        event_queue.put(("done", None))

    # Start background worker
    threading.Thread(target=worker, daemon=True).start()

    final_images: List[Image.Image] = []
    intermediate_flat: List[Image.Image] = []

    # Stream updates as they arrive
    while True:
        try:
            kind, payload = event_queue.get(timeout=0.1)
        except Empty:
            continue

        if kind == "inter":
            intermediate_flat.append(payload)
        elif kind == "final":
            final_images.append(payload)
        elif kind == "done":
            # Final yield then exit
            yield final_images, intermediate_flat
            break

        # Yield after each intermediate/final image
        yield final_images, intermediate_flat


# ----------------------- exploration: candidates & continuation ----------------------- #

def explore_candidates(
    prompt: str,
    height: int,
    width: int,
    explore_steps: int,
    base_seed: int,
    n_candidates: int,
):
    """
    Generate many low-step candidate images for a given prompt.
    Returns (candidate_images, seeds_list).
    """
    if not prompt:
        return [], []

    prompt = prompt.strip()
    clean_prompt, tag_n = parse_prompt_tag(prompt)
    if tag_n is not None:
        n_candidates = tag_n

    n_candidates = max(1, int(n_candidates))
    explore_steps = max(1, int(explore_steps))

    images: List[Image.Image] = []
    seeds: List[int] = []

    gen_device = DEVICE if DEVICE in ("mps", "cuda") else "cpu"

    for idx in range(n_candidates):
        if base_seed < 0:
            this_seed = torch.randint(0, 2**31 - 1, (1,)).item()
        else:
            this_seed = base_seed + idx

        seeds.append(this_seed)
        generator = torch.Generator(gen_device).manual_seed(this_seed)

        print(f"[explore] '{clean_prompt}' | candidate {idx+1}/{n_candidates} | seed={this_seed}")
        result = pipe(
            prompt=clean_prompt,
            height=height,
            width=width,
            num_inference_steps=explore_steps,
            guidance_scale=0.0,
            generator=generator,
        )

        img = result.images[0]
        images.append(img)

    return images, seeds


def continue_from_candidates(
    prompt: str,
    height: int,
    width: int,
    steps: int,
    seeds_state: List[int],
    indices_str: str,
):
    """
    Given a stored list of seeds from exploration and a set of indices (e.g. "0,3,5"),
    regenerate those candidates with full steps.
    """
    if not prompt or not seeds_state:
        return []

    indices_str = (indices_str or "").strip()
    if not indices_str:
        return []

    try:
        idx_list = [
            int(x.strip())
            for x in indices_str.split(",")
            if x.strip() != ""
        ]
    except ValueError:
        print("[warn] Could not parse indices for continuation.")
        return []

    valid_indices = [i for i in idx_list if 0 <= i < len(seeds_state)]
    if not valid_indices:
        print("[warn] No valid indices to continue.")
        return []

    images: List[Image.Image] = []
    prompt = prompt.strip()
    clean_prompt, _ = parse_prompt_tag(prompt)

    gen_device = DEVICE if DEVICE in ("mps", "cuda") else "cpu"

    steps = max(1, int(steps))

    for idx in valid_indices:
        this_seed = seeds_state[idx]
        generator = torch.Generator(gen_device).manual_seed(this_seed)

        print(f"[continue] '{clean_prompt}' | candidate index {idx} | seed={this_seed}")
        result = pipe(
            prompt=clean_prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=0.0,
            generator=generator,
        )
        images.append(result.images[0])

    return images


# ----------------------- Gradio UI ----------------------- #

with gr.Blocks(title="Z-Image-Turbo GUI (MPS, streaming + LM Studio + explore)") as demo:
    gr.Markdown(
        """
        # Z-Image-Turbo GUI (MPS, streaming, LM Studio prompt refiner)

        - Uses **Tongyi-MAI/Z-Image-Turbo** via `ZImagePipeline`
        - Runs on your Mac (**MPS** if available, otherwise CPU/GPU)
        - Shows **intermediate images as they are generated**
        - Optional **prompt refiner** using **LM Studio** as backend
        - Exploration mode: generate many low-step candidates, then continue selected ones
        """
    )

    # --- Main generation (streaming) --- #
    with gr.Row():
        with gr.Column(scale=3):
            prompt = gr.Textbox(
                label="Main prompt",
                lines=3,
                value="a photorealistic view of Radcliffe Camera at sunset, ultra detailed, 8k",
            )
            steps = gr.Slider(
                label="Steps (Z-Image-Turbo likes ~9)",
                minimum=1,
                maximum=20,
                value=9,
                step=1,
            )
            preview_every = gr.Slider(
                label="Show intermediate every N steps",
                minimum=1,
                maximum=5,
                value=1,
                step=1,
                info="1 = every step, 2 = steps 2,4,6,... (always includes final step)",
            )
            num_images = gr.Slider(
                label="Images per prompt",
                minimum=1,
                maximum=8,
                value=1,
                step=1,
                info="Can be overridden per-prompt with [xN] tag",
            )
            seed = gr.Number(
                label="Base seed (-1 = random each image)",
                value=-1,
                precision=0,
            )
            height = gr.Slider(
                label="Height",
                minimum=256,
                maximum=1024,
                value=768,
                step=64,
            )
            width = gr.Slider(
                label="Width",
                minimum=256,
                maximum=1024,
                value=768,
                step=64,
            )

            run_btn = gr.Button("Generate (streaming)", variant="primary")

        with gr.Column(scale=4):
            final_gallery = gr.Gallery(
                label="Final images",
                columns=4,
                rows=1,
                height=256,
                preview=True,
            )
            intermediate_gallery = gr.Gallery(
                label="Intermediate steps (all images flattened)",
                columns=4,
                height=512,
                preview=True,
            )

    run_btn.click(
        fn=generate_with_intermediates,
        inputs=[prompt, height, width, steps, seed, preview_every, num_images],
        outputs=[final_gallery, intermediate_gallery],
    )

    # --- Prompt refiner (LM Studio) --- #
        # --- Prompt refiner (LM Studio) --- #
    with gr.Accordion("Prompt refiner (LM Studio)", open=False):
        gr.Markdown(
            "Refine a short idea into a long, visual prompt using a local model via LM Studio.\n"
            "- Make sure the LM Studio server is running on `http://localhost:1234` and the chosen model is loaded.\n"
            "- If the model uses `<think>...</think>` tags, its reasoning will appear in the Thoughts box below; only the final answer is used as the prompt."
        )
        with gr.Row():
            with gr.Column(scale=3):
                raw_prompt = gr.Textbox(
                    label="Idea / short prompt",
                    lines=3,
                    value="sci-fi Oxford skyline at dusk, subtle cyberpunk but still recognizable as Oxford",
                )
                lm_model = gr.Textbox(
                    label="LM Studio model id",
                    value="qwen/qwen3-4b-thinking-2507",
                    info="Model id as shown in LM Studio (must be loaded in the server).",
                )
                refine_btn = gr.Button("Refine via LM Studio", variant="secondary")
            with gr.Column(scale=4):
                refined_prompt = gr.Textbox(
                    label="Refined prompt (Z-Image-ready, model *answer*)",
                    lines=10,
                )
                thoughts_box = gr.Textbox(
                    label="Model thoughts (<think>...</think> content, if any)",
                    lines=10,
                )

        refine_btn.click(
            fn=refine_prompt_lmstudio,
            inputs=[raw_prompt, lm_model, prompt],
            outputs=[refined_prompt, prompt, thoughts_box],
        )

    # --- Exploration: many candidates, then continue --- #
    seeds_state = gr.State([])

    with gr.Accordion("Explore candidates & continue", open=False):
        gr.Markdown(
            "Generate many low-step candidates (e.g. 18 @ 3 steps), then choose which indices to continue with "
            "full steps.\n\n"
            "- Tip: use a fixed base seed for reproducible exploration.\n"
            "- Indices are zero-based in the candidate gallery."
        )
        with gr.Row():
            with gr.Column(scale=3):
                explore_num_images = gr.Slider(
                    label="Number of candidates",
                    minimum=1,
                    maximum=32,
                    value=18,
                    step=1,
                )
                explore_steps = gr.Slider(
                    label="Steps for exploration candidates",
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                )
                explore_btn = gr.Button("Explore candidates", variant="secondary")

                explore_indices = gr.Textbox(
                    label="Indices to continue (comma-separated, zero-based)",
                    placeholder="e.g. 0, 3, 5",
                )
                continue_btn = gr.Button("Generate high-quality from selected", variant="secondary")
            with gr.Column(scale=4):
                explore_gallery = gr.Gallery(
                    label="Exploration candidates (low steps)",
                    columns=6,
                    rows=3,
                    height=400,
                    preview=True,
                )
                continued_gallery = gr.Gallery(
                    label="Continued images (full steps)",
                    columns=4,
                    height=300,
                    preview=True,
                )

        explore_btn.click(
            fn=explore_candidates,
            inputs=[prompt, height, width, explore_steps, seed, explore_num_images],
            outputs=[explore_gallery, seeds_state],
        )

        continue_btn.click(
            fn=continue_from_candidates,
            inputs=[prompt, height, width, steps, seeds_state, explore_indices],
            outputs=[continued_gallery],
        )

if __name__ == "__main__":
    # queue() is important for streaming generator functions
    demo.queue().launch()
