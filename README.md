# Z-Image Turbo Tools (Mac M-series + MPS)

Local tooling around **[Tongyi-MAI / Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)**, tuned for Apple Silicon (M-series) Macs:

- 🧠 **Full-model CLI** (BF16 on MPS) with interactive REPL.
- 🎛️ **Streaming GUI** (Gradio) that:
  - Shows **intermediate denoising steps live** as they are generated.
  - Integrates a **prompt refiner** using **LM Studio** (OpenAI-compatible, supports `<think>...</think>` “thinking” models).
  - Supports an **“explore → pick → continue”** workflow: generate many low-step candidates, then refine selected seeds.

Optimized for a MacBook Pro with M3 + MPS, but should work on other Apple Silicon Macs with a recent PyTorch + diffusers stack.

---

## Files

- `z_image_full_mps.py`  
  Full-model BF16 **CLI/REPL** script (runs on MPS where available; falls back to CPU/CUDA).
- `z_image_full_gui_stream.py`  
  **Gradio GUI** with:
  - Streaming intermediate images.
  - LM Studio prompt refiner with `<think>` handling.
  - Candidate exploration + continuation.

Rename as desired, but the examples below assume these filenames.

---

## Quick Start

### 1. Create environment & install deps

```bash
python3 -m venv zimage-env
source zimage-env/bin/activate
python -m pip install --upgrade pip

pip install torch torchvision torchaudio
pip install diffusers transformers accelerate safetensors
pip install pillow gradio requests
```

If the model is gated on Hugging Face (not at the time of upload):

```bash
pip install huggingface_hub
huggingface-cli login
```

### 2. CLI: one-shot generation

```bash
source zimage-env/bin/activate

python z_image_full_mps.py \
  --device mps \
  --height 768 \
  --width 768 \
  --steps 9 \
  --prompt "a photorealistic view of Radcliffe Camera at sunset, detailed, natural lighting"
```

Images are saved to `zimage_full_outputs/` with timestamped filenames.

### 3. GUI: streaming + LM Studio + explore

```bash
source zimage-env/bin/activate
python z_image_full_gui_stream.py
```

Then open the printed URL, e.g.:

```text
http://127.0.0.1:7860
```

---

<details>
<summary><strong>Environment & Requirements (details)</strong></summary>

### Hardware

- Apple Silicon Mac (M1 / M2 / M3).
- Enough RAM for your target resolution (e.g. 32 GB+ is comfortable for 768–1024 px).

### Software

- **Python** ≥ 3.10 (tested with 3.12).
- **pip** current.

Python packages (in the venv):

- `torch` (with MPS support; standard macOS arm64 wheels are fine).
- `torchvision`, `torchaudio` (optional but commonly installed together).
- `diffusers`
- `transformers`
- `accelerate`
- `safetensors`
- `Pillow`
- `gradio`
- `requests`

### Hugging Face

If the model requires it:

- Accept the license on the model page.
- Log in via:

```bash
pip install huggingface_hub
huggingface-cli login
```

</details>

---

## CLI: `z_image_full_mps.py`

### What it does

- Loads `Tongyi-MAI/Z-Image-Turbo` via `ZImagePipeline`.
- Uses **BF16** on **MPS** when available; falls back to CPU or CUDA.
- Uses Z-Image-Turbo’s recommended defaults:
  - `num_inference_steps = 9`  
    → gives 8 DiT transformer passes internally.
  - `guidance_scale = 0.0`  
    → Turbo is distilled with conditioning baked in; CFG + negative prompts are effectively ignored.

### One-shot examples

Single image:

```bash
python z_image_full_mps.py \
  --device mps \
  --height 768 \
  --width 768 \
  --steps 9 \
  --prompt "a modern gym interior with squat rack and rowing machine, natural lighting, editorial style"
```

Smaller, faster image:

```bash
python z_image_full_mps.py \
  --device mps \
  --height 512 \
  --width 512 \
  --steps 9 \
  --prompt "a cyberpunk Oxford skyline at night, neon reflections on wet streets"
```

### REPL mode

Start:

```bash
python z_image_full_mps.py --device mps
```

Available commands inside the REPL:

- `/q` — quit.
- `/seed <int>` — fixed seed (e.g. `/seed 123`).
- `/seed random` — random seed per generation.
- `/height <int>` — change height.
- `/width <int>` — change width.
- `/steps <int>` — change number of diffusion steps.
- `/n <int>` — change number of images per prompt.
- Append `[xN]` at the end of a prompt to override `/n` for that prompt only.

Example session:

```text
Prompt> /height 512
[cfg] height → 512
Prompt> /width 512
[cfg] width → 512
Prompt> /n 3
[cfg] images per prompt → 3
Prompt> a cyberpunk Oxford skyline at night [x5]
# → 5 images from that prompt, each with a different seed
Prompt> /seed 123
[seed] fixed → 123
Prompt> a cozy reading corner in the British Library, warm lamp light, rain on windows
# → 3 images, seeds 123, 124, 125
```

---

## GUI: `z_image_full_gui_stream.py`

The GUI provides:

1. **Streaming generation** with intermediate images.
2. **Prompt refiner** using **LM Studio** (with `<think>` support).
3. **Explore → pick → continue** workflow.

### Streaming denoising

- The main “Generate (streaming)” button:
  - Runs the Z-Image pipeline in a background thread.
  - Uses `callback_on_step_end` to decode latents at selected steps.
  - Updates two galleries in real time:
    - **Final images**.
    - **Intermediate steps** (flattened across all images).

Controls:

- `Steps` — total diffusion steps (Z-Image recommends ~9).
- `Show intermediate every N steps` — controls how often intermediates are decoded (e.g. 1 = every step, 2 = 2,4,6,…).
- `Images per prompt` — default number of images; can be overridden by `[xN]` tag.
- `Base seed (-1 = random)` — negative for random each time; otherwise `base_seed + index` for image index.
- `Height` / `Width` — resolution sliders.

---

<details>
<summary><strong>LM Studio Prompt Refiner</strong></summary>

The **Prompt Refiner (LM Studio)** accordion lets you turn a short idea into a long, concrete, Z-Image-ready prompt using a local model.

### How it works

- Talks to LM Studio’s OpenAI-compatible API:

  - Base URL: `http://localhost:1234`
  - Endpoint: `/v1/chat/completions`

- Sends a detailed **system prompt** (embedded in the code) that:

  - Forces a very **visual, concrete** description.
  - Preserves core elements from the user’s prompt (subject, quantity, actions, specific names, colors, text).
  - Performs “Generative Reasoning” when needed (e.g. for design / “how to” prompts) to construct a visualizable solution first.
  - Requires all text in the image to be:
    - Quoted using `"..."`.
    - Described precisely (position, size, material, font-like style, etc.) when present.
  - Forbids:
    - Metaphors, emotional language, figurative speech.
    - Tags like `8K`, `masterpiece`, or other generic quality tags.
  - Instructs the model to **strictly output only the final modified prompt** (no commentary).


### `<think>...</think>` support

If your LM Studio model is a **thinking model** that emits `<think>...</think>`:

- All `<think>...</think>` content is parsed out and shown in a **“Model thoughts”** textbox.
- Only the text **after the last `</think>`** is treated as the **refined prompt** and:
  - Displayed in the **“Refined prompt (Z-Image-ready)”** box.
  - Automatically copied into the main **“Main prompt”** field used for image generation.

If the model does not use `<think>` tags, the entire output is treated as the final prompt.

### How to use

1. Start **LM Studio**, load a model, and start the local server on `http://localhost:1234`.
2. In the GUI:
   - Open **“Prompt refiner (LM Studio)”**.
   - Fill **“Idea / short prompt”** with your brief idea.
   - Set **“LM Studio model id”** to the name shown in LM Studio (e.g. `qwen2.5-7b-instruct` or a `*-thinking` variant).
   - Click **“Refine via LM Studio”**.
3. Inspect:
   - **Refined prompt (Z-Image-ready)** — the final, concrete prompt used by Z-Image.
   - **Model thoughts** — `<think>` content (if any), purely for your curiosity.

Then hit **“Generate (streaming)”** with the refined prompt.

</details>

---

<details>
<summary><strong>Explore → Pick → Continue</strong></summary>

The **“Explore candidates & continue”** accordion lets you:

1. Rapidly explore a grid of **low-step candidates**.
2. Select indices you like.
3. Regenerate only those seeds with **full steps** for higher quality.

### 1. Explore candidates

Settings:

- `Number of candidates` — e.g. 18.
- `Steps for exploration candidates` — e.g. 3.
- (Optional) set a fixed `Base seed` in the main controls (e.g. 123) to make exploration reproducible.

Click **“Explore candidates”**:

- The **“Exploration candidates (low steps)”** gallery fills with `N` quick images.
- Internally, the script stores the **seeds** used for each candidate in `gr.State`.

### 2. Continue selected candidates

- Decide which candidate indices you like (zero-based):
  - First image → index `0`.
  - Second → `1`.
  - etc.
- Enter them in **“Indices to continue (comma-separated, zero-based)”**, e.g.:

  ```text
  0, 3, 5
  ```

- Set full `Steps` in the main controls (e.g. 9).
- Click **“Generate high-quality from selected”**.

The **“Continued images (full steps)”** gallery will show the higher-quality versions of those seeds, regenerated with the full step count.

</details>

---

## Model-Specific Notes (Z-Image Turbo)

- **Steps**  
  The authors recommend **~9 steps**, which correspond internally to **8 DiT transformer forwards**.
- **Guidance scale**  
  Set **`guidance_scale = 0.0`**:
  - Turbo has the conditioning baked in.
  - Classifier-free guidance and negative prompts are effectively ignored.
- **Prompting style**  
  Z-Image-Turbo favors:
  - Long, **concrete, visual** descriptions (subject, environment, lighting, composition, materials, color palette).
  - Explicit handling of text:
    - Put any text to be rendered in **double quotes**, e.g. `"Text"`.
    - Specify where it appears (sign, screen, label), and how it looks (size, color, style).

---

## Saving Images

- **CLI (`z_image_full_mps.py`)**
  - Saves images automatically into `zimage_full_outputs/` with timestamped filenames.

- **GUI (`z_image_full_gui_stream.py`)**
  - Does **not** save images to disk by default.
  - Options:
    - Right-click → “Save image as…” in the browser.
    - Or add `image.save(...)` calls in the Python code where final images are generated.

---

<details>
<summary><strong>Troubleshooting</strong></summary>

### CUDA warning on Mac

You might see:

```text
User provided device_type of 'cuda', but CUDA is not available. Disabling
```

This is harmless. On Apple Silicon the scripts prefer **`mps`**, then fall back gracefully to CPU if needed.

### Model download / auth issues

- Make sure your network is working.
- Accept the model license on Hugging Face if required.
- Log in with:

```bash
huggingface-cli login
```

if the model is gated.

### LM Studio errors

If the refiner shows:

```text
ERROR contacting LM Studio: ...
```

check:

- The LM Studio server is running on `http://localhost:1234`.
- The **model id** in the GUI matches the server’s active model name.
- There is no firewall or port conflict blocking `localhost:1234`.

</details>

---

## License

- The **Z-Image-Turbo model** itself is governed by the license on its Hugging Face page:  
  <https://huggingface.co/Tongyi-MAI/Z-Image-Turbo>  
  Please read and follow those terms.
