# Z-Image Turbo Tools (Mac M-series + MPS)

Local utilities around **[Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)** tuned for Apple Silicon.

## Quick Start (install)

```bash
python3 -m venv zimage-env
source zimage-env/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision torchaudio transformers accelerate safetensors pillow gradio requests
pip install git+https://github.com/huggingface/diffusers
```

Model access: Z-Image-Turbo is public (no gating needed). 

## Quick run: streaming GUI (recommended)

Streaming Gradio UI with live denoising previews and optional LM Studio prompt refiner.

```bash
source zimage-env/bin/activate
python z_image_full_gui_stream.py
# open the printed http://127.0.0.1:7860 link
```

CLI fallback (same model, BF16 on MPS): `python z_image_full_mps.py --device mps --steps 9 --prompt "..."`

## Train a LoRA (workflow)

1) **Caption your images (copies + captions in one go)**

```bash
source zimage-env/bin/activate
python caption_with_VLM_for_LORA.py \
  /path/to/raw_photos \
  /path/to/raw_photos_captioned \
  --model qwen3-vl-32b-instruct \
  --token "<your_trigger_token>" \
  --class-word "<class_word>"
```

- Reads images (and optional sibling .txt captions) from the input dir.
- Copies them into the output dir and writes per-image captions that already include your trigger token and class word.
- **Token** = the unique string you’ll later put into prompts to invoke the LoRA (e.g., `mycharv1`).
- **Class word** = broad category for the subject (e.g., `person`, `woman`, `man`, `statue`); empty string is allowed if you want token-only.

2) **Set up AI Toolkit for LoRA training**

General repo: <https://github.com/ostris/ai-toolkit> (MPS-specific notes: <https://github.com/ivanfioravanti/ai-toolkit/tree/main>).

```bash
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
python3 -m venv venv
source venv/bin/activate
# install PyTorch with MPS support
pip3 install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0
pip3 install -r requirements.txt
```

Drop your captioned dataset folder (images + .txt files) into the AI-Toolkit data layout and launch LoRA training per their docs (use the MPS guide for device settings / flags).

## Run LoRA inference (`z_image_lora.py`)

Batch or interactive REPL; loads base Z-Image plus your LoRA.

```bash
source zimage-env/bin/activate
python z_image_lora.py --lora-path /path/to/your_lora.safetensors
```

Inside the REPL you can change on the fly:
- `/seed <int|random>`
- `/steps <int>` (Turbo: ~6–9)
- `/n <int>` images per prompt
- `/size WxH`
- `/cfg <float>` (Turbo likes 0.0)

Prompts you enter will generate with the current settings; fixed seeds auto-increment after each batch. The same REPL controls apply to `z_image_full_mps.py` when you run it without a LoRA.

## Download LoRA samples & make GIFs

Takes `.jpg` frames named like `...__000000050_0.jpg` (step + sample id) and builds per-sample GIFs with step labels.

```bash
source zimage-env/bin/activate
python make_lora_gifs.py /path/to/lora_samples -o gifs --duration 500 --corner top_left
```

## File map

- `z_image_full_gui_stream.py` — streaming GUI with LM Studio prompt refiner and explore → pick → continue flow.
- `caption_with_VLM_for_LORA.py` — caption helper for LoRA datasets.
- `z_image_lora.py` — run Z-Image with a trained LoRA.
- `make_lora_gifs.py` — turn downloaded LoRA samples into GIFs.
- `z_image_full_mps.py` — CLI/REPL (BF16 on MPS; falls back to CPU/CUDA).
