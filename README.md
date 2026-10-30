# BatVision AI

> “The white cane tells you what’s underfoot. BatVision tells you what’s coming.”

BatVision AI is a wearable assistive-navigation system that mixes fast on-device reflexes with cloud-based reasoning. It watches the world with a camera, estimates depth, spots hazards, warns you in real time, and can answer natural-language questions like:

- “Where is the red water bottle?”
- “What does the sign say?”
- “Is the path clear?”
- “Describe what is around me.”

The system is designed to be safe-first: the on-device perception stack does the life-or-death work, while the cloud model is there to help with richer understanding and natural-language interaction.

---

## Why this project exists

A cane catches low obstacles. It does not reliably catch:

- head-height obstacles like shelves, branches, or truck mirrors
- fast silent threats like scooters and bikes
- spatial context like “the door handle is on the right”

BatVision splits the work into two brains:

| Layer | Runs where | What it does | Why it matters |
| --- | --- | --- | --- |
| Reflex layer | On your device | depth, corridor analysis, looming detection, object names, beeps | works offline and fast |
| Reasoning layer | In the app, optionally with Nebius | natural language, scene understanding, object lookup | answers the hard questions |

The key safety principle: the cloud model is never trusted to decide whether the path is safe. The local device verifies every answer before it is spoken.

---

## What BatVision can do

- Detect obstacles in a 3D corridor using depth from a single RGB camera
- Warn about approaching hazards with fast on-device checks
- Use stereo-like spatial beeps and spoken alerts
- Track objects and answer visual questions in natural language
- Support offline operation with a mock backend for development
- Use NVIDIA Nemotron on Nebius Token Factory when an API key is available

---

## Project layout

```text
batvision/
├── config/
│   └── edge_config.yaml
├── eval/
│   ├── eval_agent_tools.py
│   ├── eval_detector_coco.py
│   └── eval_vizwiz_vlm.py
├── models/
│   └── downloaded ONNX model files
├── scripts/
│   ├── download_models.py
│   ├── prepare_coco_subset.py
│   ├── prepare_vizwiz.py
│   ├── show_agent_tools.py
│   └── smoke_test_model.py
├── src/
│   └── batvision/
├── tests/
├── training/
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements-dev.txt
├── requirements-edge.txt
├── requirements-lock.txt
└── README.md
```

---

## Quick start

### 1) Clone the repo

```bash
git clone https://github.com/yadavmanan/batvision.git
cd batvision
```

### 2) Create a virtual environment

#### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```bash
pip install -U pip
pip install -r requirements-dev.txt
pip install -e .
```

This installs the project in editable mode and brings in the local development dependencies.

### 4) Download the model weights

```bash
python scripts/download_models.py
```

This fetches the ONNX model assets needed for depth estimation and object detection. If you are only testing offline behavior, you can also skip the live model downloads and use the mock backend.

### 5) Run the offline tests

```bash
pytest
```

If you want the full test suite without the slow live/latency benchmarks:

```bash
pytest -m "not latency and not live"
```

---

## Running the app

### Reflex-only mode (safe and offline)

```bash
python -m batvision.edge.app --set cloud.enabled=false
```

This mode keeps the edge safety system active and disables cloud model calls. Great for testing camera setup, calibration, and obstacle warnings without any external key.

### Live cloud mode

Create a local `.env` file in the project root:

```env
NEBIUS_API_KEY=your_key_here
VLM_BACKEND=nebius
```

Then run:

```bash
python -m batvision.edge.app
```

Or explicitly:

```powershell
$env:VLM_BACKEND = "nebius"
python -m batvision.edge.app --source 0 --set cloud.enabled=true --set voice.backend=keyboard
```

### Mock mode

Use the built-in mock provider for development without a network or API key:

```bash
VLM_BACKEND=mock python -m batvision.edge.app
```

PowerShell:

```powershell
$env:VLM_BACKEND = "mock"
python -m batvision.edge.app
```

---

## Configuration and tuning

The main runtime config is in `config/edge_config.yaml`.

A few useful examples:

```bash
python -m batvision.edge.app --source 0 --set cloud.enabled=false
python -m batvision.edge.app --source "C:\demo\walking.mp4" --set cloud.enabled=false
python -m batvision.edge.app --set depth.input_size=192 --set detector.hz=1
python -m batvision.edge.app --headless --frames 300
```

Important settings:

- `camera.source`: webcam index, video path, RTSP URL, or phone camera stream
- `mount.height_m`: camera height above the floor
- `mount.pitch_deg`: camera tilt relative to level
- `cloud.enabled`: turn model calls on or off
- `voice.backend`: `keyboard`, `vosk`, or `none`
- `debug.show_windows`: show the GUI overlay for debugging

---

## Camera setup and calibration

For best results, mount the camera where it matches real use. A chest-mounted webcam or phone camera works well. The app uses the calibration settings to estimate floor and obstacle geometry.

Typical values:

```yaml
mount:
  height_m: 1.30
  pitch_deg: 12.0
  user_height_m: 1.75
```

If the camera is too high, too low, or tilted incorrectly, depth warnings will drift. Tune those values until the floor plane and obstacle warnings feel right in real use.

---

## Model backend options

BatVision supports a local-first structure and a cloud-based reasoning backend.

### Local mode

The app can run completely offline with the edge stack and a mock responder.

### Nebius / NVIDIA Nemotron mode

Set the following in your `.env` file:

```env
NEBIUS_API_KEY=your_token_factory_key
NEBIUS_MODEL_ID=nvidia/Nemotron-3.5-Super-VL
VLM_BACKEND=nebius
```

The app expects a standard OpenAI-compatible endpoint from Nebius Token Factory.

---

## Useful scripts

### Smoke test the cloud model

```bash
python scripts/smoke_test_model.py
```

This checks that the model backend is reachable and that the safety verifier still functions when a remote answer is returned.

### Show agent tools

```bash
python scripts/show_agent_tools.py
```

This is useful for inspecting the image-analysis tools that the reasoning agent can call.

### Eval scripts

```bash
python eval/eval_agent_tools.py --backend nebius
python eval/eval_vizwiz_vlm.py --limit 100
python eval/eval_detector_coco.py
```

---

## Troubleshooting

### The app starts but says the cloud is unavailable

Check that the `.env` file exists and contains a valid `NEBIUS_API_KEY`.

```env
NEBIUS_API_KEY=your_key_here
```

### Camera opens but the app is noisy or unstable

- lower the resolution
- increase your CPU headroom
- test with a static video file
- check the webcam index and permissions

### The app claims the path is clear when it should not

This is exactly why the safety stack exists: the verifier checks the raw depth data and refuses unsafe claims. If the warning logic feels off, recalibrate the mount and revisit the floor-plane settings.

### The cloud answer is wrong or too vague

- use higher-quality framing
- keep the subject centered
- try the mock model offline first
- lower latency expectations if the remote model is slow

---

## Safety note

This project is an assistive prototype, not a medical device and not a replacement for a cane, guide dog, or trained assistance. It is intended to augment situational awareness and should be used with appropriate caution.

The on-device reflex layer is the safety-critical component; the cloud reasoning layer is an assistive layer.

---

## License

This project is distributed under the MIT license. See the repository license file for details.

---

## Quick commands summary

```bash
# 1. setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
python scripts/download_models.py

# 2. run offline
python -m batvision.edge.app --set cloud.enabled=false

# 3. run with API key
printf "NEBIUS_API_KEY=your_key_here\nVLM_BACKEND=nebius\n" > .env
python -m batvision.edge.app

# 4. run mock mode
VLM_BACKEND=mock python -m batvision.edge.app
```

If you want the shortest possible version of the story: BatVision is a wearable safety copilot that keeps the dangerous decisions local and lets the cloud do the thinking when it helps.

This is the kind of system that makes you feel: “Okay, the camera sees the world, the model interprets it, and the device still refuses to be dumb.”

---

## Final thought

The move here is simple: let the machine be fast on the edge, smart in the cloud, and strict about safety. That is the whole idea.

If that sounds good, fire up the app, point the camera at the world, and let BatVision do the heavy lifting while you stay in motion.
