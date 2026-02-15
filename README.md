<p align="center">
  <h1 align="center">🤟 Sanchar Mitra</h1>
  <p align="center"><strong>Real-time ASL Sign Language Recognition & Voice Communication System</strong></p>
  <p align="center">
    <em>Bridging the communication gap for the deaf and mute community</em>
  </p>
  <p align="center">
    <a href="#features">Features</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#hardware">Hardware</a> •
    <a href="#setup">Setup</a> •
    <a href="#usage">Usage</a> •
    <a href="#team">Team</a>
  </p>
</p>

---

## About

**Sanchar Mitra** ("Communication Friend" in Hindi) is an end-to-end communication system that enables deaf/mute individuals to communicate naturally using American Sign Language (ASL). The system combines on-device neural network inference on an STM32 microcontroller with a laptop-side ML pipeline to deliver real-time sign recognition, text-to-speech output, speech-to-text input, and LLM-powered conversational responses.

> Built for the **ST Innovation Fair 2026**

---

## Features

| Feature | Description |
|---------|-------------|
| **Real-time ASL Recognition** | On-device palm detection & hand landmark extraction (21 landmarks) on STM32N6570-DK, classified by a Random Forest model on the laptop |
| **Text-to-Speech (TTS)** | Recognized words are spoken aloud using Microsoft Heera (English India) voice — no internet required |
| **Speech-to-Text (STT)** | Whisper-based offline speech recognition — spoken words are sent to the board and displayed on LCD |
| **LLM Integration** | Accumulated signed text can be sent to Ollama (Llama 3.2) for AI-powered conversational responses, which are then spoken via TTS |
| **Full-Duplex UART** | Simultaneous bidirectional communication between board and laptop over a single UART connection |
| **On-Board Display** | Live camera feed with landmark overlay, recognized signs, and voice text displayed on the STM32 LCD |

---

## Architecture

```
┌──────────────────────────┐        UART TX (21 landmarks)        ┌─────────────────────────┐
│    STM32N6570-DK Board   │ ──────────────────────────────────▶  │    Laptop (Python)      │
│                          │                                      │                         │
│  • Camera capture        │        UART RX (letter/text)         │  • uart.py (ASL mode)   │
│  • Palm detection (NN)   │ ◀──────────────────────────────────  │  • stt.py  (Voice mode) │
│  • Hand landmarks (NN)   │                                      │  • Random Forest model  │
│  • LCD display           │                                      │  • Ollama LLM           │
│  • ATON NN accelerator   │                                      │  • TTS (Heera voice)    │
└──────────────────────────┘                                      └─────────────────────────┘
```

### Data Flow — ASL Recognition Mode

1. **Board** captures camera frames and runs palm detection + hand landmark extraction using on-device neural networks (ATON runtime)
2. **Board** streams 21 hand landmarks (x, y, z) over UART at ~1 frame/sec
3. **Laptop** (`uart.py`) receives landmarks, extracts x,y features, normalizes, and classifies using a Random Forest
4. **Laptop** sends the predicted letter back to the board for LCD display
5. When a word is completed (hand removed), the word is spoken aloud via **TTS**
6. When the user confirms with a "SEND" gesture, the accumulated text is sent to **Ollama** for an AI response, which is also spoken via TTS

### Data Flow — Speech-to-Text Mode

1. **Laptop** (`stt.py`) listens to the microphone using Whisper (offline)
2. Recognized text is sent to the board via UART with `>text\n` framing
3. **Board** displays the text on LCD under a "Voice:" label

---

## Hardware

| Component | Details |
|-----------|---------|
| **Board** | STM32N6570-DK (Discovery Kit) |
| **MCU** | STM32N6xx series (Cortex-M based) |
| **Camera** | On-board camera via DCMIPP |
| **Display** | On-board LCD with dual-layer rendering |
| **UART** | 115200 baud, 8N1, USB-serial (COM3) |
| **NN Accelerator** | ATON runtime for palm detection & hand landmark models |

---

## Project Structure

```
sanchar-mitra/
├── app.c                  # Board-side: camera, NN inference, display, UART (FreeRTOS)
├── uart.py                # Laptop: real-time ASL recognition + Ollama + TTS
├── stt.py                 # Laptop: speech-to-text → UART → board display
├── train_model.py         # Train Random Forest classifier
├── collect_data.py        # Collect custom training data via UART
├── collect_all.py         # Batch data collection for all letters
├── download_dataset.py    # Download Kaggle ASL landmark dataset
├── speak_heera.ps1        # TTS helper: Microsoft Heera voice via COM SAPI
├── check_voices.py        # Diagnostic: list installed SAPI5/OneCore voices
├── fix_voices.ps1         # Register OneCore voices for SAPI5 (run as admin)
├── PROJECT_CONTEXT.md     # Detailed technical documentation
└── README.md              # This file
```

---

## Setup

### Prerequisites

- **Python 3.10+**
- **STM32N6570-DK** board with firmware flashed via STM32CubeIDE
- **Ollama** installed with `llama3.2:3b` model pulled (`ollama pull llama3.2:3b`)
- **Windows** with English (India) voice pack installed (Settings → Time & Language → Speech)

### Installation

```bash
# Clone the repository
git clone https://github.com/maitreya0106/sanchar-mitra.git
cd sanchar-mitra

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows

# Install dependencies
pip install numpy scikit-learn pyserial requests

# For Speech-to-Text mode (optional)
pip install faster-whisper sounddevice
```

### Train the Model

```bash
# Download Kaggle ASL landmark dataset
python download_dataset.py

# (Optional) Collect custom training data from the board
python collect_data.py

# Train the Random Forest classifier
python train_model.py
```

---

## Usage

### ASL Recognition Mode

```bash
# Connect STM32N6570-DK via USB-serial
# Make sure Ollama is running: ollama serve

python uart.py
```

**Terminal output:**
```
Model loaded from asl_model.pkl
Connected to COM3. Recognizing ASL signs...

  >> A (85%)  (1/2)
  >> A (87%)  *** Confirmed: A  |  Buffer: A
  >> B (72%)  (1/2)
  >> B (79%)  *** Confirmed: B  |  Buffer: AB
  [no hand] Space added  |  Buffer: AB
  [TTS] Speaking: AB
```

### Speech-to-Text Mode

```bash
# Stop uart.py first (they share the same COM port)
python stt.py
```

**Terminal output:**
```
Loading Whisper model (base)...
Model loaded.
Connected to COM3 at 115200 baud.

  ... hello world
  >> Sent: "hello world"
```

---

## Technical Details

### ML Pipeline

- **Features:** 42 per sample (21 landmarks × x,y — z is dropped as noisy from 2D camera)
- **Normalization:** Wrist-centered, scaled by wrist-to-middle-finger-MCP distance (position & scale invariant)
- **Classifier:** Random Forest with 200 trees
- **Confirmation:** 2 consecutive identical predictions above 20% confidence
- **Skipped letters:** J and Z (require motion, cannot classify from single frame)

### UART Protocol

| Direction | Format | Example |
|-----------|--------|---------|
| Board → Laptop | `LM:x0,y0,z0, x1,y1,z1, ... x20,y20,z20,\r\n` | Landmark data |
| Laptop → Board (sign) | Single byte `A`–`Z` | `A` |
| Laptop → Board (STT) | `>text\n` | `>hello world\n` |

### TTS Engine

- Uses **Microsoft Heera** (English India) OneCore voice via COM SAPI
- Accesses OneCore registry directly (`HKLM:\SOFTWARE\Microsoft\Speech_OneCore\Voices`)
- Fully offline — no internet required after voice pack installation

### LLM Integration

- **Ollama** with `llama3.2:3b` running locally
- Streaming enabled for real-time token output
- System prompt enforces concise 1-2 sentence replies
- Response is spoken aloud via TTS pipeline

---

## Board Firmware (app.c)

The STM32N6570-DK runs **5 FreeRTOS tasks**:

| Task | Purpose |
|------|---------|
| `nn_thread_fct` | Palm detection → hand landmark inference loop |
| `dp_thread_fct` | LCD display rendering with landmark overlay |
| `isp_thread_fct` | Image Signal Processor updates |
| `uart_landmark_task` | TX: streams landmarks to laptop |
| `uart_receive_task` | RX: receives sign letters or STT text |

The board uses the **ATON neural network accelerator** for on-device inference of palm detection and hand landmark models, achieving real-time performance without cloud dependencies.

---

## Dependencies

### Python (Laptop)

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations |
| `scikit-learn` | Random Forest classifier |
| `pyserial` | UART communication |
| `requests` | Ollama API calls |
| `faster-whisper` | Offline Whisper STT (for `stt.py`) |
| `sounddevice` | Microphone capture (for `stt.py`) |

### Board (C)

- STM32 HAL, FreeRTOS, ATON NN Runtime, LCD utilities, Camera middleware

---

## Team

| | Name | Role |
|-|------|------|
| 👨‍💻 | **Ankur Majumdar** | JIIT Noida |
| 👨‍💻 | **Maitreya Agarwal** | JIIT Noida |

Built with ❤️ for the **ST Innovation Fair 2026**

---

## License

This project is open source and available under the [MIT License](LICENSE).
