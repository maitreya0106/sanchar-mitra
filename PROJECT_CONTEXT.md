# Sanchar Mitra — Full Project Context

> **Last updated:** February 10, 2026
> **Team:** Ankur Majumdar & Maitreya Agarwal (JIIT)
> **Event:** ST Innovation Fair 2026

---

## 1. Project Overview

**Sanchar Mitra** is a real-time ASL (American Sign Language) recognition system. An **STM32N6570-DK** board runs hand detection + landmark extraction using on-device neural networks, streams 21 hand landmarks over UART to a laptop, where a **Random Forest classifier** identifies the signed letter. The recognized letter is sent back to the board over the same UART (full-duplex) and displayed on the board's LCD.

### Data Flow

```
┌─────────────────────┐       UART TX (landmarks)       ┌──────────────────────┐
│   STM32N6570-DK     │ ──────────────────────────────▶  │   Laptop (Python)    │
│                     │                                  │                      │
│  - Camera capture   │       UART RX (letter)           │  - uart.py           │
│  - Palm detection   │ ◀──────────────────────────────  │  - asl_model.pkl     │
│  - Hand landmarks   │                                  │  - scikit-learn RF   │
│  - LCD display      │                                  │                      │
└─────────────────────┘                                  └──────────────────────┘
```

---

## 2. Hardware

| Component | Details |
|-----------|---------|
| Board | STM32N6570-DK (Discovery Kit) |
| MCU | STM32N6xx series (Cortex-M based) |
| Camera | On-board camera via DCMIPP |
| Display | On-board LCD (foreground ARGB4444 + background RGB888/ARGB8888) |
| UART | `huart1` — 115200 baud, 8N1, connected to laptop via USB-serial (COM3) |
| NN Accelerator | ATON runtime (palm_detector + hand_landmark models) |

---

## 3. File Structure & Purpose

### Board-side (C, FreeRTOS)

| File | Purpose |
|------|---------|
| `app.c` | Main application — camera, NN inference, display, UART tasks |

### Laptop-side (Python)

| File | Purpose |
|------|---------|
| `uart.py` | Real-time recognition: reads landmarks from UART, classifies, sends letter back to board |
| `train_model.py` | Trains Random Forest on Kaggle + custom data, saves `asl_model.pkl` |
| `collect_data.py` | Collects custom training data from board via UART |
| `collect_all.py` | Batch collection for all letters |
| `download_dataset.py` | Downloads Kaggle ASL landmark dataset |
| `stt.py` | Speech-to-Text mode: listens to laptop mic (Whisper offline via faster-whisper), sends transcribed text to board via UART |
| `speak_heera.ps1` | PowerShell helper: speaks a word using Microsoft Heera (English India) OneCore voice via COM SAPI |
| `check_voices.py` | Diagnostic: lists all SAPI5 Desktop and OneCore voices installed on the system |

### Data Files

| File | Purpose |
|------|---------|
| `asl_model.pkl` | Trained Random Forest model (200 trees) |
| `asl_dataset.csv` | Custom-collected data (42 features + label per row) |
| `X_train_xy.npy` | Consolidated Kaggle features (x,y only, 42 per sample) |
| `y_train.npy` | Consolidated Kaggle labels |
| `kaggle_dataset/landmarks/A-Z/` | Raw Kaggle .npy files (63 values: 21 landmarks × x,y,z) |

---

## 4. app.c — Board Code Architecture

### Constants & Config

- `LCD_BG_WIDTH/HEIGHT` — LCD resolution (from `app_config.h`)
- `PD_MAX_HAND_NB = 1` — single hand tracking
- `DISPLAY_BUFFER_NB = DISPLAY_DELAY + 2` — triple-buffered display
- `LD_LANDMARK_NB = 21` — hand landmarks (defined in `ld.h`)
- Font: `Font20` on DK, `Font12` on Nucleo

### Data Structures

- `roi_t` — Region of Interest (cx, cy, w, h, rotation)
- `hand_info_t` — Per-hand data: validity, palm detection box, ROI, 21 landmarks
- `display_info_t` — Full frame info: timing, hand data, flags
- `display_t` — Thread-safe display state with FreeRTOS mutex + semaphore
- `bqueue_t` — Double-buffered queue for NN input frames
- `pd_model_info_t` / `hl_model_info_t` — Neural network I/O buffers

### FreeRTOS Tasks (5 total)

| Task | Priority | Function | Purpose |
|------|----------|----------|---------|
| `nn` | IDLE + half + 1 | `nn_thread_fct` | Palm detection → hand landmark inference loop |
| `dp` | IDLE + half - 2 | `dp_thread_fct` | Display rendering (waits on `disp.update` semaphore) |
| `isp` | IDLE + half + 2 | `isp_thread_fct` | ISP (Image Signal Processor) updates |
| `uart_lm` | IDLE + 1 | `uart_landmark_task` | TX: sends landmarks to laptop every 1000ms |
| `uart_rx` | IDLE + 1 | `uart_receive_task` | RX: receives sign letters (A-Z) or STT text (>msg\n) from laptop |

### UART Protocol

**Board → Laptop (TX, every ~1s):**
```
(x,y,z) LM:x0,y0,z0,   x1,y1,z1,   ... x20,y20,z20,\r\n
```
- 21 landmarks, each with integer x, y, z (screen-projected coordinates)
- Format: `(x,y,z) LM:` prefix, comma-separated, `\r\n` terminated
- Sent via `HAL_UART_Transmit(&huart1, ...)` with `HAL_MAX_DELAY`
- Only sent when a hand is detected (`pd_hand_nb > 0 && hands[0].is_valid`)

**Laptop → Board (RX) — two message types:**

| Type | Format | Source | Example |
|------|--------|--------|---------|
| Sign prediction | bare `A`–`Z` byte | `uart.py` | `A` |
| STT sentence | `>text\n` (0x3E prefix + newline) | `stt.py` | `>hello world\n` |

- Received via `HAL_UART_Receive(&huart1, &rx_byte, 1, 100)` (100ms timeout polling)
- `uart_receive_task` uses a state machine: `RX_NORMAL` vs `RX_STT_ACCUM`
  - In `RX_NORMAL`: if byte is `>` → switch to `RX_STT_ACCUM`; if `A`–`Z` → sign mode
  - In `RX_STT_ACCUM`: accumulate bytes into `stt_text[]` until `\n` → set `stt_active = 1`
  - Safety: aborts STT accumulation after 5s of silence

### Display Globals

- `static volatile char rx_char` — latest sign character (A–Z or `\0`)
- `static volatile char stt_text[128]` — accumulated STT sentence
- `static volatile int stt_text_len` — length of current STT text
- `static volatile int stt_active` — 1 = showing STT text, 0 = sign mode
- **Sign mode:** `"Sign: X"` in Font24, yellow on white label
- **STT mode:** `"Voice:"` label in cyan, text in white using `LCD_FONT` with word-wrap (up to 3 lines)
- Switching: receiving `A`–`Z` clears STT display; receiving `>text\n` clears sign display

### NN Pipeline (nn_thread_fct)

1. Get frame from `nn_input_queue` (double-buffered camera pipe)
2. If **not tracking**: run palm detector → get bounding box + ROI
3. If **tracking**: reuse ROI from landmark-based prediction
4. Run hand landmark model on cropped/rotated ROI
5. If valid: compute next ROI from landmarks (tracking loop)
6. Update `disp.info` under mutex for display thread

### Display (dp_thread_fct → Display_NetworkOutput)

- Clears foreground layer
- Draws branding: "SANCHAAR MITRA", team info
- Shows CPU load, inference times, FPS (top-right)
- Draws landmarks (yellow dots + black skeleton) and/or palm detection box
- Shows received data at bottom-left: `"Sign: X"` (sign mode) or `"Voice: text"` (STT mode, word-wrapped)
- Credits at bottom-right

### Rotation Support

- `HAS_ROTATION_SUPPORT == 1`: Uses NeoChrom GPU (GFXMMU + nema) for affine transform
- `HAS_ROTATION_SUPPORT == 0`: Simple bilinear resize (no rotation correction)

---

## 5. uart.py — Laptop Recognition Logic

### Config

```python
SERIAL_PORT = 'COM3'
BAUD_RATE   = 115200
MODEL_FILE  = 'asl_model.pkl'
CONFIRM_FRAMES = 2      # consecutive same predictions to confirm
MIN_CONFIDENCE = 0.20   # minimum prediction probability
```

### Landmark Parsing

1. Receive line via `ser.readline()` (1s timeout)
2. Look for `LM:` prefix, split by commas → expect 63 values
3. Extract x,y only (drop z) → 42 values
4. Normalize: translate wrist to origin, scale by wrist-to-MCP9 distance

### Recognition Loop

```
For each UART frame:
  ├─ Parse 63 coords → 42 (x,y only)
  ├─ Normalize (wrist-centered, scale-invariant)
  ├─ Random Forest predict + predict_proba
  ├─ If confidence > 20%:
  │   ├─ Send prediction letter to board via ser.write()  ← FULL DUPLEX
  │   ├─ Count consecutive same predictions
  │   └─ If count == 2 → CONFIRMED → add to sentence buffer
  └─ If confidence ≤ 20% → reset counters, skip

When readline() times out (no UART data for 1s):
  └─ Add space to sentence buffer (word separator)
     (Nothing sent to board — space is local only)
```

### Key Behaviors

- **Confirmation:** 2 consecutive identical predictions at >20% confidence
- **Space insertion:** When UART stops sending (hand removed), `readline()` times out → space added to local buffer once. Only happens if buffer isn't empty and last char isn't already a space
- **Full-duplex:** `ser.write()` sends every above-threshold prediction (unconfirmed) to the board, while `ser.readline()` reads landmarks — pyserial handles both directions on the same port
- **No space sent to board:** When hand is absent, nothing is written to serial
- **TTS (Text-to-Speech):** When a space is added to the buffer (hand removed), the completed word is sent to a background thread via a `queue.Queue`. The TTS engine speaks the word using the **Microsoft Heera** (English India) OneCore voice. Implementation details:
  - `speak_heera.ps1` is called via `subprocess` from `uart.py`
  - Uses COM `SAPI.SpVoice` object with `SAPI.SpObjectTokenCategory` pointed at the OneCore registry path (`HKLM:\SOFTWARE\Microsoft\Speech_OneCore\Voices`) — this bypasses the SAPI5 Desktop limitation where only David and Zira are visible to `System.Speech.Synthesis.SpeechSynthesizer`
  - Speech rate: **-2** (SAPI range is -10 to 10, 0 = default)
  - Runs with `-ExecutionPolicy Bypass` and `CREATE_NO_WINDOW` flag (no console flash)
  - Words are spoken one at a time in FIFO order. No internet required.
- **Sign cleared on no hand:** `uart_receive_task` uses a timeout counter — only clears `rx_char` to `'\0'` after **15 consecutive** 100ms timeouts (1.5s of silence). This prevents flickering during active tracking where there are natural gaps between landmark frames. The character stays solid on-screen while the hand is present, and only disappears ~1.5s after the hand is truly removed.
- **Sign display size:** The recognized letter is drawn at bottom-left using Font24 rendered as a 2×2 bold block (4 offset prints at two Y positions) in yellow for high visibility, with the "Sign:" label in white.

---

## 5b. stt.py — Speech-to-Text Mode

Run **instead of** `uart.py` (they share the same COM port).

### How It Works

1. Opens laptop microphone via `sounddevice` (16 kHz, mono, 16-bit PCM)
2. Accumulates audio chunks, using RMS energy to detect speech vs silence
3. Shows `[listening...]` feedback in terminal while speech is detected
4. On silence after speech (1.5s), transcribes accumulated audio with **Whisper** (`faster-whisper`, `base` model, CPU int8)
5. Filters out Whisper hallucinations (common false positives on silence)
6. Sends recognized text as `>text\n` to board via UART
7. Board displays the text on LCD ("Voice:" label + word-wrapped text)
8. A drain thread discards incoming landmark data (board still sends them)

### Config

```python
SERIAL_PORT = 'COM3'
BAUD_RATE   = 115200
SAMPLE_RATE = 16000          # Whisper needs 16 kHz
BLOCK_SIZE  = 1600           # 100ms per callback block
MODEL_SIZE  = 'base'         # Whisper model size (tiny/base/small/medium/large-v3)
SILENCE_THRESHOLD = 500      # RMS energy threshold for silence detection
SILENCE_DURATION  = 1.5      # seconds of silence to trigger transcription
MIN_SPEECH_DURATION = 0.5    # minimum speech length to transcribe
```

### Usage

```bash
python stt.py              # default COM3
python stt.py --port COM5  # different port
```

### Mode Switching

- Run `stt.py` → board shows voice text
- Stop `stt.py`, run `uart.py` → first sign prediction clears voice text, board returns to sign mode
- No explicit mode command needed — the board auto-switches based on data type received

---

## 6. train_model.py — Model Training

### Data Sources

1. **Kaggle dataset** (`kaggle_dataset/landmarks/A-Z/frame_*.npy`)
   - Each .npy is 63 floats (21 landmarks × 3) or shape (21,3)
   - Fast path via consolidated `X_train_xy.npy` + `y_train.npy`
2. **Custom data** (`asl_dataset.csv`)
   - 42 float features + label per row (already x,y only)

### Pipeline

1. Load Kaggle + custom data
2. Extract x,y → normalize (wrist origin, MCP9 scale)
3. Skip J and Z (motion-based, can't do single-frame)
4. Train/test split: 80/20, stratified
5. Random Forest: 200 trees, all cores
6. Save to `asl_model.pkl`

### Features (42 per sample)

```
[wrist_x, wrist_y, thumb_cmc_x, thumb_cmc_y, ..., pinky_tip_x, pinky_tip_y]
```
All normalized: wrist at origin, scaled by distance to landmark 9 (middle finger MCP).

---

## 7. collect_data.py — Custom Data Collection

- Connects to board via UART
- Prompts for letter and sample count
- For each sample: reads landmarks, normalizes, appends to `asl_dataset.csv`
- Can run interactively or with CLI args: `python collect_data.py A 50`

---

## 8. Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Features | x,y only (drop z) | z is noisy from 2D camera, x,y sufficient for static signs |
| Normalization | Wrist origin + MCP9 scale | Position/scale invariant |
| Classifier | Random Forest (200 trees) | Fast inference, good with tabular data, works on laptop |
| Confirmation | 2 consecutive frames | Balance between responsiveness and accuracy |
| UART direction | Full-duplex on single UART | TX landmarks, RX letters — no extra hardware needed |
| What's sent back | Unconfirmed predictions | Board shows live prediction; confirmation logic is laptop-side |
| Space character | Local only (not sent to board) | Board doesn't need spaces; just shows current sign |
| Board UART RX method | Polling in FreeRTOS task | Simple, no extra IRQ handler, fine at 115200 baud |
| Skipped letters | J, Z | Require motion → can't classify from single frame |
| TTS voice | Microsoft Heera (English India) | Indian-accented English, natural for Indian audience |
| TTS method | COM SAPI.SpVoice + OneCore registry | `System.Speech.Synthesis` can't see OneCore voices; COM approach accesses them directly |
| TTS speed | Rate = -2 | Slower than default for clarity |
| STT engine | Whisper (offline, base model via faster-whisper) | No internet needed (after first download), much higher accuracy than Vosk, CPU int8 inference |
| STT protocol | `>text\n` prefix | Distinguishes from bare A–Z sign bytes; minimal overhead (1 byte marker) |
| STT script | Separate `stt.py` | No port conflicts with `uart.py`; run one or the other |
| STT mode switching | Auto via data type | Board detects `>` vs `A`–`Z` — no explicit mode command needed |

---

## 9. Future Work / TODO

- [ ] Send confirmed characters to board for sentence display on LCD
- [ ] Add gesture-based backspace/clear
- [x] Text-to-speech on laptop side (Heera OneCore voice via COM SAPI, word-level, background thread)
- [x] Speech-to-text mode: Whisper offline STT → UART → board LCD display (`stt.py`)
- [ ] Improve model with more custom training data
- [ ] Add J and Z with temporal/sequence model
- [ ] Bluetooth/Wi-Fi instead of wired UART
- [ ] On-device classification (no laptop needed)

---

## 10. How to Run

### First Time Setup

```bash
# 1. Download Kaggle dataset
python download_dataset.py

# 2. (Optional) Collect custom data
python collect_data.py

# 3. Train model
python train_model.py
```

### Running Sign Language Recognition

```bash
# Flash app.c to STM32N6570-DK (via STM32CubeIDE or STM32CubeProgrammer)
# Connect board to laptop via USB-serial (COM3)

python uart.py
```

### Expected Terminal Output (uart.py)

```
Model loaded from asl_model.pkl
Connected to COM3. Recognizing ASL signs...

  >> A (85%)  (1/2)
  >> A (87%)  *** Confirmed: A  |  Buffer: A
  >> B (72%)  (1/2)
  >> B (79%)  *** Confirmed: B  |  Buffer: AB
  [no hand] Space added  |  Buffer: AB
  >> C (68%)  (1/2)
  >> C (71%)  *** Confirmed: C  |  Buffer: AB C
```

### Running Speech-to-Text Mode

```bash
# Stop uart.py first (they share the same COM port)

python stt.py
# First run auto-downloads Whisper base model
```

### Expected Terminal Output (stt.py)

```
Loading Whisper model (base)...
(First run will download the model automatically)

Model loaded.

Connected to COM3 at 115200 baud.

============================================================
  SPEECH-TO-TEXT MODE  (Whisper offline)
  Speak into your microphone.
  Recognized text is sent to the board via UART.
  Press Ctrl+C to exit.
============================================================

  ... hello world
  >> Sent: "hello world"
  ... how are you
  >> Sent: "how are you"
```

---

## 11. Dependencies

### Python (laptop)

- `pyserial` — UART communication
- `numpy` — array operations
- `scikit-learn` — Random Forest classifier
- `pickle` — model serialization
- `faster-whisper` — offline Whisper STT engine via CTranslate2 (used by `stt.py`)
- `sounddevice` — microphone audio capture (used by `stt.py`)

### PowerShell (laptop, TTS)

- `SAPI.SpVoice` COM object — speech synthesis
- `SAPI.SpObjectTokenCategory` COM object — access OneCore voices from `HKLM:\SOFTWARE\Microsoft\Speech_OneCore\Voices`
- Requires **English (India)** voice pack installed via Windows Settings → Time & Language → Speech

### C (board)

- STM32 HAL (`stm32n6xx_hal.h`)
- FreeRTOS (`FreeRTOS.h`, `task.h`, `semphr.h`)
- ATON NN Runtime (`ll_aton_runtime.h`)
- LCD utilities (`stm32_lcd.h`, `stm32_lcd_ex.h`, `scrl.h`)
- Camera middleware (`cmw_camera.h`, `app_cam.h`)
- NeoChrom GPU (optional, for rotation support)
