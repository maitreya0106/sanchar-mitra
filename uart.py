import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import serial
import time
import pickle
import threading
import queue
import subprocess
import os
import numpy as np
import requests

SERIAL_PORT = 'COM3'
BAUD_RATE = 115200
MODEL_FILE = 'asl_model.pkl'
OLLAMA_MODEL = 'llama3.2:3b'
OLLAMA_URL = 'http://localhost:11434/api/generate'

NUM_LANDMARKS = 21

# --- TTS background thread ---
tts_queue = queue.Queue()
SPEAK_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'speak_heera.ps1')

def tts_worker():
    """Background thread: speaks words using Heera OneCore voice via PowerShell."""
    while True:
        word = tts_queue.get()
        if word is None:
            break
        try:
            subprocess.run(
                ['powershell', '-ExecutionPolicy', 'Bypass',
                 '-File', SPEAK_SCRIPT, '-text', word],
                creationflags=subprocess.CREATE_NO_WINDOW,
                timeout=15
            )
        except Exception as e:
            print(f"  [TTS error] {e}", flush=True)
        tts_queue.task_done()

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()


# --- Ollama background thread ---
ollama_queue = queue.Queue()

OLLAMA_SYSTEM = (
    "give me a 1-2 short reply concise."
)

def ollama_worker(ser):
    """Background thread: sends buffer text to Ollama and streams response to board."""
    while True:
        text = ollama_queue.get()
        if text is None:
            break
        try:
            print(f"\n  [Ollama] Sending: \"{text}\"\n  [Ollama] Thinking...", flush=True)
            resp = requests.post(OLLAMA_URL, json={
                'model': OLLAMA_MODEL,
                'system': OLLAMA_SYSTEM,
                'prompt': text,
                'stream': True,
                'options': {
                    'num_predict': 100,   # cap output length for speed
                    'temperature': 0.7,
                },
            }, timeout=120, stream=True)
            resp.raise_for_status()

            # Stream tokens as they arrive
            chunks = []
            print("  [Ollama] ", end='', flush=True)
            for line in resp.iter_lines():
                if line:
                    try:
                        token_data = __import__('json').loads(line)
                        token = token_data.get('response', '')
                        if token:
                            chunks.append(token)
                            print(token, end='', flush=True)
                        if token_data.get('done', False):
                            break
                    except Exception:
                        pass
            print(flush=True)

            answer = ''.join(chunks).strip()
            if answer:
                print(f"  [Ollama] Done.\n", flush=True)
                # Speak the Ollama response via TTS
                tts_queue.put(answer)
                print("  [TTS] Speaking Ollama response...", flush=True)
            else:
                print("  [Ollama] Empty response.", flush=True)
        except Exception as e:
            print(f"  [Ollama error] {e}", flush=True)
        ollama_queue.task_done()


def parse_landmarks(line):
    """Parse UART line → 63 floats (21 landmarks × 3), then extract x,y only → 42."""
    if 'LM:' not in line:
        return None
    raw = line.split('LM:')[1]
    values = [v.strip() for v in raw.split(',') if v.strip() != '']
    if len(values) != 63:
        return None
    try:
        coords = [float(v) for v in values]
    except ValueError:
        return None
    # Keep only x, y (drop z) → 42 values
    xy = []
    for i in range(NUM_LANDMARKS):
        xy.append(coords[i * 3])       # x
        xy.append(coords[i * 3 + 1])   # y
    return xy


def normalize_xy(xy_42):
    """Translate so wrist is origin, scale by distance to middle-finger MCP."""
    wx, wy = xy_42[0], xy_42[1]
    translated = []
    for i in range(NUM_LANDMARKS):
        translated.append(xy_42[i * 2]     - wx)
        translated.append(xy_42[i * 2 + 1] - wy)
    mx, my = translated[9 * 2], translated[9 * 2 + 1]
    scale = max((mx ** 2 + my ** 2) ** 0.5, 1e-6)
    return [v / scale for v in translated]


def main():
    # Load trained model
    with open(MODEL_FILE, 'rb') as f:
        clf = pickle.load(f)
    print(f"Model loaded from {MODEL_FILE}", flush=True)

    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to {SERIAL_PORT}. Recognizing ASL signs...\n", flush=True)

    # Start Ollama worker thread (needs ser for sending response to board)
    ollama_thread = threading.Thread(target=ollama_worker, args=(ser,), daemon=True)
    ollama_thread.start()

    sentence = []
    current_word = []  # letters of the word being built
    last_letter = None
    repeat_count = 0
    space_added = False  # prevent multiple consecutive spaces
    CONFIRM_FRAMES = 2  # consecutive same predictions to confirm a letter
    MIN_CONFIDENCE = 0.20  # lowered — real-world data has lower confidence
    was_receiving = False  # track if we were getting data before

    try:
        while True:
            line = ser.readline().decode('utf-8', errors='replace').strip()
            if line:
                was_receiving = True
                space_added = False
                xy = parse_landmarks(line)
                if xy:
                    norm = normalize_xy(xy)
                    X = np.array(norm).reshape(1, -1)
                    prediction = clf.predict(X)[0]
                    confidence = max(clf.predict_proba(X)[0])

                    # Always show what the model sees
                    print(
                        f"  >> {prediction.upper()} ({confidence:.0%})",
                        end='',
                        flush=True,
                    )

                    if confidence > MIN_CONFIDENCE:
                        # Send every prediction to the board (unconfirmed)
                        ser.write(prediction.upper().encode())

                        if prediction == last_letter:
                            repeat_count += 1
                        else:
                            repeat_count = 1
                            last_letter = prediction

                        if repeat_count > 0 and repeat_count % CONFIRM_FRAMES == 0:
                            if prediction == 'send':
                                # SEND gesture confirmed → send buffer to Ollama
                                buf_text = ''.join(sentence).strip()
                                if buf_text:
                                    print(
                                        f"  *** SEND confirmed  "
                                        f"|  Sending buffer to Ollama: \"{buf_text}\"",
                                        flush=True,
                                    )
                                    ollama_queue.put(buf_text)
                                else:
                                    print(
                                        "  *** SEND confirmed  |  Buffer empty, nothing to send",
                                        flush=True,
                                    )
                                # Clear buffer
                                sentence.clear()
                                current_word.clear()
                                last_letter = None
                                repeat_count = 0
                            else:
                                sentence.append(prediction.upper())
                                current_word.append(prediction.upper())
                                word = ''.join(sentence)
                                print(
                                    f"  *** Confirmed: {prediction.upper()}  "
                                    f"|  Buffer: {word}",
                                    flush=True,
                                )
                        else:
                            print(f"  ({repeat_count % CONFIRM_FRAMES}/{CONFIRM_FRAMES})", flush=True)
                    else:
                        last_letter = None
                        repeat_count = 0
                        print("  (low conf, skip)", flush=True)
            else:
                # readline timed out — UART stopped sending (no hand)
                if was_receiving and not space_added and sentence and sentence[-1] != ' ':
                    sentence.append(' ')
                    space_added = True
                    last_letter = None
                    repeat_count = 0
                    # TTS: speak the completed word
                    if current_word:
                        finished_word = ''.join(current_word)
                        tts_queue.put(finished_word)
                        print(f"  [TTS] Speaking: {finished_word}", flush=True)
                        current_word.clear()
                    word = ''.join(sentence)
                    print(f"  [no hand] Space added  |  Buffer: {word}", flush=True)
                was_receiving = False
    except KeyboardInterrupt:
        print("\nStopping...", flush=True)
    finally:
        if ser.is_open:
            ser.close()


if __name__ == '__main__':
    try:
        main()
    except serial.SerialException as e:
        print(f"Serial error: {e}", flush=True)
    except FileNotFoundError:
        print(
            f"Model file '{MODEL_FILE}' not found.\n"
            "Run train_model.py first to train the classifier.",
            flush=True,
        )