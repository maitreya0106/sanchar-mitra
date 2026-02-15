"""
stt.py — Speech-to-Text → UART sender for Sanchar Mitra

Listens to laptop microphone using Whisper (offline STT via faster-whisper),
transcribes speech, and sends the recognized text to the STM32N6570-DK board
via UART.

Protocol:  \x02text\x03   (STX/ETX framed)
  - STX (0x02) marks the start of a new sentence and resets the board buffer.
  - ETX (0x03) marks the end — the board finalizes and displays it.
  - If a previous sentence was incomplete, a new STX always resets cleanly.
The board's uart_receive_task distinguishes this from sign-letter bytes (A-Z).

Run INSTEAD of uart.py — they share the same COM port.

Usage:
    python stt.py              # default COM3
    python stt.py --port COM5  # different port

Controls:
    Speak naturally — recognized sentences are sent automatically.
    Press Ctrl+C to exit.
"""

import sys
import queue
import threading

import numpy as np
import serial
import sounddevice as sd
from faster_whisper import WhisperModel

# Framing bytes
STX = b'\x02'   # Start of sentence
ETX = b'\x03'   # End of sentence

# ─── Configuration ───────────────────────────────────────────────────────────
SERIAL_PORT = 'COM3'
BAUD_RATE = 115200
SAMPLE_RATE = 16000          # Whisper needs 16 kHz mono
BLOCK_SIZE = 1600            # 100ms per callback block
MODEL_SIZE = "base"          # Whisper model: tiny, base, small, medium, large-v3
SILENCE_THRESHOLD = 500      # RMS threshold below which audio is "silence"
SILENCE_DURATION = 1.5       # seconds of silence after speech to trigger transcription
MIN_SPEECH_DURATION = 0.5    # minimum seconds of speech worth transcribing

# Whisper sometimes hallucinates on silence — filter these out
HALLUCINATION_FILTER = {
    "", "[blank_audio]", "(music)", "thank you", "thanks for watching",
    "you", "the", "bye", "okay",
}

# ─── UART drain thread ──────────────────────────────────────────────────────
def uart_drain(ser: serial.Serial, stop_event: threading.Event):
    """Read and discard incoming UART data (board still sends landmarks)."""
    while not stop_event.is_set():
        try:
            ser.read(256)  # non-blocking drain (timeout from serial obj)
        except Exception:
            break

# ─── Audio utilities ─────────────────────────────────────────────────────────
def rms(audio_chunk: np.ndarray) -> float:
    """Compute RMS energy of an int16 audio chunk."""
    return float(np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2)))

# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    # Parse args
    port = SERIAL_PORT
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == '--port' and i < len(sys.argv) - 1:
            port = sys.argv[i + 1]

    # Load Whisper model (auto-downloads on first run)
    print(f"Loading Whisper model ({MODEL_SIZE})...")
    print("(First run will download the model automatically)\n")
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    print("Model loaded.\n")

    # Open UART
    ser = serial.Serial(port, BAUD_RATE, timeout=0.5)
    print(f"Connected to {port} at {BAUD_RATE} baud.")

    # Start drain thread (discard incoming landmarks)
    stop_evt = threading.Event()
    drain_t = threading.Thread(target=uart_drain, args=(ser, stop_evt), daemon=True)
    drain_t.start()

    # Audio queue for sounddevice callback
    audio_q: queue.Queue = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"  [audio] {status}", file=sys.stderr, flush=True)
        audio_q.put(indata.copy())

    print("\n" + "=" * 60)
    print("  SPEECH-TO-TEXT MODE  (Whisper offline)")
    print("  Speak into your microphone.")
    print("  Recognized text is sent to the board via UART.")
    print("  Press Ctrl+C to exit.")
    print("=" * 60 + "\n")

    # State for silence-based speech segmentation
    speech_buffer: list[np.ndarray] = []  # accumulated audio chunks
    silence_counter = 0                    # consecutive silent blocks
    is_speaking = False
    blocks_per_sec = SAMPLE_RATE / BLOCK_SIZE
    silence_blocks = int(SILENCE_DURATION * blocks_per_sec)
    min_speech_blocks = int(MIN_SPEECH_DURATION * blocks_per_sec)

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE,
            dtype='int16', channels=1, callback=audio_callback
        ):
            while True:
                chunk = audio_q.get()
                energy = rms(chunk)

                if energy > SILENCE_THRESHOLD:
                    # Speech detected
                    silence_counter = 0
                    is_speaking = True
                    speech_buffer.append(chunk)
                    print("\r  [listening...]                              ",
                          end='', flush=True)
                elif is_speaking:
                    # Silence while we were speaking — count it
                    silence_counter += 1
                    speech_buffer.append(chunk)  # keep trailing silence

                    if silence_counter >= silence_blocks:
                        # Enough silence after speech → transcribe
                        if len(speech_buffer) >= min_speech_blocks:
                            # Concat and convert int16 → float32 [-1, 1]
                            audio = (np.concatenate(speech_buffer, axis=0)
                                     .flatten()
                                     .astype(np.float32) / 32768.0)

                            print("\r  [transcribing...]                           ",
                                  end='', flush=True)

                            segments, _ = model.transcribe(
                                audio, beam_size=5, language="en",
                                vad_filter=True,
                            )
                            text = " ".join(
                                seg.text.strip() for seg in segments
                            ).strip()

                            if text.lower() not in HALLUCINATION_FILTER:
                                # Send to board:  STX + text + ETX
                                payload = text.encode('ascii',
                                                      errors='replace')
                                ser.write(STX + payload + ETX)
                                ser.flush()
                                print(f"\r  >> Sent: \"{text}\""
                                      "                              ",
                                      flush=True)
                            else:
                                print("\r                                              ",
                                      end='', flush=True)
                        else:
                            print("\r                                              ",
                                  end='', flush=True)

                        # Reset state
                        speech_buffer.clear()
                        silence_counter = 0
                        is_speaking = False

    except KeyboardInterrupt:
        print("\n\nExiting STT mode.", flush=True)
    finally:
        stop_evt.set()
        ser.close()
        print("Serial port closed.")


if __name__ == '__main__':
    main()
