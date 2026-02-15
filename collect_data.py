"""
Collect your own ASL training data via UART from the STM32N6 (x,y only).

Usage:
  python collect_data.py                  # interactive mode
  python collect_data.py A 50             # collect 50 samples for letter A
  python collect_data.py A                # collect 30 samples for letter A (default)
  python collect_data.py send 50          # collect 50 samples for custom label 'send'
"""
import serial
import time
import csv
import os
import sys

SERIAL_PORT = 'COM3'
BAUD_RATE = 115200
DATA_FILE = 'asl_dataset.csv'
NUM_LANDMARKS = 21
DEFAULT_SAMPLES = 30          # samples per letter


def parse_landmarks(line):
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
    # Keep only x, y → 42 values
    xy = []
    for i in range(NUM_LANDMARKS):
        xy.append(coords[i * 3])
        xy.append(coords[i * 3 + 1])
    return xy


def normalize_xy(xy_42):
    wx, wy = xy_42[0], xy_42[1]
    translated = []
    for i in range(NUM_LANDMARKS):
        translated.append(xy_42[i * 2]     - wx)
        translated.append(xy_42[i * 2 + 1] - wy)
    mx, my = translated[9 * 2], translated[9 * 2 + 1]
    scale = max((mx ** 2 + my ** 2) ** 0.5, 1e-6)
    return [v / scale for v in translated]


def ensure_header(filepath):
    """Write CSV header if the file doesn't exist yet."""
    if not os.path.exists(filepath):
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            header = []
            for i in range(NUM_LANDMARKS):
                header.extend([f'x{i}', f'y{i}'])
            header.append('label')
            writer.writerow(header)


def collect_letter(ser, letter, num_samples, writer, fileobj):
    """Collect num_samples frames for a single letter."""
    label = letter.lower()
    print(f"\n  >> Hold sign '{letter.upper()}' steady in front of the camera ...")
    print(f"  >> Collecting {num_samples} samples ...\n")
    count = 0
    while count < num_samples:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='replace').strip()
            xy = parse_landmarks(line)
            if xy:
                norm = normalize_xy(xy)
                writer.writerow(norm + [label])
                count += 1
                bar = '█' * count + '░' * (num_samples - count)
                print(f"    [{bar}] {count}/{num_samples}", end='\r')
        else:
            time.sleep(0.01)
    fileobj.flush()
    print(f"\n\n  ✓ Done — {num_samples} samples saved for '{letter.upper()}'")


def main_single(letter, num_samples):
    """Non-interactive: collect for one letter and exit."""
    ensure_header(DATA_FILE)
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to {SERIAL_PORT}")
    # flush stale data
    time.sleep(0.3)
    ser.reset_input_buffer()

    with open(DATA_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        collect_letter(ser, letter, num_samples, writer, f)
    ser.close()
    print(f"\nDataset saved to {DATA_FILE}")


def main_interactive():
    """Original interactive loop."""
    ensure_header(DATA_FILE)
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to {SERIAL_PORT}")
    print("=== ASL Data Collection ===")
    print("Hold a sign, type the letter (a-z), press Enter to record samples.")
    print("Type 'quit' to stop.\n")
    time.sleep(0.3)
    ser.reset_input_buffer()

    with open(DATA_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        while True:
            label = input("Enter label to record (a-z or 'send', 'quit' to stop): ").strip().lower()
            if label == 'quit':
                break
            if not label.isalnum() and label != 'send':
                print("  Please enter a letter a-z or a custom label like 'send'")
                continue
            try:
                n = int(input(f"  How many samples? [{DEFAULT_SAMPLES}]: ").strip() or DEFAULT_SAMPLES)
            except ValueError:
                n = DEFAULT_SAMPLES
            collect_letter(ser, label, n, writer, f)

    ser.close()
    print(f"\nDataset saved to {DATA_FILE}")


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        label = sys.argv[1].strip()
        # Accept single letters (A-Z) or custom labels (e.g., 'send')
        if not label.isalnum() and label.lower() not in ('send',):
            print("Error: argument must be a letter A-Z or a custom label like 'send'")
            sys.exit(1)
        n = int(sys.argv[2]) if len(sys.argv) >= 3 else DEFAULT_SAMPLES
        main_single(label, n)
    else:
        main_interactive()
