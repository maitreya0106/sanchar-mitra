"""Batch-collect ASL data for all static letters (skipping J, Z which need motion).
Gives a countdown between letters so you can switch signs."""
import serial
import time
import csv
import os
import sys

SERIAL_PORT = 'COM3'
BAUD_RATE = 115200
DATA_FILE = 'asl_dataset.csv'
NUM_LANDMARKS = 21
SAMPLES_PER_LETTER = 30

# All static ASL letters (no J or Z — they require motion)
ALL_LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")


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


def ensure_header():
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            header = []
            for i in range(NUM_LANDMARKS):
                header.extend([f'x{i}', f'y{i}'])
            header.append('label')
            writer.writerow(header)


def get_existing_letters():
    """Check which letters already have data."""
    if not os.path.exists(DATA_FILE):
        return {}
    with open(DATA_FILE, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        from collections import Counter
        labels = [row[-1].upper() for row in reader if row]
        return dict(Counter(labels))


def collect_letter(ser, letter, num_samples, writer, fileobj):
    label = letter.lower()
    count = 0
    ser.reset_input_buffer()
    while count < num_samples:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='replace').strip()
            xy = parse_landmarks(line)
            if xy:
                norm = normalize_xy(xy)
                writer.writerow(norm + [label])
                count += 1
                done = int(count / num_samples * 30)
                bar = '█' * done + '░' * (30 - done)
                print(f"    [{bar}] {count}/{num_samples}", end='\r')
        else:
            time.sleep(0.01)
    fileobj.flush()
    print()


def main():
    ensure_header()
    existing = get_existing_letters()

    # Figure out which letters still need data
    remaining = [l for l in ALL_LETTERS if l not in existing]

    if not remaining:
        print("All letters already have data!")
        print("Existing:", existing)
        return

    print("=" * 50)
    print("  ASL BATCH DATA COLLECTION")
    print("=" * 50)
    print(f"  Samples per letter : {SAMPLES_PER_LETTER}")
    print(f"  Already collected  : {sorted(existing.keys()) if existing else 'none'}")
    print(f"  Remaining letters  : {' '.join(remaining)}")
    print(f"  Total to collect   : {len(remaining)} letters")
    print("=" * 50)
    print()

    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to {SERIAL_PORT}")
    time.sleep(0.3)
    ser.reset_input_buffer()

    with open(DATA_FILE, 'a', newline='') as f:
        writer = csv.writer(f)

        for idx, letter in enumerate(remaining):
            num_left = len(remaining) - idx
            print(f"\n{'─' * 50}")
            print(f"  NEXT: '{letter}' ({idx+1}/{len(remaining)}, {num_left} left)")
            print(f"{'─' * 50}")

            # Countdown so user can switch hand sign
            for sec in range(5, 0, -1):
                print(f"  Hold '{letter}' sign — starting in {sec}s ...", end='\r')
                time.sleep(1)
            print(f"  Collecting '{letter}' NOW!                        ")

            collect_letter(ser, letter, SAMPLES_PER_LETTER, writer, f)
            print(f"  ✓ '{letter}' done!")

    ser.close()

    # Summary
    final = get_existing_letters()
    print(f"\n{'=' * 50}")
    print(f"  COLLECTION COMPLETE!")
    print(f"  Total samples: {sum(final.values())}")
    print(f"  Letters: {sorted(final.keys())}")
    print(f"  Saved to: {DATA_FILE}")
    print(f"{'=' * 50}")


if __name__ == '__main__':
    main()
