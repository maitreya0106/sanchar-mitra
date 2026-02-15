"""Train a Random Forest on ASL hand-landmark data (x,y only) and save asl_model.pkl."""
import os
import csv
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MODEL_FILE = 'asl_model.pkl'
KAGGLE_DIR = os.path.join('kaggle_dataset', 'landmarks')
OWN_DATA   = 'asl_dataset.csv'

SKIP_LABELS = {'j', 'z'}   # motion-based letters
CUSTOM_LABELS = {'send'}   # non-letter labels (e.g., trigger gestures)
NUM_LANDMARKS = 21
# We use x,y only → 42 features per sample


def extract_xy(coords_63):
    """From a flat array of 63 (x,y,z per landmark), keep only x,y → 42 values."""
    xy = []
    for i in range(NUM_LANDMARKS):
        xy.append(coords_63[i * 3])       # x
        xy.append(coords_63[i * 3 + 1])   # y
    return xy


def normalize_xy(xy_42):
    """Translate so wrist is origin, scale by distance to middle-finger MCP."""
    wx, wy = xy_42[0], xy_42[1]
    translated = []
    for i in range(NUM_LANDMARKS):
        translated.append(xy_42[i * 2]     - wx)
        translated.append(xy_42[i * 2 + 1] - wy)
    # landmark 9 = middle finger MCP
    mx, my = translated[9 * 2], translated[9 * 2 + 1]
    scale = max((mx ** 2 + my ** 2) ** 0.5, 1e-6)
    return [v / scale for v in translated]


# ------------------------------------------------------------------
# Loaders
# ------------------------------------------------------------------
def load_kaggle():
    """Load pre-consolidated X_train_xy.npy / y_train.npy (fast),
       or fall back to individual .npy files in kaggle_dataset/landmarks/."""
    # Fast path: consolidated files
    if os.path.exists('X_train_xy.npy') and os.path.exists('y_train.npy'):
        X = np.load('X_train_xy.npy')
        y = np.load('y_train.npy')
        # Normalize
        Xn = []
        for row in X:
            Xn.append(normalize_xy(row.tolist()))
        return np.array(Xn), y

    # Slow path: individual .npy files
    X, y = [], []
    if not os.path.isdir(KAGGLE_DIR):
        print(f"  Directory not found: {KAGGLE_DIR}")
        return np.array(X), np.array(y)

    for letter in sorted(os.listdir(KAGGLE_DIR)):
        letter_dir = os.path.join(KAGGLE_DIR, letter)
        if not os.path.isdir(letter_dir):
            continue
        label = letter.strip().lower()
        if label in SKIP_LABELS:
            continue

        count = 0
        for fname in sorted(os.listdir(letter_dir)):
            if not fname.endswith('.npy'):
                continue
            filepath = os.path.join(letter_dir, fname)
            data = np.load(filepath)
            if data.shape == (63,):
                xy = extract_xy(data.tolist())
                norm = normalize_xy(xy)
                X.append(norm)
                y.append(label)
                count += 1
            elif data.shape == (21, 3):
                flat = data.flatten().tolist()
                xy = extract_xy(flat)
                norm = normalize_xy(xy)
                X.append(norm)
                y.append(label)
                count += 1

        if count:
            print(f"  {letter}: {count} samples")

    return np.array(X), np.array(y)


def load_own():
    """Load our own collected CSV data (already x,y normalized)."""
    X, y = [], []
    if not os.path.exists(OWN_DATA):
        return np.array(X), np.array(y)
    with open(OWN_DATA, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            label = row[-1].strip().lower()
            if label in SKIP_LABELS:
                continue
            coords = [float(v) for v in row[:-1]]
            if len(coords) == 42:
                X.append(coords)
                y.append(label)
    return np.array(X), np.array(y)


# ------------------------------------------------------------------
def main():
    print("=== Loading Kaggle .npy dataset ===")
    Xk, yk = load_kaggle()
    print(f"Kaggle total: {len(Xk)} samples\n")

    print("=== Loading own dataset ===")
    Xo, yo = load_own()
    print(f"Own total:    {len(Xo)} samples\n")

    # Combine
    parts_X, parts_y = [], []
    if len(Xk): parts_X.append(Xk); parts_y.append(yk)
    if len(Xo): parts_X.append(Xo); parts_y.append(yo)
    if not parts_X:
        print("ERROR: No data found! Run download_dataset.py first.")
        return

    X = np.vstack(parts_X)
    y = np.concatenate(parts_y)
    classes = sorted(set(y))
    print(f"Total: {len(X)} samples, {len(classes)} classes: {classes}\n")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    print("Training Random Forest (200 trees) ...")
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    accuracy = (y_pred == y_test).mean()
    print(f"Overall Accuracy: {accuracy:.1%}\n")

    # Save
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Model saved to {MODEL_FILE}  ({os.path.getsize(MODEL_FILE):,} bytes)")


if __name__ == '__main__':
    main()
