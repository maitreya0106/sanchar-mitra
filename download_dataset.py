"""Download ASL hand-landmark dataset from Kaggle.

First-time setup:
  1. Create a free Kaggle account at https://www.kaggle.com
  2. Go to  https://www.kaggle.com/settings  →  API section  →  Create New Token
  3. It will download  kaggle.json  (contains {"username":"...","key":"..."})
  4. Copy that file to:  C:\\Users\\<you>\\.kaggle\\kaggle.json
     OR set environment variables   KAGGLE_USERNAME  and  KAGGLE_KEY
  5. Run this script again.
"""
import kagglehub
import shutil
import os
import sys
import json

DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle_dataset")
KAGGLE_JSON = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")


def check_credentials():
    """Return True if Kaggle credentials are available."""
    # New-style single token
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    # Old-style env vars
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    # kaggle.json file
    if os.path.exists(KAGGLE_JSON):
        return True
    return False


def main():
    if not check_credentials():
        print("Kaggle credentials not found.")
        print("Set environment variable:  $env:KAGGLE_API_TOKEN = 'KGAT_...'")
        print("Or place kaggle.json in ~/.kaggle/")
        sys.exit(1)

    print("\nDownloading ASL MediaPipe landmark dataset from Kaggle ...")
    path = kagglehub.dataset_download("risangbaskoro/asl-alphabet-mediapipe")
    print(f"Downloaded to: {path}")

    if os.path.exists(DEST):
        shutil.rmtree(DEST)
    shutil.copytree(path, DEST)
    print(f"Copied to: {DEST}")

    # Show what we got
    for root, dirs, files in os.walk(DEST):
        for f in files:
            fp = os.path.join(root, f)
            size = os.path.getsize(fp)
            print(f"  {os.path.relpath(fp, DEST)}  ({size:,} bytes)")

    print("\nDone! Now run:  python train_model.py")


if __name__ == "__main__":
    main()
