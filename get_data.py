"""Fetch the Bank Account Fraud (BAF) dataset into data/.

Dataset: https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022
License: CC-BY-NC-SA (non-commercial use; coursework/research OK)

Preferred path uses the Kaggle CLI (needs ~/.kaggle/access_token or
kaggle.json). Without it, download manually from the dataset page and
unzip Base.csv into data/.
"""
import subprocess
import sys
from pathlib import Path

DATA = Path(__file__).parent / "data" / "Base.csv"
DATASET = "sgpjesus/bank-account-fraud-dataset-neurips-2022"


def main():
    if DATA.exists():
        print(f"Base.csv already present: {DATA}")
    else:
        DATA.parent.mkdir(exist_ok=True)
        cmd = ["kaggle", "datasets", "download", "-d", DATASET,
               "-p", str(DATA.parent), "--unzip"]
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError:
            sys.exit(
                "kaggle CLI not found. Either:\n"
                "  pip install kaggle   (then configure ~/.kaggle/access_token)\n"
                "or download manually:\n"
                f"  https://www.kaggle.com/datasets/{DATASET}\n"
                "  → Download → unzip → place Base.csv into data/")
        except subprocess.CalledProcessError as e:
            sys.exit(f"kaggle download failed: {e}")

    import pandas as pd
    df = pd.read_csv(DATA, nrows=5)
    rows = sum(1 for _ in open(DATA, "rb")) - 1
    print(f"Base.csv shape: ({rows}, {df.shape[1]})")


if __name__ == "__main__":
    main()
