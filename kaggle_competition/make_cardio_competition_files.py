"""
Build Kaggle competition files from the Cardiovascular Disease dataset
(kaggle.com/datasets/sulianova/cardiovascular-disease-dataset, file cardio_train.csv).

Outputs (in ./competition_files/):
  train.csv              id + 11 features + cardio (0/1)        -> give to students
  test.csv               id + 11 features (no target)           -> give to students
  sample_submission.csv  id + cardio (all 0)                    -> give to students
  solution.csv           id + cardio + Usage (Public/Private)   -> KEEP PRIVATE, upload as solution

Usage:
  python make_cardio_competition_files.py cardio_train.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42            # change each semester so old files can't be reused
TEST_SIZE = 0.30     # share of rows held out for scoring
PUBLIC_SHARE = 0.50  # share of held-out rows shown on the public leaderboard
TARGET = "cardio"

# Look for the data and write the output next to this script,
# no matter which folder the script is run from
HERE = Path(__file__).resolve().parent
OUT = HERE / "competition_files"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else HERE / "cardio_train.csv"
    if not Path(path).exists():
        sys.exit(f"Can't find {path}\nDownload cardio_train.csv from Kaggle "
                 f"and put it in {HERE}")
    df = pd.read_csv(path, sep=";")  # the original file is semicolon-separated

    # Replace the original id with shuffled ids so row order can't leak anything
    df = df.drop(columns="id", errors="ignore")
    rng = np.random.default_rng(SEED)
    df.insert(0, "id", rng.permutation(len(df)) + 100000)

    # Stratified split keeps the ~50/50 balance identical in train and test
    train, test = train_test_split(df, test_size=TEST_SIZE,
                                   stratify=df[TARGET], random_state=SEED)
    test = test.sort_values("id")

    # Public/Private split of the test set, also stratified
    pub_idx, _ = train_test_split(test.index, train_size=PUBLIC_SHARE,
                                  stratify=test[TARGET], random_state=SEED)
    usage = np.where(test.index.isin(pub_idx), "Public", "Private")

    OUT.mkdir(exist_ok=True)
    train.sort_values("id").to_csv(OUT / "train.csv", index=False)
    test.drop(columns=TARGET).to_csv(OUT / "test.csv", index=False)
    pd.DataFrame({"id": test["id"], TARGET: test[TARGET], "Usage": usage}) \
        .to_csv(OUT / "solution.csv", index=False)
    pd.DataFrame({"id": test["id"], TARGET: 0}) \
        .to_csv(OUT / "sample_submission.csv", index=False)

    # Sanity checks
    assert set(train["id"]).isdisjoint(test["id"])
    assert TARGET not in pd.read_csv(OUT / "test.csv").columns
    print(f"Total rows:        {len(df):,}  (positive rate {df[TARGET].mean():.3f})")
    print(f"train.csv:         {len(train):,} rows")
    print(f"test.csv:          {len(test):,} rows")
    print(f"  Public / Private: {(usage == 'Public').sum():,} / {(usage == 'Private').sum():,}")
    print(f"Files written to {OUT.resolve()}")


if __name__ == "__main__":
    main()
