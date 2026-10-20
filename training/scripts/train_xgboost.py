"""Train the three XGBoost predictors from an aligned feature/target CSV."""

import argparse
from pathlib import Path

import pandas as pd
from pvcracks_training.powerloss.training import train_xgb_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=30)
    args = parser.parse_args()
    print(train_xgb_models(pd.read_csv(args.table), args.output_dir, seed=args.seed))


if __name__ == "__main__":
    main()
