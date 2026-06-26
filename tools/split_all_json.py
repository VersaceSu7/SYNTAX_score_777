import json
import random
from pathlib import Path
import argparse


def split_json(input_path, train_path, test_path, test_ratio=0.2, seed=42):
    input_path = Path(input_path)
    train_path = Path(train_path)
    test_path = Path(test_path)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rng = random.Random(seed)
    rng.shuffle(data)

    n_total = len(data)
    n_test = int(round(n_total * test_ratio))
    test_data = data[:n_test]
    train_data = data[n_test:]

    with train_path.open("w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)

    with test_path.open("w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Split all.json into train/test.")
    parser.add_argument("--input", default="data/all.json")
    parser.add_argument("--train", default="data/all_train.json")
    parser.add_argument("--test", default="data/all_test.json")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_json(args.input, args.train, args.test, args.test_ratio, args.seed)


if __name__ == "__main__":
    main()
