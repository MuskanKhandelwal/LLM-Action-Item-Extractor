import json
import random
from pathlib import Path

# =========================
# CONFIG
# =========================

INPUT_FILES = [
    "data/processed/real_converted.jsonl",
    "data/synthetic_positive/synthetic_positive.jsonl",
    "data/synthetic_mixed/synthetic_mixed.jsonl",
    "data/synthetic_negative/synthetic_negative.jsonl",
]

OUTPUT_DIR = Path("data/merged")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_OUT = OUTPUT_DIR / "train.jsonl"
EVAL_OUT = OUTPUT_DIR / "eval.jsonl"

EVAL_RATIO = 0.1   # 10% holdout
RANDOM_SEED = 42

# =========================
# UTILS
# =========================

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def validate_record(rec):
    assert "instruction" in rec
    assert "input" in rec
    assert "output" in rec
    assert "action_items" in rec["output"]
    assert isinstance(rec["output"]["action_items"], list)


# =========================
# MAIN
# =========================

def main():
    all_records = []

    for file in INPUT_FILES:
        path = Path(file)
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        records = load_jsonl(path)
        print(f"Loaded {len(records)} from {path}")

        for r in records:
            validate_record(r)

        all_records.extend(records)

    print(f"\nTotal before shuffle: {len(all_records)}")

    # Shuffle deterministically
    random.seed(RANDOM_SEED)
    random.shuffle(all_records)

    # Split
    eval_size = int(len(all_records) * EVAL_RATIO)
    eval_records = all_records[:eval_size]
    train_records = all_records[eval_size:]

    # Write train
    with open(TRAIN_OUT, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write eval
    with open(EVAL_OUT, "w", encoding="utf-8") as f:
        for r in eval_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✅ Train set: {len(train_records)} → {TRAIN_OUT}")
    print(f"✅ Eval set:  {len(eval_records)} → {EVAL_OUT}")


if __name__ == "__main__":
    main()
