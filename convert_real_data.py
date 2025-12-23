import json
from pathlib import Path

INPUT = Path("data/processed/action_items.jsonl")
OUTPUT = Path("data/processed/real_converted.jsonl")

def normalize_actions(actions):
    if actions is None:
        return []
    if isinstance(actions, list):
        return actions
    if isinstance(actions, dict):
        return [actions]
    return []  # fallback safety

def main():
    converted = []

    with open(INPUT, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)

            actions = normalize_actions(rec.get("actions"))

            converted.append({
                "instruction": "Extract action items from the meeting transcript.",
                "input": rec["transcript"],
                "output": {
                    "action_items": actions
                }
            })

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for r in converted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ Converted {len(converted)} real records → {OUTPUT}")

if __name__ == "__main__":
    main()
