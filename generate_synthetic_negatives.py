import json
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# =========================
# CONFIG
# =========================
OUTPUT_DIR = Path("data/synthetic_negative")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "gpt-4o"   # faster, good enough
N_SAMPLES = 60

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# PROMPT
# =========================
PROMPT = """
Generate a realistic business meeting transcript.

Requirements:
- 8–15 dialogue turns
- 2–4 speakers
- Discussion, status updates, or reflections ONLY
- NO commitments, tasks, follow-ups, or decisions
- Avoid phrases like "I'll", "we'll", "let's", "can you"

Return VALID JSON ONLY in this format:
{
  "transcript": string
}
"""

# =========================
# MAIN
# =========================
def main():
    records = []

    attempts = 0
    while len(records) < N_SAMPLES:
        attempts += 1
        print(f"Generating synthetic negative {len(records)+1}/{N_SAMPLES}")

        response = client.responses.create(
            model=MODEL_NAME,
            temperature=0.4,
            input=[
                {"role": "system", "content": "You generate realistic workplace conversations."},
                {"role": "user", "content": PROMPT}
            ],
        )

        raw = response.output_text.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0]

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            print("⚠️ JSON parse failed, retrying")
            continue

        if "transcript" not in data:
            print("⚠️ Missing transcript, retrying")
            continue

        record = {
            "instruction": "Extract action items from the meeting transcript.",
            "input": data["transcript"],
            "output": {"action_items": []}   # 👈 hard-coded
        }

        records.append(record)

    out = OUTPUT_DIR / "synthetic_negative.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✅ Saved {len(records)} negative examples → {out}")
    print(f"Total attempts: {attempts}")


if __name__ == "__main__":
    main()
