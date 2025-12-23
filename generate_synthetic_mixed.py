import json
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# =========================
# CONFIG
# =========================
OUTPUT_DIR = Path("data/synthetic_mixed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "gpt-4o"  # fast + sufficient
N_SAMPLES = 40

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# PROMPT
# =========================
PROMPT = """
Generate a realistic business meeting transcript.

Requirements:
- 8–15 dialogue turns
- 2–4 speakers
- Mostly discussion and vague ideas
- EXACTLY ONE clear action item
- Action must use explicit commitment language ("I'll", "I can take", "We need to")
- No deadlines unless explicitly stated
"""

# =========================
# MAIN
# =========================
def main():
    records = []

    for i in range(N_SAMPLES):
        print(f"Generating synthetic mixed {i+1}/{N_SAMPLES}")

        response = client.responses.create(
                    model=MODEL_NAME,
                    temperature=0.4,
                    input=[
                        {"role": "system", "content": "You generate realistic workplace conversations and structured outputs."},
                        {"role": "user", "content": PROMPT + """
                        
                Return VALID JSON ONLY in this exact format:
                {{
                "transcript": string,
                "action_items": [
                    {
                    "action": string,
                    "owner": string | null,
                    "deadline": string | null
                    }
                ]
                }}
                """}
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
            print("⚠️ JSON parse failed, skipping example")
            continue

        # Enforce exactly ONE action item
        if "action_items" not in data or len(data["action_items"]) != 1:
            print("⚠️ Not a mixed example, skipping")
            continue

        record = {
            "instruction": "Extract action items from the meeting transcript.",
            "input": data["transcript"],
            "output": {"action_items": data["action_items"]}
        }

        records.append(record)

    out = OUTPUT_DIR / "synthetic_mixed.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(records)} mixed examples → {out}")


if __name__ == "__main__":
    main()
