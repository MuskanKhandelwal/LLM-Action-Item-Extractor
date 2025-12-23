import json
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# =========================
# CONFIG
# =========================

OUTPUT_DIR = Path("data/synthetic_positive")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "gpt-4o"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

N_SAMPLES = 250
TEMPERATURE = 0.4   # controlled variation

# =========================
# PROMPTS
# =========================

TRANSCRIPT_PROMPT = """
Generate a realistic business meeting transcript.

Requirements:
- 8–15 dialogue turns
- 2–4 named speakers
- Include 1–3 CLEAR action items
- Use explicit commitment language ("I'll", "I can take", "We need to")
- Some actions may not have an owner or deadline
- Avoid vague language like "we should consider"
- Output ONLY the transcript text
"""


LABEL_PROMPT = """
You are labeling meeting transcripts for a machine learning dataset.

Task:
Extract all action items from the transcript below.

Rules:
- Only include concrete tasks
- Do NOT hallucinate owners or deadlines
- If owner or deadline is missing, use null
- If there are no action items, return an empty list
- Return VALID JSON ONLY (no markdown)

Schema:
{{
  "action_items": [
    {{
      "action": string,
      "owner": string | null,
      "deadline": string | null
    }}
  ]
}}

Transcript:
{transcript}
"""

# =========================
# HELPERS
# =========================
def clean_json_output(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]

    return text.strip()

def normalize_nulls(obj):
    if isinstance(obj, dict):
        return {k: normalize_nulls(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [normalize_nulls(v) for v in obj]
    elif obj == "null":
        return None
    else:
        return obj


def generate_text(prompt: str, temperature=0.4) -> str:
    response = client.responses.create(
        model=MODEL_NAME,
        temperature=temperature,
        input=[
            {"role": "system", "content": "You generate realistic workplace conversations."},
            {"role": "user", "content": prompt}
        ],
    )

    return response.output_text.strip()


def label_transcript(transcript: str):
    response = client.responses.create(
        model=MODEL_NAME,
        temperature=0.0,
        input=[
            {"role": "system", "content": "You are a careful data annotator."},
            {"role": "user", "content": LABEL_PROMPT.format(transcript=transcript)}
        ],
    )

    raw = response.output_text

    if not raw or not raw.strip():
        print("⚠️ Empty label output")
        return {"action_items": []}

    cleaned = clean_json_output(raw)

    try:
        parsed = json.loads(cleaned)
        parsed = normalize_nulls(parsed)
        return parsed
    except json.JSONDecodeError:
        print("⚠️ JSON parse failed")
        print(cleaned)
        return {"action_items": []}

# =========================
# MAIN
# =========================

def main():
    records = []

    for i in range(N_SAMPLES):
        print(f"Generating synthetic positive {i+1}/{N_SAMPLES}")

        transcript = generate_text(TRANSCRIPT_PROMPT, TEMPERATURE)
        labels = label_transcript(transcript)

        # Safety check: enforce positivity
        if not labels.get("action_items"):
            print("⚠️ Skipping empty action example")
            continue

        record = {
            "instruction": "Extract action items from the meeting transcript.",
            "input": transcript,
            "output": labels
        }

        records.append(record)

    output_path = OUTPUT_DIR / "synthetic_positive.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(records)} synthetic positive examples → {output_path}")


if __name__ == "__main__":
    main()
