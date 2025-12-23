import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import os
# =========================
# CONFIG
# =========================

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "gpt-4o"   # any strong LLM works
openai_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_key)


BOOTSTRAP_PROMPT = """
You are labeling meeting transcripts for a machine learning dataset.

Task:
Extract all action items from the transcript below.

Rules:
- Only include concrete tasks
- Do NOT hallucinate owners or deadlines
- If owner or deadline is missing, use null
- If there are no action items, return an empty list
- Return VALID JSON ONLY (no markdown, no explanation)

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
# UTILS
# =========================

def read_text_file(path: Path) -> str:
    """
    Read text file with encoding fallback (real-world safe).
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return f.read().strip()


def clean_json_output(text: str) -> str:
    """
    Remove markdown code fences if present.
    """
    text = text.strip()

    if text.startswith("```"):
        # Remove opening fence
        text = text.split("\n", 1)[1]
        # Remove closing fence
        text = text.rsplit("```", 1)[0]

    return text.strip()



def bootstrap_label_transcript(transcript: str):
    prompt = BOOTSTRAP_PROMPT.format(transcript=transcript)

    response = client.responses.create(
        model=MODEL_NAME,
        temperature=0.0,
        input=[
            {
                "role": "system",
                "content": "You are a careful data annotator."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    raw_output = response.output_text.strip()
    cleaned_output = clean_json_output(raw_output)

    try:
        return json.loads(cleaned_output)
    except json.JSONDecodeError:
        print("⚠️ JSON parse failed")
        print(cleaned_output)
        return []

# =========================
# MAIN PIPELINE
# =========================

def main():
    transcripts = {}
    labeled_data = {}

    # Step 1: Read transcripts
    for txt_file in DATA_DIR.glob("*.txt"):
        text = read_text_file(txt_file)
        transcripts[txt_file.stem] = text

    print(f"Loaded {len(transcripts)} transcripts")

    # Step 2: Bootstrap labeling
    for conv_id, transcript in transcripts.items():
        print(f"Labeling {conv_id}...")

        actions = bootstrap_label_transcript(transcript)

        labeled_data[conv_id] = {
            "transcript": transcript,
            "actions": actions
        }

    # Step 3: Save for manual review
    review_path = OUTPUT_DIR / "bootstrap_labeled2.json"
    with open(review_path, "w", encoding="utf-8") as f:
        json.dump(labeled_data, f, indent=2, ensure_ascii=False)

    print(f"Saved bootstrap labels to {review_path}")

    # Step 4: Save JSONL for training
    jsonl_path = OUTPUT_DIR / "action_items.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for conv_id, item in labeled_data.items():
            record = {
                "id": conv_id,
                "transcript": item["transcript"],
                "actions": item["actions"]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved training file to {jsonl_path}")


if __name__ == "__main__":
    main()
