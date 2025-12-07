import csv
import os
from datetime import datetime
from openai import OpenAI

# ---------- CONFIG ----------
INPUT_CSV = "questions.csv"
OUTPUT_CSV = "responses.csv"

MODEL_NAME = "gpt-4.1-mini" #for the purpose of testing, I will use this in here

COND_BASE = "C_base"
COND_UNCERTAINTY = "C_uncertainty"

SYSTEM_PROMPT_BASE = (
    "You are a helpful assistant answering student questions. "
    "Answer clearly and concisely."
)

SYSTEM_PROMPT_UNCERTAINTY = (
    "You are a careful assistant answering student questions. "
    "If you are not confident or lack the necessary information, "
    "explicitly say that you are unsure and encourage the student to "
    "check reliable sources. Avoid inventing specific facts."
)

# ---------- OPENAI CLIENT ----------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- HELPERS ----------
def load_questions(path):
    questions = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append({
                "id": row["id"],
                "question": row["question"],
                "answer": row.get("answer", "").strip()
            })
    return questions


def ask_model(question_text, system_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question_text},
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.2,
        max_tokens=256,
    )

    return response.choices[0].message.content


def ensure_output_file(path):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "id",
                "question",
                "canonical_answer",
                "condition",
                "model_name",
                "response",
                "timestamp"
            ])


def append_response(path, row_dict):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            row_dict["id"],
            row_dict["question"],
            row_dict["canonical_answer"],
            row_dict["condition"],
            row_dict["model_name"],
            row_dict["response"],
            row_dict["timestamp"],
        ])


# ---------- MAIN ----------

def main():
    questions = load_questions(INPUT_CSV)
    ensure_output_file(OUTPUT_CSV)

    print(f"Loaded {len(questions)} questions.")
    print(f"Writing responses to {OUTPUT_CSV}")

    for q in questions:
        qid = q["id"]
        text = q["question"]
        canonical = q["answer"]

        print(f"\nQuestion {qid}: {text}")

        # Baseline condition
        resp_base = ask_model(text, SYSTEM_PROMPT_BASE)
        print(f"[{COND_BASE}] {resp_base[:120]}...")

        append_response(OUTPUT_CSV, {
            "id": qid,
            "question": text,
            "canonical_answer": canonical,
            "condition": COND_BASE,
            "model_name": MODEL_NAME,
            "response": resp_base,
            "timestamp": datetime.utcnow().isoformat()
        })

        # Uncertainty-aware condition
        resp_unc = ask_model(text, SYSTEM_PROMPT_UNCERTAINTY)
        print(f"[{COND_UNCERTAINTY}] {resp_unc[:120]}...")

        append_response(OUTPUT_CSV, {
            "id": qid,
            "question": text,
            "canonical_answer": canonical,
            "condition": COND_UNCERTAINTY,
            "model_name": MODEL_NAME,
            "response": resp_unc,
            "timestamp": datetime.utcnow().isoformat()
        })

    print("\nDone.")


if __name__ == "__main__":
    main()
