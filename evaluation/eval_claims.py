import csv
import os
import sys
import time
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline import ClaimVerificationPipeline

# expected=None means "show the result, but don't score it"
EVAL_SET = [
    ("The Eiffel Tower is located in Paris", "TRUE"),
    ("The Eiffel Tower was the tallest structure in the world", "TRUE"),
    ("The Eiffel Tower is the tallest structure in the world", "FALSE"),
    ("World War II ended in 1945", "TRUE"),
    ("Elizabeth II is the current Queen of the United Kingdom", "FALSE"),
    ("The Great Wall of China is one continuous wall", "FALSE"),
    ("The Great Wall of China can be seen from space", "MISLEADING"),
    ("Albert Einstein developed the theory of relativity", "TRUE"),
    ("Einstein failed math in school", "FALSE"),
    ("Python is a programming language created by Guido van Rossum", "TRUE"),
    ("The currency of Japan is the yen", "TRUE"),
    ("Bats are completely blind", "FALSE"),
    ("Napoleon Bonaparte was extremely short for his era", "MISLEADING"),
    ("Mount Everest is the tallest mountain on Earth by elevation", "TRUE"),
    ("The Statue of Liberty is located in Los Angeles", "FALSE"),
    ("BTS has an album named Arirang", "TRUE"),
    ("Deadline is an EP released by BTS", "FALSE"),
    ("BTS is currently on the Arirang world tour.", "TRUE"),
]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(RESULTS_DIR, f"eval_{timestamp}.csv")
    txt_path = os.path.join(RESULTS_DIR, f"eval_{timestamp}.txt")

    pipeline = ClaimVerificationPipeline()

    correct = scored = 0
    misses = []
    latencies = []
    rows = []
    lines = []

    def log(line=""):
        print(line)
        lines.append(line)

    log(f"{'Expected':<11}{'Actual':<12}{'Source':<28}Claim")
    log("-" * 100)

    for claim, expected in EVAL_SET:
        start = time.time()
        r = pipeline.verify_single_claim(claim)

        actual = r["verdict"]
        match = ""
        if expected is not None:
            scored += 1
            if actual == expected:
                correct += 1
                match = "yes"
            else:
                match = "no"
                misses.append((claim, expected, actual))

        rows.append({
            "claim": claim,
            "expected": expected or "n/a",
            "actual": actual,
            "match": match,
            "source_title": r.get("source_title"),
            "citation_verified": r.get("citation_verified"),
            "supporting_quote": r.get("supporting_quote"),
            "explanation": r.get("explanation")
        })

        log(f"{(expected or 'n/a'):<11}{actual:<12}{str(r['source_title'])[:26]:<28}{claim}")
        time.sleep(1)  # stay under Groq free-tier rate limits

    log("-" * 100)
    log(f"Accuracy: {correct}/{scored} ({correct / scored * 100:.1f}%)")
    if misses:
        log("\nMisses:")
        for claim, expected, actual in misses:
            log(f"  - {claim}: expected {expected}, got {actual}")

    # Full details (explanations, quotes) as CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Same table + summary as plain text
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {txt_path}")