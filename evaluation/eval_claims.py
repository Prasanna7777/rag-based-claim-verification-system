import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline import ClaimVerificationPipeline

# (claim, expected_verdict) — a mix of TRUE/FALSE/MISLEADING/UNKNOWN,
# including known edge cases found during development
EVAL_SET = [
    ("The Great Wall of China is one continuous wall", "FALSE"),
    ("The Great Wall of China can be seen from space", "MISLEADING"),
    ("The Eiffel Tower is located in Paris", "TRUE"),
    ("The Eiffel Tower was the tallest structure in the world", "TRUE"),
    ("The Eiffel Tower is the tallest structure in the world", "FALSE"),
    ("World War II ended in 1945", "TRUE"),
    ("Elizabeth II is the current Queen of the United Kingdom", "FALSE"),
    ("Albert Einstein developed the theory of relativity", "TRUE"),
    ("Einstein failed math in school", "FALSE"),
    ("Python is a programming language created by Guido van Rossum", "TRUE"),
    ("The currency of Japan is the yen", "TRUE"),
    ("Mount Everest is the tallest mountain on Earth by elevation", "TRUE"),
    ("The Statue of Liberty is located in Los Angeles", "FALSE"),
]


def run_evaluation():
    pipeline = ClaimVerificationPipeline()

    correct = 0
    total = len(EVAL_SET)
    results_log = []

    for claim, expected in EVAL_SET:
        result = pipeline.verify_single_claim(claim)
        actual = result["verdict"]
        is_correct = actual == expected

        if is_correct:
            correct += 1

        results_log.append({
            "claim": claim,
            "expected": expected,
            "actual": actual,
            "correct": is_correct,
        })

        status = "✓" if is_correct else "✗"
        print(f"{status} [{expected:>10} / {actual:>10}] {claim}")

    accuracy = correct / total * 100
    print(f"\n{'='*60}")
    print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    print('='*60)

    # breakdown of misses
    misses = [r for r in results_log if not r["correct"]]
    if misses:
        print("\nMisses:")
        for m in misses:
            print(f"  - '{m['claim']}' — expected {m['expected']}, got {m['actual']}")

    return results_log, accuracy


if __name__ == "__main__":
    run_evaluation()