import os
import sys
import time

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
    ("Mount Everest is the tallest mountain on Earth by elevation", "TRUE"),
    ("The Statue of Liberty is located in Los Angeles", "FALSE"),
    ("BTS has an album named Arirang", "TRUE"),
    ("BTS is still on the Arirang tour as of now", "TRUE"),
]


if __name__ == "__main__":
    pipeline = ClaimVerificationPipeline()

    correct = scored = 0
    misses = []
    latencies = []

    print(f"{'Expected':<11}{'Actual':<12}{'Source':<28}Claim")
    print("-" * 100)
    for claim, expected in EVAL_SET:
        start = time.time()
        r = pipeline.verify_single_claim(claim)
        latencies.append(time.time() - start)

        actual = r["verdict"]
        if expected is not None:
            scored += 1
            if actual == expected:
                correct += 1
            else:
                misses.append((claim, expected, actual))

        print(f"{(expected or 'n/a'):<11}{actual:<12}{str(r['source_title'])[:26]:<28}{claim}")
        time.sleep(1)  # stay under Groq free-tier rate limits

    print("-" * 100)
    print(f"Accuracy: {correct}/{scored} ({correct / scored * 100:.1f}%)")
    print(f"Average latency: {sum(latencies) / len(latencies):.2f}s per claim")
    if misses:
        print("\nMisses:")
        for claim, expected, actual in misses:
            print(f"  - {claim}: expected {expected}, got {actual}")