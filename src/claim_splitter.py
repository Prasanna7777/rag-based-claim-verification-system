import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])


SPLIT_PROMPT = """Split the following text into individual, standalone factual claims.
Each claim should be a single, self-contained statement that can be independently verified.

Rules:
- If the text is already a single claim, return it as a list with one item.
- Do not add any information that isn't in the original text.
- Make each claim understandable on its own (e.g., resolve "it"/"he" to the actual subject if the original text makes that clear).
- Ignore opinions, questions, or non-factual statements.

Text: "{text}"

Respond with ONLY a valid JSON array of strings, nothing else. Example format:
["claim one", "claim two"]"""


def split_into_claims(text, model="openai/gpt-oss-120b"):
    prompt = SPLIT_PROMPT.format(text=text)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    raw_output = response.choices[0].message.content.strip()

    if raw_output.startswith("```"):
        raw_output = raw_output.strip("`")
        if raw_output.startswith("json"):
            raw_output = raw_output[4:].strip()

    try:
        claims = json.loads(raw_output)
        if isinstance(claims, list) and all(isinstance(c, str) for c in claims):
            return claims
    except json.JSONDecodeError:
        pass

    # fallback: if parsing fails, treat the whole input as one claim
    return [text.strip()]


if __name__ == "__main__":
    test_inputs = [
        "The Eiffel Tower is located in Paris",
        "The Eiffel Tower is in Paris, and Albert Einstein was born in Germany",
        "India is a South Asian country. It shares a border with China and Pakistan.",
    ]

    for text in test_inputs:
        claims = split_into_claims(text)
        print(f"Input: {text}")
        print(f"Split into {len(claims)} claim(s):")
        for c in claims:
            print(f"  - {c}")
        print()