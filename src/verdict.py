import os
import json
from datetime import date
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])


VERDICT_PROMPT = """You are a careful fact-checking assistant. Today's date is {today}.

Your job is to verify the claim below using ONLY the evidence provided. Do not use outside knowledge, even if you are confident about it — base your verdict strictly on what the evidence says.

Verdict definitions:
- TRUE: the evidence directly and explicitly supports the claim.
- FALSE: the evidence directly contradicts the claim. This also applies when the evidence is an explicitly comprehensive, exhaustive account of the relevant category (e.g., a full discography, filmography, official roster, or complete list of positions/roles held) and the claimed item does not appear in it — absence from a genuinely exhaustive list is valid evidence of falsehood.
- MISLEADING: the claim is technically true in isolation, but the evidence shows important missing context, an outdated status, or a framing that would give a reader a false impression.
- UNKNOWN: the evidence does not address the claim's subject at all, or only gives incidental/partial information that does not amount to a complete accounting of the relevant category. Do not use UNKNOWN merely because the evidence doesn't use the exact wording of the claim — reason about substance. But do use UNKNOWN when the evidence is general prose (e.g. a biography paragraph) that simply doesn't happen to mention the claimed fact, rather than a structured, exhaustive list that would be expected to include it.

Time-sensitive claims: if the claim refers to something being "current," "ongoing," "still happening," or similar, compare any dates in the evidence against today's date above. If the evidence gives a clear end date that has passed, or a clear start date with no indication of an end, reason accordingly rather than defaulting to UNKNOWN. Only return UNKNOWN if the evidence gives no relevant dates or status information at all.

Claim: "{claim}"

Evidence:
{evidence}

Rules for your response:
1. The "supporting_quote" must be copied exactly, word-for-word, from the evidence above. Never paraphrase it, complete it from memory, or write a quote that does not appear in the evidence text.
2. If your verdict is based on absence from an exhaustive list, set "supporting_quote" to the exact text of that list (or the relevant portion), and explain the absence reasoning in "explanation".
3. If verdict is UNKNOWN, set "supporting_quote" to an empty string.
4. Keep "explanation" concise (1-3 sentences) and specific to what the evidence actually says.

Respond with ONLY valid JSON in this exact format, and nothing else — no markdown, no preamble:
{{
  "verdict": "TRUE" | "FALSE" | "MISLEADING" | "UNKNOWN",
  "explanation": "a concise 1-3 sentence explanation grounded in the evidence",
  "supporting_quote": "an exact quote from the evidence, or empty string if UNKNOWN"
}}"""


def generate_verdict(claim, retrieved_chunks, model="openai/gpt-oss-120b"):
    evidence_text = "\n\n---\n\n".join([c["text"] for c in retrieved_chunks])
    today = date.today().strftime("%B %d, %Y")

    prompt = VERDICT_PROMPT.format(claim=claim, evidence=evidence_text, today=today)

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
        result = json.loads(raw_output)
    except json.JSONDecodeError:
        result = {
            "verdict": "UNKNOWN",
            "explanation": "Could not parse a valid verdict from the model.",
            "supporting_quote": "",
        }

    return result


if __name__ == "__main__":
    from wiki_search import get_best_matching_page
    from retrieval import get_embedding_model, chunk_text, embed_and_index, retrieve_relevant_chunks

    claim = "The Eiffel Tower is located in Paris"

    page = get_best_matching_page(claim)
    print(f"Retrieved article: {page.title}")

    chunks = chunk_text(page.text)
    model = get_embedding_model()
    index = embed_and_index(model, chunks)
    retrieved = retrieve_relevant_chunks(model, index, chunks, claim, top_k=3)

    verdict = generate_verdict(claim, retrieved)

    print(f"\nClaim: {claim}")
    print(f"Verdict: {verdict['verdict']}")
    print(f"Explanation: {verdict['explanation']}")
    print(f"Supporting quote: {verdict['supporting_quote']}")