import os
import re
import requests
import spacy
import wikipediaapi
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

wiki = wikipediaapi.Wikipedia(
    user_agent="ClaimVerification/1.0 (student project)",
    language="en"
)

WIKI_SEARCH_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "ClaimVerification/1.0 (student project)"}
session = requests.Session()
session.headers.update(HEADERS)

# Prefer the transformer NER model; fall back to the small one if it isn't installed
try:
    nlp = spacy.load("en_core_web_trf")
except OSError:
    nlp = spacy.load("en_core_web_sm")

RELEVANT_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "FAC", "WORK_OF_ART", "EVENT", "NORP", "PRODUCT"}

MAX_ENTITIES = 2           # how many entities contribute candidates
CANDIDATES_PER_ENTITY = 4  # opensearch results kept per entity


def search_wikipedia_titles(query, limit=6):
    """Full-text search; used only as a fallback."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": limit,
    }
    response = session.get(WIKI_SEARCH_API, params=params, timeout=5)
    response.raise_for_status()
    data = response.json()
    return [result["title"] for result in data["query"]["search"]]


def opensearch_titles(query, limit=5):
    """Title-based matching (the search-box autocomplete API)."""
    params = {
        "action": "opensearch",
        "search": query,
        "limit": limit,
        "namespace": 0,
        "format": "json",
    }
    response = session.get(WIKI_SEARCH_API, params=params, timeout=5)
    response.raise_for_status()
    data = response.json()
    return data[1] if len(data) > 1 else []


def get_wikipedia_page(query):
    page = wiki.page(query)
    if not page.exists():
        return None
    return page


def extract_entities_fast(claim_text):
    """
    Extracts named entities with spaCy NER, ordered by where they appear in
    the claim (the subject usually comes first), not by length.
    Falls back to a capitalized-word regex if spaCy finds nothing.
    """
    doc = nlp(claim_text)
    ents = [(e.start_char, e.text) for e in doc.ents if e.label_ in RELEVANT_ENTITY_LABELS]

    if not ents:
        ents = [
            (m.start(), m.group())
            for m in re.finditer(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b', claim_text)
            if len(m.group()) > 1
        ]

    ents.sort()  # by character position in the claim

    seen, ordered = set(), []
    for _, text in ents:
        if text not in seen:
            seen.add(text)
            ordered.append(text)
    return ordered


def collect_candidates(claim_text):
    """
    Pools candidate article titles from each of the first few entities,
    so the title picker sees pages for every entity, not just one.
    """
    entities = extract_entities_fast(claim_text)

    candidates = []
    for entity in entities[:MAX_ENTITIES]:
        for title in opensearch_titles(entity, limit=CANDIDATES_PER_ENTITY):
            if title not in candidates:
                candidates.append(title)

    if not candidates:
        candidates = search_wikipedia_titles(claim_text, limit=6)

    return entities, candidates


def pick_best_title(claim_text, candidate_titles):
    """Uses the LLM to pick the candidate article best suited to check the claim."""
    if len(candidate_titles) == 1:
        return candidate_titles[0]

    titles_list = "\n".join(f"- {t}" for t in candidate_titles)
    prompt = f"""Given this claim: "{claim_text}"

Which of these Wikipedia article titles is about the MAIN SUBJECT/ENTITY of the claim
(the specific person, place, or thing the claim is directly about) —
not just a related or broader topic?

Respond with ONLY the exact title, nothing else, no quotes or punctuation.

Candidates:
{titles_list}"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    chosen = response.choices[0].message.content.strip().strip('"').strip("'").strip(".")

    for title in candidate_titles:
        if chosen.lower() == title.lower():
            return title

    return candidate_titles[0]  # fallback: first candidate (the first entity's top result)


def get_best_matching_page(claim_text):
    _, candidates = collect_candidates(claim_text)
    if not candidates:
        return None

    best_title = pick_best_title(claim_text, candidates)
    return get_wikipedia_page(best_title)


if __name__ == "__main__":
    RUN_PICKER = False  # set True to also call the LLM title picker (uses Groq quota)

    test_claims = [
        "BTS has an album called Deadline in their discography",
        "einstein failed math in school",
        "great wall of china can be seen from space",
        "The Eiffel Tower is located in Paris",
    ]

    for c in test_claims:
        entities, candidates = collect_candidates(c)
        print(f"\nClaim: {c}")
        print(f"  Entities (claim order): {entities}")
        print(f"  Candidate pool: {candidates}")
        if RUN_PICKER:
            print(f"  Picked: {pick_best_title(c, candidates) if candidates else None}")