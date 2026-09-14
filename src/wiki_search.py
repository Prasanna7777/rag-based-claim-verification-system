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

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_trf")
RELEVANT_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "FAC", "WORK_OF_ART", "EVENT", "NORP", "PRODUCT"}


def search_wikipedia_titles(query, limit=6):
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
    Extracts named entities from the claim using a local NER model (spaCy),
    which recognizes entities based on context/grammar rather than
    capitalization alone -- so it works even for casually-typed, lowercase
    claims (e.g. "great wall of china" still resolves correctly).
    Falls back to a capitalized-word regex only if spaCy finds nothing.
    """
    doc = nlp(claim_text)
    entities = [ent.text for ent in doc.ents if ent.label_ in RELEVANT_ENTITY_LABELS]

    seen = set()
    unique_entities = []
    for e in entities:
        if e not in seen:
            seen.add(e)
            unique_entities.append(e)

    if unique_entities:
        return sorted(unique_entities, key=len, reverse=True)

    matches = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b', claim_text)
    seen = set()
    fallback = []
    for m in matches:
        if m not in seen and len(m) > 1:
            seen.add(m)
            fallback.append(m)
    return sorted(fallback, key=len, reverse=True)


def pick_best_title(claim_text, candidate_titles):
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

    return candidate_titles[0]


def get_best_matching_page(claim_text):
    """
    1. Extract the likely entity using local NER (casing-independent).
    2. Search for that entity using opensearch (precise title matching).
    3. If nothing found, fall back to full-text search on the whole claim.
    4. Let the LLM pick the single best title from the candidates.
    """
    entities = extract_entities_fast(claim_text)

    candidates = []
    if entities:
        candidates = opensearch_titles(entities[0], limit=6)

    if not candidates:
        candidates = search_wikipedia_titles(claim_text, limit=6)

    if not candidates:
        return None

    best_title = pick_best_title(claim_text, candidates)
    return get_wikipedia_page(best_title)


if __name__ == "__main__":
    import time

    test_claims = [
        "great wall of china can be seen from space",
        "einstein failed math in school",
        "current prime minister of india is jaya prakash",
        "The Eiffel Tower is located in Paris",
    ]

    for c in test_claims:
        start = time.time()
        entities = extract_entities_fast(c)
        page = get_best_matching_page(c)
        elapsed = time.time() - start
        print(f"Claim: '{c}'")
        print(f"  Entities: {entities}")
        print(f"  -> {page.title if page else 'Not found'} ({elapsed:.2f}s)\n")