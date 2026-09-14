import re

STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "are", "be", "been", "being",
    "in", "on", "at", "by", "for", "with", "to", "of", "and", "or", "but",
    "this", "that", "these", "those", "it", "its", "as", "from", "has", "have",
    "had", "not", "no", "do", "does", "did", "so", "than", "then",
}


def normalize_text(text):
    """Lowercase, collapse whitespace, strip punctuation for comparison."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


def meaningful_words(text):
    """Extract words excluding common stopwords, so overlap reflects real content, not filler."""
    words = normalize_text(text).split()
    return set(w for w in words if w not in STOPWORDS and len(w) > 1)


def verify_citation(supporting_quote, evidence_chunks, min_overlap_ratio=0.75):
    """
    Checks whether the supporting_quote is substantively present in the retrieved evidence.
    Uses normalized substring matching first, then falls back to meaningful-word overlap
    (excluding stopwords) so common filler words don't inflate the match score.
    """
    if not supporting_quote or not supporting_quote.strip():
        return False, "No supporting quote provided."

    normalized_quote = normalize_text(supporting_quote)
    combined_evidence = normalize_text(" ".join(c["text"] for c in evidence_chunks))

    # exact (normalized) substring match — the strongest case
    if normalized_quote in combined_evidence:
        return True, "Quote found verbatim (after normalization) in retrieved evidence."

    # fallback: meaningful-word overlap only (ignores common filler words)
    quote_words = meaningful_words(supporting_quote)
    evidence_words = meaningful_words(combined_evidence)

    if not quote_words:
        return False, "No meaningful content words found in quote."

    overlap = len(quote_words & evidence_words) / len(quote_words)
    missing_words = quote_words - evidence_words

    if overlap >= min_overlap_ratio:
        return True, f"Quote's key content substantially matches evidence ({overlap:.0%} meaningful-word overlap)."
    else:
        return False, (
            f"Quote's key content does not sufficiently match evidence "
            f"({overlap:.0%} meaningful-word overlap). Unmatched terms: {missing_words} — possible hallucination."
        )


if __name__ == "__main__":
    from wiki_search import get_best_matching_page
    from retrieval import get_embedding_model, chunk_text, embed_and_index, retrieve_relevant_chunks
    from verdict import generate_verdict

    claim = "The Eiffel Tower is located in Paris"

    page = get_best_matching_page(claim)
    chunks = chunk_text(page.text)
    model = get_embedding_model()
    index = embed_and_index(model, chunks)
    retrieved = retrieve_relevant_chunks(model, index, chunks, claim, top_k=3)

    verdict = generate_verdict(claim, retrieved)
    print(f"Verdict: {verdict['verdict']}")
    print(f"Supporting quote: {verdict['supporting_quote']}")

    is_valid, reason = verify_citation(verdict["supporting_quote"], retrieved)
    print(f"\nCitation verified: {is_valid}")
    print(f"Reason: {reason}")

    # test with a deliberately fabricated quote
    fake_quote = "The Eiffel Tower was built by aliens in 1500."
    is_valid_fake, reason_fake = verify_citation(fake_quote, retrieved)
    print(f"\nFake quote verified: {is_valid_fake}")
    print(f"Reason: {reason_fake}")