from wiki_search import get_best_matching_page
from retrieval import (
    get_embedding_model,
    chunk_text,
    embed_and_index,
    retrieve_relevant_chunks,
    retrieve_for_claim,
)
from verdict import generate_verdict
from citation_check import verify_citation
from claim_splitter import split_into_claims


class ClaimVerificationPipeline:
    def __init__(self):
        print("Loading embedding model...")
        self.embed_model = get_embedding_model()

    def _unknown(self, claim, explanation, source_title=None):
        return {
            "claim": claim,
            "verdict": "UNKNOWN",
            "explanation": explanation,
            "supporting_quote": "",
            "citation_verified": False,
            "source_title": source_title,
            "evidence": [],
        }

    def _flat_retrieve(self, page, claim, top_k):
        """Fallback: chunk the whole article, embed, FAISS top-k."""
        chunks = chunk_text(page.text)
        if not chunks:
            return []
        index = embed_and_index(self.embed_model, chunks)
        return retrieve_relevant_chunks(self.embed_model, index, chunks, claim, top_k=top_k)

    def verify_single_claim(self, claim, top_k=3):
        page = get_best_matching_page(claim)
        if page is None:
            return self._unknown(claim, "Could not find a relevant Wikipedia source for this claim.")

        # Section-aware retrieval; fall back to whole-article search if the page has no usable sections
        retrieved = retrieve_for_claim(
            self.embed_model, page, claim, top_n_sections=2, top_k_chunks=top_k
        )
        if not retrieved:
            retrieved = self._flat_retrieve(page, claim, top_k)
        if not retrieved:
            return self._unknown(claim, "The matched source page had no usable content.", page.title)

        verdict = generate_verdict(claim, retrieved)
        is_verified, _ = verify_citation(verdict["supporting_quote"], retrieved)

        final_verdict = verdict["verdict"]
        explanation = verdict["explanation"]
        if not is_verified and final_verdict != "UNKNOWN":
            final_verdict = "UNKNOWN"
            explanation += " (Note: original verdict downgraded — supporting citation could not be verified.)"

        return {
            "claim": claim,
            "verdict": final_verdict,
            "explanation": explanation,
            "supporting_quote": verdict["supporting_quote"],
            "citation_verified": is_verified,
            "source_title": page.title,
            "evidence": [{"text": r["text"], "score": r["score"]} for r in retrieved],
        }

    def verify(self, text, top_k=3):
        claims = split_into_claims(text)
        return [self.verify_single_claim(c, top_k=top_k) for c in claims]