from wiki_search import get_best_matching_page
from retrieval import get_embedding_model, chunk_text, embed_and_index, retrieve_relevant_chunks
from verdict import generate_verdict
from citation_check import verify_citation
from claim_splitter import split_into_claims


class ClaimVerificationPipeline:
    def __init__(self):
        print("Loading embedding model...")
        self.embed_model = get_embedding_model()

    def verify_single_claim(self, claim, top_k=3):
        page = get_best_matching_page(claim)

        if page is None:
            return {
                "claim": claim,
                "verdict": "UNKNOWN",
                "explanation": "Could not find a relevant Wikipedia source for this claim.",
                "supporting_quote": "",
                "citation_verified": False,
                "source_title": None,
                "evidence": [],
            }

        chunks = chunk_text(page.text)
        if len(chunks) == 0:
            return {
                "claim": claim,
                "verdict": "UNKNOWN",
                "explanation": "The matched source page had no usable content.",
                "supporting_quote": "",
                "citation_verified": False,
                "source_title": page.title,
                "evidence": [],
            }

        index = embed_and_index(self.embed_model, chunks)
        retrieved = retrieve_relevant_chunks(self.embed_model, index, chunks, claim, top_k=top_k)

        verdict = generate_verdict(claim, retrieved)
        is_verified, reason = verify_citation(verdict["supporting_quote"], retrieved)

        # if citation can't be verified, don't let the system overstate confidence
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
        results = [self.verify_single_claim(c, top_k=top_k) for c in claims]
        return results


if __name__ == "__main__":
    pipeline = ClaimVerificationPipeline()

    test_inputs = [
        "The Eiffel Tower is in Paris",
        "The Eiffel Tower is in Paris, and Albert Einstein was born in Germany",
        "The Eiffel Tower was built by aliens in 1500",
    ]

    for text in test_inputs:
        print(f"\n{'='*60}")
        print(f"Input: {text}")
        print('='*60)

        results = pipeline.verify(text)

        for r in results:
            print(f"\nClaim: {r['claim']}")
            print(f"Verdict: {r['verdict']}")
            print(f"Explanation: {r['explanation']}")
            print(f"Citation verified: {r['citation_verified']}")
            print(f"Source: {r['source_title']}")