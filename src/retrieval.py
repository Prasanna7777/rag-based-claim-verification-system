from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import torch


def get_embedding_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model on: {device}")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    return model


def chunk_text(text, min_chunk_size=100, max_chunk_size=800):
    """
    Splits text into paragraphs (Wikipedia articles use double newlines between paragraphs).
    Very short paragraphs get merged with the next one; very long paragraphs get
    split further using a simple fallback.
    """
    raw_paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks = []
    buffer = ""

    for para in raw_paragraphs:
        candidate = (buffer + " " + para).strip() if buffer else para

        if len(candidate) <= max_chunk_size:
            buffer = candidate
        else:
            if buffer:
                chunks.append(buffer)
            if len(para) > max_chunk_size:
                for i in range(0, len(para), max_chunk_size):
                    chunks.append(para[i:i + max_chunk_size])
                buffer = ""
            else:
                buffer = para

    if buffer and len(buffer) >= min_chunk_size:
        chunks.append(buffer)
    elif buffer and chunks:
        chunks[-1] += " " + buffer

    return chunks


def flatten_sections(page, min_section_length=50):
    """
    Recursively flattens a Wikipedia page's sections (including subsections)
    into a flat list of {title, text} blocks. Includes the intro/summary
    as an implicit first section since wikipedia-api excludes it from .sections.
    """
    blocks = []

    if page.summary and len(page.summary.strip()) >= min_section_length:
        blocks.append({"title": "Introduction", "text": page.summary.strip()})

    def _walk(sections, parent_title=""):
        for section in sections:
            title = f"{parent_title} - {section.title}" if parent_title else section.title
            text = section.text.strip()
            if len(text) >= min_section_length:
                blocks.append({"title": title, "text": text})
            if section.sections:
                _walk(section.sections, title)

    _walk(page.sections)
    return blocks


def select_relevant_sections(model, claim, sections, top_n=2):
    """
    Embeds each section (title + text) and returns the top_n sections
    most relevant to the claim.
    """
    if not sections:
        return []

    section_texts = [f"{s['title']}. {s['text']}" for s in sections]
    section_embeddings = model.encode(section_texts, convert_to_numpy=True, normalize_embeddings=True)
    claim_embedding = model.encode([claim], convert_to_numpy=True, normalize_embeddings=True)

    scores = np.dot(section_embeddings, claim_embedding.T).flatten()
    top_indices = np.argsort(scores)[::-1][:top_n]

    return [sections[i] for i in top_indices]


def embed_and_index(model, chunks):
    embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def retrieve_relevant_chunks(model, index, chunks, query, top_k=3):
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({"text": chunks[idx], "score": float(score)})
    return results


def retrieve_for_claim(model, page, claim, top_n_sections=2, top_k_chunks=3):
    """
    Full section-aware retrieval pipeline for a single claim:
    1. Flatten the article into sections
    2. Pick the most relevant section(s) for the claim
    3. Chunk only those sections
    4. Retrieve the top_k most relevant chunks from within them
    """
    sections = flatten_sections(page)
    if not sections:
        return []

    relevant_sections = select_relevant_sections(model, claim, sections, top_n=top_n_sections)
    combined_text = "\n\n".join(s["text"] for s in relevant_sections)

    chunks = chunk_text(combined_text)
    if not chunks:
        return []

    index = embed_and_index(model, chunks)
    return retrieve_relevant_chunks(model, index, chunks, claim, top_k=top_k_chunks)


if __name__ == "__main__":
    from wiki_search import get_best_matching_page

    claim = "BTS is still on the Arirang tour as of now"
    page = get_best_matching_page(claim)
    print(f"Retrieved article: {page.title}")

    model = get_embedding_model()

    sections = flatten_sections(page)
    print(f"Total sections found: {len(sections)}")

    relevant_sections = select_relevant_sections(model, claim, sections, top_n=2)
    print(f"\nSelected sections:")
    for s in relevant_sections:
        print(f"  - {s['title']} ({len(s['text'])} chars)")

    results = retrieve_for_claim(model, page, claim, top_n_sections=2, top_k_chunks=3)

    print(f"\n--- Top 3 relevant passages ---")
    for i, r in enumerate(results):
        print(f"\nRank {i+1} (score: {r['score']:.4f}):")
        print(r["text"][:300])