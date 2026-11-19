from wiki_search import get_wikipedia_page
from pipeline import ClaimVerificationPipeline

claim = "Napoleon Bonaparte was extremely short for his era"
p = ClaimVerificationPipeline()
page = get_wikipedia_page("Napoleon")

for i, r in enumerate(p._flat_retrieve(page, claim, 10), 1):
    mark = "  <-- Farington" if "Farington" in r["text"] else ""
    print(f"{i}. {r['score']:.3f}  {r['text'][:80]!r}{mark}")