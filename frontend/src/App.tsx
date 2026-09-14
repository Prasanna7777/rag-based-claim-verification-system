import { useState } from "react";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";

type Verdict = "TRUE" | "FALSE" | "MISLEADING" | "UNKNOWN";

interface EvidenceItem {
  text: string;
  score: number;
}

interface ClaimResult {
  claim: string;
  verdict: Verdict;
  explanation: string;
  supporting_quote: string;
  citation_verified: boolean;
  source_title: string | null;
  evidence: EvidenceItem[];
}

interface VerifyResponse {
  submission: string;
  results: ClaimResult[];
}

const verdictConfig: Record<Verdict, { label: string; icon: string }> = {
  TRUE: { label: "Verified", icon: "✓" },
  FALSE: { label: "False", icon: "✕" },
  MISLEADING: { label: "Misleading", icon: "!" },
  UNKNOWN: { label: "Unverifiable", icon: "?" },
};

function App() {
  const [input, setInput] = useState("");
  const [response, setResponse] = useState<VerifyResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleVerify = async () => {
    if (!input.trim()) return;
    setLoading(true);
    setError(null);
    setResponse(null);

    try {
      const res = await fetch(`${API_BASE}/verify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: input, top_k: 3 }),
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => null);
        const message = errorData?.detail || "Request failed. Please try again.";
        throw new Error(message);
      }

      const data: VerifyResponse = await res.json();
      setResponse(data);
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : "Could not reach the verification service.");
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      handleVerify();
    }
  };

  const trueCount = response?.results.filter((r) => r.verdict === "TRUE").length ?? 0;
  const totalCount = response?.results.length ?? 0;

  return (
    <div className="app-shell">
      <header className="page-header">
        <h1>Claim verification</h1>
        <p>Retrieves evidence, generates a verdict, and verifies its own citation.</p>
      </header>

      <main className="container">
        <section className="intake-card">
          <label htmlFor="claim-input">Enter one or more claims to check</label>
          <textarea
            id="claim-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="e.g. The Eiffel Tower is in Paris, and it was completed in 1889."
            rows={3}
          />
          <div className="intake-footer">
            <span className="intake-hint">Ctrl/Cmd + Enter to submit</span>
            <button onClick={handleVerify} disabled={loading || !input.trim()}>
              {loading ? "Checking…" : "Check claims"}
            </button>
          </div>
          {error && <p className="intake-error">{error}</p>}
        </section>

        {loading && (
          <div className="status-line">
            <span className="spinner" />
            Splitting claims and searching sources…
          </div>
        )}

        {response && (
          <section className="results">
            <p className="submission-echo">{response.submission}</p>

            {totalCount > 1 && (
              <p className="summary-line">
                {trueCount} of {totalCount} claims verified true
              </p>
            )}

            <div className="claim-list">
              {response.results.map((r, i) => {
                const config = verdictConfig[r.verdict];
                return (
                  <article key={i} className="claim-card">
                    <div className="claim-card-header">
                      <span className={`badge badge-${r.verdict.toLowerCase()}`}>
                        <span className="badge-icon">{config.icon}</span>
                        {config.label}
                      </span>
                      {r.source_title && (
                        <a
                          className="source-link"
                          href={`https://en.wikipedia.org/wiki/${encodeURIComponent(
                            r.source_title.replace(/ /g, "_")
                          )}`}
                          target="_blank"
                          rel="noreferrer"
                        >
                          {r.source_title}
                        </a>
                      )}
                    </div>

                    <p className="claim-text">{r.claim}</p>
                    <p className="explanation">{r.explanation}</p>

                    {r.supporting_quote && (
                      <div className="quote-block">
                        <span className="quote-label">
                          {r.citation_verified ? "verified quote" : "unverified quote"}
                        </span>
                        <p className="quote-text">{r.supporting_quote}</p>
                      </div>
                    )}
                  </article>
                );
              })}
            </div>
          </section>
        )}

        {!response && !loading && (
          <div className="empty-state">
            <p>No claims checked yet.</p>
            <p className="empty-sub">
              Enter a single claim or several — each is checked independently against its own sources.
            </p>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;