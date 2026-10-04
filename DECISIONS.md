# DECISIONS

## Pipeline
PDF -> pypdf per-page text -> RecursiveCharacterTextSplitter (page number kept on every chunk)
-> all-MiniLM-L6-v2 embeddings (normalised, via the Hugging Face Inference API) -> cosine top-k
-> similarity threshold -> LLM (Hugging Face chat model) with a "context only" system prompt
-> answer + the page/passage that the answer cited.

Document used: GDG-USAR Student Handbook (2 pages, sample document for this task).

## Design choices and why
- **Embeddings via HF API, not local.** `sentence-transformers` needs PyTorch, and Windows Application Control blocked the PyTorch DLLs on my machine. The same model is called through the Hugging Face API instead (`EMBED_BACKEND=local` still works on machines without that restriction).
- **Vector store:** a plain NumPy matrix. With 8-18 chunks a vector database adds nothing; Chroma/FAISS would make sense for many or large documents.
- **Page metadata on every chunk**, so every answer can cite a page.
- **Embedding cache:** vectors are saved to disk per document and settings, which avoids repeat API calls (my network sometimes drops connections, and the API has a free quota).
- **Retries with backoff** on network errors only; auth/permission/model errors fail immediately.
- **Three grounding guards:** (1) the prompt allows only the retrieved passages, (2) a fixed refusal sentence, (3) a similarity threshold (0.25) that skips the LLM entirely.
- **Source display:** only passages the answer actually cited are shown as sources; other retrieved passages are listed separately as "not cited".

## Experiment: two chunking/retrieval settings
7 questions: 5 answerable, 2 not answerable (one on-topic but not covered, one off-topic).

| Setting | Chunks | Retrieval hit-rate | Unanswerable handled |
|---|---|---|---|
| A: size 400, overlap 50, k=3 | 18 | 5/5 | 2/2 refused |
| B: size 1000, overlap 200, k=5 | 8 | 5/5 | 2/2 refused |

Top similarity score per question (A vs B):

| Question | A | B |
|---|---|---|
| Help desk opening hours | 0.46 | 0.39 |
| How to contact the help desk | 0.66 | 0.63 |
| Where to report learning-portal issues | 0.71 | 0.65 |
| What to explain when AI tools are used | 0.56 | 0.55 |
| Open on weekends? | 0.51 | 0.41 |
| Next event date/venue (not in document) | 0.38 | 0.48 |
| Capital of France (off-topic) | 0.05 | 0.00 |

### What I observed
1. **Hit-rate did not separate the settings.** Both retrieved the correct page for 5/5 answerable questions. This is a weak comparison for B: the document only has 8 chunks there and k=5, so B retrieves more than half the whole document for every question, which makes a hit almost guaranteed. A's 5/5 (3 of 18 chunks) is the more meaningful result.
2. **Smaller chunks gave higher similarity for the relevant passage** on 5 of 5 answerable questions (for example 0.46 vs 0.39 for opening hours, 0.51 vs 0.41 for weekends). Large chunks mix several topics (help desk, events, AI use), which dilutes the match.
3. **Answer quality was similar.** Both gave the correct hours, weekend status, report location and AI-usage rule.
4. **Ambiguous question confused both.** "How can students contact the help desk?" mixes up two different desks in the handbook (the Student Support Desk and the IT help desk). A blended them; B said the IT help desk's contact details were not provided and then listed the Student Support Desk's. This comes from ambiguity in the question and document, not a retrieval failure, but B's larger context made the answer wordier and more confusing.
5. **The similarity threshold only catches off-topic questions.** The off-topic question scored 0.05/0.00 and was refused by the threshold before the LLM was called. The on-topic-but-unanswerable question (next event date/venue) scored **0.38 in A and 0.48 in B, higher than some answerable questions** (B: 0.39 for opening hours). So a score cutoff cannot detect it; the refusal came from the "context only" prompt. Score alone is not a reliable answerability signal; the prompt-level guard is essential.

### Final choice
**Setting A (400/50/k=3)**: focused chunks, higher similarity on relevant passages, a shorter and cleaner prompt context, and a hit-rate that is meaningful at k=3 of 18. The trade-off is that very small chunks can split related sentences, which did not affect these questions.

## Test questions
See the output of `python evaluate.py doc.pdf`. Summary: 5 answerable questions were all answered correctly with correct page retrieval in both settings; both unanswerable questions returned "I can't find that in the document." in both settings.

## Limitations
- Tiny document (2 pages), so differences between settings are small; I would expect a larger gap on a longer document.
- Only 7 questions, all written by me, so hit-rate is a rough indicator rather than a benchmark.
- Threshold (0.25) was set by hand; it handles off-topic questions but not topical ones that the document does not answer.
- Scanned PDFs without a text layer would extract as empty pages (OCR would be needed).
- Embeddings and the LLM depend on the Hugging Face API (free-tier quota, occasional network resets).

## Bonus features (tested)
- Conversation history: `python docqa.py chat doc.pdf`
- Short document summary: `python docqa.py summary doc.pdf`

### Conversation history: what I tested
Session: "hi" -> refused (top score 0.20, below the threshold). "what are the help desk opening hours?" -> Monday to Friday, 10:00 AM to 4:00 PM (p.1). Follow-up "and on weekends?" -> correctly answered that the desk is closed on weekends.

**Decision:** retrieval uses only the current message, so a bare follow-up like "and on weekends?" has little to search with. Before retrieving, the LLM rewrites follow-ups into a standalone question using the last few turns. Cost: one extra short LLM call per follow-up.

**Observed weakness:** the answer sometimes cites a loosely related passage as well (the intro chunk scored 0.52 for the weekends follow-up and was cited next to the correct one). The facts were still right, but citations are slightly generous.

### Short summary: what I tested
`python docqa.py summary doc.pdf` produced a five-sentence summary of the handbook (support desk, club activities, workshops, submission rules, conduct), sampling chunks evenly across the document.
