# Write-up: Document Q&A tool (GDG-USAR Student Handbook)

**Project link:** https://github.com/Akhil-Mehta/gdg-project-document-assistance

## What I built
A command-line tool that answers questions about a PDF using only the document's own text, and shows which page and passage each answer came from. It runs on the GDG-USAR Student Handbook sample document.

## How it works
1. **Extract:** `pypdf` reads the PDF page by page, so every piece of text keeps its page number.
2. **Chunk:** LangChain's recursive splitter cuts each page into overlapping chunks.
3. **Embed and retrieve:** each chunk is turned into a vector with `all-MiniLM-L6-v2` (through the Hugging Face API), and the top matches for a question are found by cosine similarity.
4. **Guard:** if the best match is below a similarity threshold (0.25), the tool refuses without calling the LLM.
5. **Generate:** a Hugging Face chat model gets only the retrieved passages and is told to answer from them alone, or say "I can't find that in the document."
6. **Show sources:** the answer prints the passages it cited, with page numbers and scores.

## Experiment: two settings
| Setting | Chunks | Retrieval hit-rate | Unanswerable questions refused |
|---|---|---|---|
| A: 400 chars, overlap 50, top-3 | 18 | 5/5 | 2/2 |
| B: 1000 chars, overlap 200, top-5 | 8 | 5/5 | 2/2 |

- Both found the right page for all five answerable questions, but B's top-5 covers most of an 8-chunk document, so its score is easier to get. A's result is the more meaningful one.
- Smaller chunks gave higher similarity for the relevant passage (for example 0.46 vs 0.39 for the help desk hours), because large chunks mix several topics. I chose **A**.
- **Key finding:** the similarity threshold only blocks off-topic questions ("capital of France" scored 0.05). A question that is on-topic but not answered in the handbook ("date and venue of the next event") scored 0.38 to 0.48, higher than some answerable questions, so refusing it depended on the prompt, not the score.

## Tests
Seven questions: five answerable (help desk hours, how to contact the desk, where to report learning-portal issues, what to document when AI tools are used, weekend opening) and two unanswerable (next event date and venue, capital of France). All five were answered correctly with the right page, and both unanswerable ones were refused. Full output is in `DECISIONS.md`.

## Bonus features
- **Conversation history:** follow-ups like "and on weekends?" are rewritten into standalone questions before retrieval.
- **Document summary:** `python docqa.py summary doc.pdf`.
- **Multiple documents:** the index accepts a list of files.

## Challenges and decisions
- Windows blocked PyTorch, so I moved embeddings to the Hugging Face API.
- My network sometimes reset connections, so I added automatic retries and an embedding cache.
- The "help desk" question confused the model because the handbook has two desks (Student Support Desk and IT help desk); this is a question ambiguity, not a retrieval failure.

## Limitations
Small document and test set; threshold set by hand; scanned PDFs would need OCR; the model occasionally cites a loosely related passage alongside the right one.
