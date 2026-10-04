"""Document Q&A (RAG) over one PDF/text file.

Pipeline: load pages -> chunk (keeping page numbers) -> embed -> retrieve top-k
-> refuse if best score is below a threshold -> generate answer ONLY from
retrieved chunks -> print answer + source pages/passages.

Usage:
  python docqa.py ask   doc.pdf "What is X?"
  python docqa.py chat  doc.pdf            # conversation history (bonus)
  python docqa.py summary doc.pdf          # short summary (bonus)
Env: HF_TOKEN; LLM_BACKEND=hf|local|anthropic (default hf);
     EMBED_BACKEND=api|local (default api, avoids torch).
"""
import os, sys, time, hashlib
from dataclasses import dataclass
import numpy as np
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")        # anthropic backend
HF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")      # hf backend (API)
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")  # local backend
_embedder = None


_hf_client = None


def hf_client():
    """One shared client = fewer new TLS connections (helps flaky networks)."""
    global _hf_client
    if _hf_client is None:
        from huggingface_hub import InferenceClient
        _hf_client = InferenceClient(token=os.getenv("HF_TOKEN"), timeout=120)
    return _hf_client


_embed_client = None


def embed_client():
    """Explicit provider = no extra 'which provider?' network lookup before every call."""
    global _embed_client
    if _embed_client is None:
        from huggingface_hub import InferenceClient
        _embed_client = InferenceClient(provider=os.getenv("HF_EMBED_PROVIDER", "hf-inference"),
                                        token=os.getenv("HF_TOKEN"), timeout=120)
    return _embed_client


def with_retry(fn, tries=8):
    """Retry on network errors (connection reset etc.) with growing waits."""
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            # don't retry real config errors: bad token / permission / bad model
            if any(code in msg for code in ("401", "403", "400", "404")) and "10054" not in msg:
                raise
            if i == tries - 1:
                raise
            wait = min(2 ** i, 10)
            print(f"  network hiccup ({type(e).__name__}); retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)


def embed(texts):
    """Return L2-normalised embeddings, shape (n, dim).
    EMBED_BACKEND=api (default): Hugging Face API, no torch needed (HF_TOKEN required).
    EMBED_BACKEND=local: sentence-transformers on your machine (needs torch)."""
    global _embedder
    if os.getenv("EMBED_BACKEND", "api") == "local":
        if _embedder is None:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer(EMBED_MODEL)
        return _embedder.encode(texts, normalize_embeddings=True)

    client = embed_client()
    out = []
    for i in range(0, len(texts), 16):  # small batches
        batch = texts[i:i + 16]
        res = np.asarray(with_retry(lambda: client.feature_extraction(batch, model=EMBED_MODEL)),
                         dtype=np.float32)
        if res.ndim == 1:               # single vector came back
            res = res[None, :]
        if res.ndim == 3:               # token-level output -> mean pool
            res = res.mean(axis=1)
        out.append(res)
    vecs = np.vstack(out)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


@dataclass
class Chunk:
    text: str
    page: int   # 1-based page number
    source: str # file name


def load_pages(path):
    """Return list of (page_number, text). Works for .pdf and .txt."""
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        return [(i + 1, (p.extract_text() or "").strip()) for i, p in enumerate(reader.pages)]
    with open(path, encoding="utf-8") as f:
        text = f.read()
    # treat ~3000-char blocks as "pages" so text files still get a locator
    return [(i // 3000 + 1, text[i:i + 3000]) for i in range(0, len(text), 3000)]


def chunk_pages(pages, source, chunk_size=800, overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""])
    chunks = []
    for page, text in pages:
        if not text:
            continue  # blank/scanned page: nothing to index (note in DECISIONS.md)
        for piece in splitter.split_text(text):
            chunks.append(Chunk(piece, page, source))
    return chunks


class DocIndex:
    def __init__(self, paths, chunk_size=800, overlap=150):
        self.chunks = []
        for p in ([paths] if isinstance(paths, str) else paths):  # multi-doc bonus
            self.chunks += chunk_pages(load_pages(p), os.path.basename(p), chunk_size, overlap)
        texts = [c.text for c in self.chunks]
        key = hashlib.md5((EMBED_MODEL + "||".join(texts)).encode()).hexdigest()[:16]
        cache = f".emb_cache_{key}.npy"   # same doc + settings = no new API calls
        if os.path.exists(cache):
            self.vecs = np.load(cache)
        else:
            self.vecs = embed(texts)
            np.save(cache, self.vecs)

    def retrieve(self, query, k=4):
        q = embed([query])[0]
        scores = self.vecs @ q  # cosine similarity (vectors are normalised)
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]


SYSTEM = (
    "You answer questions using ONLY the numbered context passages provided. "
    "If the passages do not contain the answer, reply exactly: "
    "'I can't find that in the document.' Do not use outside knowledge. "
    "Cite passages like [1], [2] after the claims they support."
)


_local_pipe = None


def llm(system, user, history=None, max_tokens=700):
    """LLM_BACKEND=hf (default): Hugging Face Inference API, needs HF_TOKEN.
       LLM_BACKEND=local: small model run on your own machine, no key needed.
       LLM_BACKEND=anthropic: Anthropic API, needs ANTHROPIC_API_KEY."""
    backend = os.getenv("LLM_BACKEND", "hf")
    msgs = [{"role": "system", "content": system}] + (history or []) + \
           [{"role": "user", "content": user}]

    if backend == "hf":
        client = hf_client()
        r = with_retry(lambda: client.chat_completion(
            messages=msgs, model=HF_MODEL, max_tokens=max_tokens, temperature=0.1))
        return r.choices[0].message.content.strip()

    if backend == "local":
        global _local_pipe
        if _local_pipe is None:
            from transformers import pipeline
            _local_pipe = pipeline("text-generation", model=LOCAL_MODEL)
        out = _local_pipe(msgs, max_new_tokens=max_tokens, do_sample=False)
        return out[0]["generated_text"][-1]["content"].strip()

    import anthropic
    r = anthropic.Anthropic().messages.create(
        model=LLM_MODEL, max_tokens=max_tokens, system=system, messages=msgs[1:])
    return r.content[0].text.strip()


NO_ANSWER = "I can't find that in the document."


def condense(question, history):
    """Turn a follow-up ('and on weekends?') into a standalone search query using chat history."""
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    return llm("Rewrite the user's last question as one standalone question that makes sense "
               "without the conversation. Output ONLY the question.",
               f"Conversation:\n{convo}\n\nLast question: {question}", max_tokens=80)


def answer(index, question, k=4, min_score=0.25, history=None):
    search_q = condense(question, history) if history else question
    hits = index.retrieve(search_q, k)
    if hits[0][1] < min_score:  # retrieval-level guard against hallucination
        return NO_ANSWER, hits
    ctx = "\n\n".join(f"[{i+1}] (p.{c.page}, {c.source})\n{c.text}"
                      for i, (c, _) in enumerate(hits))
    # history carries only past Q/A text; context is re-retrieved every turn
    out = llm(SYSTEM, f"Context:\n{ctx}\n\nQuestion: {question}", history)
    return out, hits


def show(ans, hits):
    import re
    print("\nANSWER:", ans)
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", ans) if 0 < int(n) <= len(hits)})
    if ans.strip().startswith(NO_ANSWER[:20]):
        print("\n(No answer given: the document does not contain this information.)")
        print("Closest passages retrieved (NOT used for an answer):")
        cited = []
    else:
        print("\nSOURCES USED:")
        for n in cited:
            c, sc = hits[n - 1]
            print(f"  [{n}] {c.source} p.{c.page}  score={sc:.2f}\n      {c.text[:300].replace(chr(10), ' ')}")
        print("\nOther passages retrieved (not cited):")
    for i, (c, sc) in enumerate(hits, 1):
        if i not in cited:
            print(f"  [{i}] {c.source} p.{c.page}  score={sc:.2f}  {c.text[:80].replace(chr(10), ' ')}...")


def summarize(index, max_chunks=40):
    step = max(1, len(index.chunks) // max_chunks)  # evenly sample the doc
    text = "\n\n".join(c.text for c in index.chunks[::step][:max_chunks])
    return llm("Summarize ONLY from the text given, in 5 sentences.", text)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    mode, path = sys.argv[1], sys.argv[2]
    idx = DocIndex(path)
    if mode == "ask":
        a, h = answer(idx, " ".join(sys.argv[3:])); show(a, h)
    elif mode == "summary":
        print(summarize(idx))
    elif mode == "chat":
        hist = []
        while (q := input("\nYou> ").strip()) not in ("", "quit"):
            a, h = answer(idx, q, history=hist); show(a, h)
            hist += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
