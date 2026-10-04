# Document Q&A (RAG) - GDG-USAR Student Handbook

Answers questions about a PDF/text document using only retrieved passages, and shows the page and passage used.

## Setup
```
python -m venv venv
venv\Scripts\activate            # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```
Create a Hugging Face token (Settings -> Access Tokens -> Fine-grained, tick
"Make calls to Inference Providers"), then set it in your terminal (never commit it):
```
set HF_TOKEN=hf_...              # PowerShell: $env:HF_TOKEN="hf_..."   Mac/Linux: export HF_TOKEN=hf_...
set HF_MODEL=<a chat model id available to your account>
```
List available chat models: `GET https://router.huggingface.co/v1/models` (with your token).

## Usage
```
python docqa.py ask doc.pdf "What are the opening hours of the help desk?"
python docqa.py chat doc.pdf         # conversation history; blank line to quit
python docqa.py summary doc.pdf      # short document summary
python evaluate.py doc.pdf           # 7 test questions under two chunking settings
```

## Environment variables
| Variable | Default | Meaning |
|---|---|---|
| `HF_TOKEN` | - | Hugging Face token (required for the default API backends) |
| `HF_MODEL` | Qwen/Qwen2.5-7B-Instruct | Chat model for answers (must be available to your account) |
| `LLM_BACKEND` | `hf` | `hf`, `local` (needs torch) or `anthropic` |
| `EMBED_BACKEND` | `api` | `api` (no torch) or `local` (sentence-transformers) |

## Features
- Page-aware chunking, embeddings, cosine top-k retrieval, similarity threshold, "context only" prompt
- Sources: shows only the passages the answer cited (with page numbers)
- Bonus: conversation history (follow-ups are rewritten into standalone questions before retrieval), document summary.
- Embedding cache (`.emb_cache_*.npy`) and automatic retries on network errors

See `DECISIONS.md` for design choices and the chunking experiment.
