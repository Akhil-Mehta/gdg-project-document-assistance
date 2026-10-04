"""Run the same questions under two settings and print a comparison table.
Edit QUESTIONS to match your document. Keep >=5, include >=1 unanswerable.
Fill 'expect_page' by checking the PDF yourself, then retrieval hit-rate is automatic.
Usage: python evaluate.py doc.pdf
"""
import sys
from docqa import DocIndex, answer

QUESTIONS = [  # drafted from passages seen in output; VERIFY each against the PDF
    {"q": "What are the opening hours of the help desk?", "expect_page": 1},
    {"q": "How can students contact the help desk?", "expect_page": 1},
    {"q": "Where should students report learning-portal technical issues?", "expect_page": 2},
    {"q": "What must students explain if they used AI tools in a submission?", "expect_page": 2},
    {"q": "Is the help desk open on weekends?", "expect_page": 1},
    {"q": "What is the date and venue of the next GDG-USAR event?", "expect_page": None},  # doc says not specified
    {"q": "What is the capital of France?", "expect_page": None},  # fully off-topic
]
SETTINGS = {
    "A: size=400 overlap=50 k=3":  dict(chunk_size=400, overlap=50, k=3),
    "B: size=1000 overlap=200 k=5": dict(chunk_size=1000, overlap=200, k=5),
}

path = sys.argv[1]
for name, cfg in SETTINGS.items():
    idx = DocIndex(path, cfg["chunk_size"], cfg["overlap"])
    print(f"\n=== {name}  ({len(idx.chunks)} chunks) ===")
    hits_ok = 0
    for t in QUESTIONS:
        ans, hits = answer(idx, t["q"], k=cfg["k"])
        pages = [c.page for c, _ in hits]
        if t["expect_page"]:
            hits_ok += t["expect_page"] in pages
        refused = "can't find" in ans.lower()
        tag = ""
        if t["expect_page"] is None:
            tag = "  [REFUSED - correct]" if refused else "  [ANSWERED - should have refused!]"
        print(f"\nQ: {t['q']}\n  top score={hits[0][1]:.2f} pages={pages}\n  A: {ans}{tag}")
    n = sum(1 for t in QUESTIONS if t["expect_page"])
    print(f"\nRetrieval hit-rate: {hits_ok}/{n}")
