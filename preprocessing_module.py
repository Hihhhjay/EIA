"""
Module 2 – Report Preprocessing Module
=======================================
Mirrors the ESGReveal Report Preprocessing Module:

  1. Layout Analysis  → pdfplumber (text + table extraction)
  2. Content Extraction → text chunks + table rows
  3. Multi-type KnowledgeBase → FAISS vector index
     - TextKnowledgeBase   (paragraph chunks)
     - TableKnowledgeBase  (table cells flattened to sentences)
     - DocOutlineKnowledgeBase (section headers / TOC)
"""

import os, json, re
import numpy as np
import pandas as pd
import pdfplumber
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, INDEX_DIR


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into overlapping character-level chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end].strip())
        start += size - overlap
    return [c for c in chunks if len(c) > 20]


def _table_to_sentences(table) -> list:
    """Convert a pdfplumber table (list of lists) to natural-language sentences."""
    sentences = []
    if not table or len(table) < 2:
        return sentences
    headers = [str(h).strip() if h else "" for h in table[0]]
    for row in table[1:]:
        parts = []
        for h, cell in zip(headers, row):
            if cell and str(cell).strip():
                parts.append(f"{h}: {str(cell).strip()}")
        if parts:
            sentences.append(" | ".join(parts))
    return sentences


def _extract_outline(pages) -> list:
    """Heuristically extract section headings (all-caps or short bold-like lines)."""
    headings = []
    pattern  = re.compile(r"^([A-Z][A-Z\s\d\.]{3,60})$")
    for page in pages:
        text = page.extract_text() or ""
        for line in text.split("\n"):
            line = line.strip()
            if pattern.match(line) and len(line) < 80:
                headings.append(line)
    return list(dict.fromkeys(headings))   # deduplicate


# ── Main class ────────────────────────────────────────────────────────────────

class ReportPreprocessor:
    """
    Processes a PDF report and builds three FAISS knowledge bases:
      - text_kb    : paragraph chunks
      - table_kb   : flattened table rows
      - outline_kb : section headings
    """

    def __init__(self, pdf_path: str, report_name: str = "report"):
        self.pdf_path    = pdf_path
        self.report_name = report_name
        self.model       = SentenceTransformer(EMBED_MODEL)

        self.text_chunks:    list = []
        self.table_sents:    list = []
        self.outline_items:  list = []

        self.text_index:   faiss.IndexFlatIP = None
        self.table_index:  faiss.IndexFlatIP = None
        self.outline_index: faiss.IndexFlatIP = None

    # ── Extraction ────────────────────────────────────────────────────────────

    def extract(self):
        """Run PDF extraction for text, tables, and outline."""
        print(f"[Preprocessing] Opening PDF: {self.pdf_path}")
        with pdfplumber.open(self.pdf_path) as pdf:
            pages = pdf.pages
            print(f"[Preprocessing] Total pages: {len(pages)}")

            full_text = ""
            for page in tqdm(pages, desc="Extracting text"):
                t = page.extract_text()
                if t:
                    full_text += t + "\n"

            for page in tqdm(pages, desc="Extracting tables"):
                for tbl in (page.extract_tables() or []):
                    self.table_sents.extend(_table_to_sentences(tbl))

            self.outline_items = _extract_outline(pages)

        self.text_chunks = _chunk_text(full_text)
        print(f"[Preprocessing] Text chunks: {len(self.text_chunks)}")
        print(f"[Preprocessing] Table sents: {len(self.table_sents)}")
        print(f"[Preprocessing] Outline items: {len(self.outline_items)}")

    # ── Embedding & indexing ──────────────────────────────────────────────────

    def _build_index(self, sentences: list) -> faiss.IndexFlatIP:
        if not sentences:
            return None
        embs = self.model.encode(sentences, show_progress_bar=False,
                                  normalize_embeddings=True)
        dim  = embs.shape[1]
        idx  = faiss.IndexFlatIP(dim)   # inner-product = cosine on normalised vecs
        idx.add(embs.astype("float32"))
        return idx

    def build_knowledge_bases(self):
        """Encode all extracted content and create FAISS indices."""
        print("[Preprocessing] Building Text KnowledgeBase...")
        self.text_index = self._build_index(self.text_chunks)

        print("[Preprocessing] Building Table KnowledgeBase...")
        self.table_index = self._build_index(self.table_sents)

        print("[Preprocessing] Building Outline KnowledgeBase...")
        self.outline_index = self._build_index(self.outline_items)

        print("[Preprocessing] All knowledge bases built.")

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self):
        prefix = os.path.join(INDEX_DIR, self.report_name)

        def _save(index, items, name):
            if index is None:
                return
            faiss.write_index(index, f"{prefix}_{name}.faiss")
            with open(f"{prefix}_{name}_items.json", "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)

        _save(self.text_index,    self.text_chunks,   "text")
        _save(self.table_index,   self.table_sents,   "table")
        _save(self.outline_index, self.outline_items, "outline")
        print(f"[Preprocessing] Saved indices to {INDEX_DIR}/")

    def load(self):
        prefix = os.path.join(INDEX_DIR, self.report_name)

        def _load(name):
            fi = f"{prefix}_{name}.faiss"
            ji = f"{prefix}_{name}_items.json"
            if not os.path.exists(fi):
                return None, []
            index = faiss.read_index(fi)
            with open(ji, encoding="utf-8") as f:
                items = json.load(f)
            return index, items

        self.text_index,    self.text_chunks   = _load("text")
        self.table_index,   self.table_sents   = _load("table")
        self.outline_index, self.outline_items = _load("outline")
        print(f"[Preprocessing] Loaded indices from {INDEX_DIR}/")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_n: int = 10) -> list:
        """
        Vector similarity search across all three knowledge bases.
        Returns top_n combined results as (score, text) tuples.
        """
        q_emb = self.model.encode([query], normalize_embeddings=True).astype("float32")
        results = []

        for index, items in [
            (self.text_index,    self.text_chunks),
            (self.table_index,   self.table_sents),
            (self.outline_index, self.outline_items),
        ]:
            if index is None or not items:
                continue
            k = min(top_n, index.ntotal)
            scores, ids = index.search(q_emb, k)
            for score, idx in zip(scores[0], ids[0]):
                if idx < len(items):
                    results.append((float(score), items[idx]))

        # Sort by cosine score descending, return top_n
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_n]


if __name__ == "__main__":
    proc = ReportPreprocessor("data/EPA-Report-East-Rockingham-Waste-to-Energy-Revised-Proposal.pdf",
                               report_name="epa_report")
    proc.extract()
    proc.build_knowledge_bases()
    proc.save()
