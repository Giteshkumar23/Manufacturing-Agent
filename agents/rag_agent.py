"""
Agent 5 — RAG Knowledge Agent
Handles document ingestion, chunking, local vector storage (TF-IDF based),
and retrieval for grounding AI explanations in uploaded documentation.
"""
import time
import logging
import json
import os
import re
from datetime import datetime
from models.database import get_db, log_agent

logger = logging.getLogger(__name__)

# Try to load scikit-learn for TF-IDF vectorization
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available — RAG will use keyword matching fallback")

# Try PyPDF2 for PDF reading
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


CHUNK_SIZE = 400      # characters per chunk
CHUNK_OVERLAP = 80
TOP_K = 3             # number of chunks to retrieve


class RAGKnowledgeAgent:
    name = "RAGKnowledgeAgent"

    def __init__(self):
        self._vectorizer = None
        self._corpus = []     # list of {'doc_id': int, 'chunk_idx': int, 'text': str}
        self._vectors = None  # numpy array (n_chunks × vocab)
        self._loaded = False

    def _ensure_loaded(self):
        if not self._loaded:
            self._reload_index()
            self._loaded = True

    def _reload_index(self):
        """Load all document chunks from the DB and rebuild the TF-IDF index."""
        with get_db() as conn:
            rows = conn.execute("SELECT id, chunks FROM documents WHERE chunks IS NOT NULL").fetchall()

        self._corpus = []
        for row in rows:
            try:
                chunks = json.loads(row['chunks'])
                for i, chunk in enumerate(chunks):
                    self._corpus.append({
                        'doc_id': row['id'],
                        'chunk_idx': i,
                        'text': chunk,
                    })
            except Exception:
                pass

        if not self._corpus:
            self._vectorizer = None
            self._vectors = None
            return

        if SKLEARN_AVAILABLE:
            texts = [c['text'] for c in self._corpus]
            self._vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            self._vectors = self._vectorizer.fit_transform(texts)
        # else: keyword fallback — no pre-computation needed

    def ingest_document(self, filename: str, original_name: str,
                        content: str, doc_type: str = 'manual') -> int:
        """
        Chunk content, store in DB, rebuild index.
        Returns the new document ID.
        """
        chunks = self._chunk_text(content)
        with get_db() as conn:
            cursor = conn.execute("""
                INSERT INTO documents (filename, original_name, doc_type, content, chunks)
                VALUES (?,?,?,?,?)
            """, (filename, original_name, doc_type, content[:5000], json.dumps(chunks)))
            doc_id = cursor.lastrowid

        # Rebuild index
        self._loaded = False
        self._ensure_loaded()
        log_agent(self.name, 'ingest', f"Ingested {len(chunks)} chunks from {original_name}")
        return doc_id

    def retrieve(self, query: str, top_k: int = TOP_K) -> list:
        """
        Retrieve the most relevant chunks for a query.
        Returns list of {'text': str, 'doc_id': int, 'score': float, 'source': str}
        """
        self._ensure_loaded()
        if not self._corpus:
            return []

        if SKLEARN_AVAILABLE and self._vectorizer is not None:
            return self._tfidf_retrieve(query, top_k)
        else:
            return self._keyword_retrieve(query, top_k)

    def _tfidf_retrieve(self, query: str, top_k: int) -> list:
        q_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._vectors).flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] < 0.01:
                continue
            chunk = self._corpus[idx]
            source = self._get_doc_name(chunk['doc_id'])
            results.append({
                'text':   chunk['text'],
                'doc_id': chunk['doc_id'],
                'score':  round(float(scores[idx]), 4),
                'source': source,
            })
        return results

    def _keyword_retrieve(self, query: str, top_k: int) -> list:
        """Simple keyword overlap scoring fallback."""
        query_words = set(query.lower().split())
        scored = []
        for chunk in self._corpus:
            chunk_words = set(chunk['text'].lower().split())
            overlap = len(query_words & chunk_words)
            if overlap > 0:
                scored.append((overlap, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, chunk in scored[:top_k]:
            source = self._get_doc_name(chunk['doc_id'])
            results.append({
                'text':   chunk['text'],
                'doc_id': chunk['doc_id'],
                'score':  score,
                'source': source,
            })
        return results

    def _get_doc_name(self, doc_id: int) -> str:
        with get_db() as conn:
            row = conn.execute(
                "SELECT original_name FROM documents WHERE id=?", (doc_id,)
            ).fetchone()
        return row['original_name'] if row else f"Document #{doc_id}"

    def build_rag_context(self, query: str) -> tuple[str, list]:
        """
        Returns (context_string, source_list) for use in AI prompts.
        """
        chunks = self.retrieve(query)
        if not chunks:
            return '', []
        context_parts = []
        sources = []
        for c in chunks:
            context_parts.append(f"[From: {c['source']}]\n{c['text']}")
            if c['source'] not in sources:
                sources.append(c['source'])
        return '\n\n---\n\n'.join(context_parts), sources

    def get_documents(self):
        with get_db() as conn:
            rows = conn.execute(
                "SELECT id, original_name, doc_type, uploaded_at, "
                "(SELECT COUNT(*) FROM json_each(chunks)) as chunk_count "
                "FROM documents ORDER BY uploaded_at DESC"
            ).fetchall()
        # Handle potential sqlite version issues
        with get_db() as conn:
            rows = conn.execute(
                "SELECT id, original_name, doc_type, uploaded_at, chunks FROM documents ORDER BY uploaded_at DESC"
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            try:
                d['chunk_count'] = len(json.loads(d.get('chunks') or '[]'))
            except Exception:
                d['chunk_count'] = 0
            d.pop('chunks', None)
            result.append(d)
        return result

    def delete_document(self, doc_id: int):
        with get_db() as conn:
            conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        self._loaded = False

    @staticmethod
    def _chunk_text(text: str) -> list:
        """Split text into overlapping chunks."""
        text = re.sub(r'\s+', ' ', text).strip()
        chunks = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    @staticmethod
    def extract_text(file_path: str, original_name: str) -> str:
        """Extract text from uploaded file."""
        lower = original_name.lower()
        if lower.endswith('.pdf') and PDF_AVAILABLE:
            try:
                text = ''
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text += page.extract_text() or ''
                return text
            except Exception as e:
                logger.error(f"PDF extraction error: {e}")
                return ''
        else:
            # Treat as plain text
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Text extraction error: {e}")
                return ''
