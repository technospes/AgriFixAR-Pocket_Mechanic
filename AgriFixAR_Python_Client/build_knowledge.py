"""
build_knowledge.py — Production RAG Knowledge Base Builder v5.1
AgriFixAR Multimodal Diagnostic System

CHANGES IN 5.1:
  - Relaxed regex to accept chunk IDs without brackets (e.g., "🔹 Chunk 1 —")
  - Added filename-based machine_type inference (removes strict requirement)
  - Added smart auto-categorization for problem_categories if missing
"""

import os
import re
import json
import hashlib
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("knowledge_build.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Universal problem categories for auto-tagging
CATEGORY_KEYWORDS = {
    "not_starting": ["not start", "won't start", "no start", "dead", "crank", "starter", "self", "battery"],
    "noise": ["noise", "sound", "knocking", "clicking", "grinding", "rattling", "squealing", "awaaz"],
    "leaking": ["leak", "dripping", "seeping", "oil leak", "tapak"],
    "overheating": ["overheat", "hot", "boiling", "temperature", "garam", "radiator"],
    "vibration": ["vibrat", "shaking", "wobbl", "shake", "hil"],
    "smoke": ["smoke", "fumes", "dhuan"],
    "power_loss": ["low power", "no power", "weak", "sluggish", "power loss", "pickup"],
    "electrical": ["battery", "wiring", "fuse", "relay", "alternator", "voltage", "bijli"],
    "fuel": ["fuel", "diesel", "petrol", "injector", "carburetor", "pump"],
    "water_flow": ["water", "flow", "pressure", "discharge", "pani"],
    "hydraulic": ["hydraulic", "lift", "3 point", "cylinder"],
    "transmission": ["gear", "clutch", "pto", "transmission"],
}

# ═══════════════════════════════════════════════════════════════════════════
# STRUCTURED CHUNK PARSER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ParsedChunk:
    chunk_id: str
    problem: str
    machine_type: str
    tags: List[str]
    problem_categories: List[str]
    parts: List[str]
    escalate_if: str
    content: str
    source_file: str
    
    def to_metadata(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "problem": self.problem,
            "machine_type": self.machine_type,
            "tags": self.tags if len(self.tags) > 0 else ["none"],
            "problem_categories": self.problem_categories if len(self.problem_categories) > 0 else ["general"],
            "parts": self.parts if len(self.parts) > 0 else ["none"],
            "escalate_if": self.escalate_if,
            "source_file": self.source_file,
            "content_hash": hashlib.md5(
                re.sub(r"\s+", " ", self.content).strip().encode()
            ).hexdigest(),
        }

class StructuredChunkParser:
    # 5.1 FIX: Accepts "🔹 Chunk 1 —" OR "🔹 Chunk [1] —"
    CHUNK_BOUNDARY = re.compile(r"🔹\s*Chunk\s+\[?([\w\d_]+)\]?\s*[—–-]\s*(.+?)(?=\n|$)", re.IGNORECASE)
    
    FIELD_PATTERNS = {
        "problem": re.compile(r"(?:PROBLEM|Problem)\s*:\s*(.+?)(?=\n(?:MACHINE_TYPE|TAGS|SYMPTOM|LIKELY_CAUSES|PROBLEM_CATEGORIES|PARTS|ESCALATE_IF|CAUSE|FIX|$))", re.IGNORECASE | re.DOTALL),
        "machine_type": re.compile(r"(?:MACHINE_TYPE|Machine_Type)\s*:\s*(.+?)(?=\n|$)", re.IGNORECASE),
        "tags": re.compile(r"(?:TAGS|Tags)\s*:\s*(.+?)(?=\n|$)", re.IGNORECASE),
        "problem_categories": re.compile(r"(?:PROBLEM_CATEGORIES|Problem_Categories)\s*:\s*(.+?)(?=\n|$)", re.IGNORECASE),
        "parts": re.compile(r"(?:PARTS|Parts)\s*:\s*(.+?)(?=\n|$)", re.IGNORECASE),
        "escalate_if": re.compile(r"(?:ESCALATE_IF|Escalate_If)\s*:\s*(.+?)(?=\n(?:🔹|$))", re.IGNORECASE | re.DOTALL),
    }
    
    def __init__(self):
        self.rejected_chunks: List[Tuple[str, str]] = []
        
    def parse_file(self, file_path: Path) -> List[ParsedChunk]:
        logger.info(f"📖 Parsing: {file_path.name}")
        text = file_path.read_text(encoding="utf-8")
        
        chunks_raw = self._split_chunks(text)
        parsed = []
        for chunk_id, chunk_text in chunks_raw:
            try:
                chunk_obj = self._parse_chunk(chunk_id, chunk_text, file_path.name)
                if chunk_obj:
                    parsed.append(chunk_obj)
            except Exception as e:
                self.rejected_chunks.append((chunk_id, f"Parse error: {e}"))
        
        logger.info(f"   ✓ {file_path.name} → {len(parsed)} valid chunks, {len(chunks_raw) - len(parsed)} rejected")
        return parsed
    
    def _split_chunks(self, text: str) -> List[Tuple[str, str]]:
        chunks = []
        matches = list(self.CHUNK_BOUNDARY.finditer(text))
        for i, match in enumerate(matches):
            chunk_id = match.group(1).strip()
            start_pos = match.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chunks.append((chunk_id, text[start_pos:end_pos].strip()))
        return chunks
    
    def _parse_chunk(self, chunk_id: str, text: str, source_file: str) -> Optional[ParsedChunk]:
        problem_match = self.FIELD_PATTERNS["problem"].search(text)
        machine_match = self.FIELD_PATTERNS["machine_type"].search(text)
        
        if not problem_match:
            self.rejected_chunks.append((chunk_id, "Missing PROBLEM field"))
            return None
            
        problem = problem_match.group(1).strip()
        
        # 5.1 FIX: Smart machine_type inference if missing from text
        if machine_match:
            machine_type = machine_match.group(1).strip().lower()
        else:
            filename_lower = source_file.lower()
            if "harvester" in filename_lower: machine_type = "harvester"
            elif "pump" in filename_lower: machine_type = "water_pump"
            elif "thresher" in filename_lower: machine_type = "thresher"
            else: machine_type = "tractor"
            
        tags_match = self.FIELD_PATTERNS["tags"].search(text)
        tags = self._parse_list_field(tags_match.group(1) if tags_match else "")
        
        cats_match = self.FIELD_PATTERNS["problem_categories"].search(text)
        problem_categories = self._parse_list_field(cats_match.group(1) if cats_match else "")
        
        # 5.1 FIX: Auto-tag problem categories if empty (Crucial for RAG Pass 1)
        if not problem_categories:
            combined_text = f"{problem} {' '.join(tags)}".lower()
            inferred = []
            for cat, kws in CATEGORY_KEYWORDS.items():
                if any(kw in combined_text for kw in kws):
                    inferred.append(cat)
            problem_categories = inferred
            
        parts_match = self.FIELD_PATTERNS["parts"].search(text)
        parts = self._parse_list_field(parts_match.group(1) if parts_match else "")
        
        escalate_match = self.FIELD_PATTERNS["escalate_if"].search(text)
        escalate_if = escalate_match.group(1).strip() if escalate_match else ""
        
        # Strip metadata from body so embeddings focus purely on diagnostics
        content_clean = text
        for pattern in self.FIELD_PATTERNS.values():
            content_clean = pattern.sub("", content_clean)
        content_clean = re.sub(r"\n{3,}", "\n\n", content_clean).strip()
        
        return ParsedChunk(
            chunk_id=chunk_id, problem=problem, machine_type=machine_type,
            tags=tags, problem_categories=problem_categories, parts=parts,
            escalate_if=escalate_if, content=content_clean, source_file=source_file
        )
    
    @staticmethod
    def _parse_list_field(raw: str) -> List[str]:
        if not raw: return []
        return sorted(set([item.strip().lower() for item in re.split(r"[,;]+", raw) if item.strip()]))

# ═══════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE BUILDER
# ═══════════════════════════════════════════════════════════════════════════

class KnowledgeBaseBuilder:
    def __init__(self, knowledge_dir: str, db_dir: str):
        self.knowledge_dir = Path(knowledge_dir)
        self.db_dir = Path(db_dir)
        self.parser = StructuredChunkParser()
        self.stats = {
            "files_processed": 0, "files_failed": 0, "chunks_parsed": 0,
            "chunks_rejected": 0, "chunks_stored": 0, "machine_types": set(),
            "problem_categories": set(),
        }
    
    def build(self) -> bool:
        logger.info("=" * 80)
        logger.info("🚀 AgriFix RAG Knowledge Base Builder v5.1")
        logger.info("=" * 80)
        
        try:
            api_key = os.getenv("GOOGLE_AI_API_KEY")
            if not api_key: raise ValueError("GOOGLE_AI_API_KEY not found in .env")
            
            all_chunks = self._parse_all_files()
            if not all_chunks:
                logger.error("❌ No valid chunks parsed from knowledge base!")
                return False
                
            documents = self._deduplicate([self._to_document(c) for c in all_chunks])
            success = self._embed_and_store(documents, api_key)
            self._generate_report(all_chunks)
            
            if success:
                logger.info("=" * 80)
                logger.info("✅ Knowledge base build completed successfully!")
                logger.info(f"📊 {self.stats['chunks_stored']} chunks stored in ChromaDB")
                logger.info(f"🤖 Machine types: {', '.join(self.stats['machine_types'])}")
                logger.info("=" * 80)
            return success
        except Exception as e:
            logger.error(f"❌ Build failed: {e}", exc_info=True)
            return False
            
    def _parse_all_files(self) -> List[ParsedChunk]:
        all_chunks = []
        # 5.1 FIX: Recursive globbing incase they are in subfolders
        txt_files = list(self.knowledge_dir.rglob("*.txt"))
        if not txt_files:
            logger.warning(f"⚠️  No .txt files found in {self.knowledge_dir}")
            return []
            
        logger.info(f"📁 Found {len(txt_files)} knowledge files")
        for file_path in txt_files:
            try:
                chunks = self.parser.parse_file(file_path)
                all_chunks.extend(chunks)
                self.stats["files_processed"] += 1
            except Exception as e:
                logger.error(f"❌ Failed to process {file_path.name}: {e}")
                self.stats["files_failed"] += 1
                
        self.stats["chunks_parsed"] = len(all_chunks)
        self.stats["chunks_rejected"] = len(self.parser.rejected_chunks)
        return all_chunks
        
    def _to_document(self, chunk: ParsedChunk) -> Document:
        self.stats["machine_types"].add(chunk.machine_type)
        self.stats["problem_categories"].update(chunk.problem_categories)
        return Document(page_content=chunk.content, metadata=chunk.to_metadata())
        
    def _deduplicate(self, documents: List[Document]) -> List[Document]:
        seen = set()
        unique = []
        for doc in documents:
            if doc.metadata["content_hash"] not in seen:
                seen.add(doc.metadata["content_hash"])
                unique.append(doc)
        if len(documents) - len(unique) > 0:
            logger.info(f"🗑️  Removed {len(documents) - len(unique)} duplicate chunks")
        return unique
        
    def _embed_and_store(self, documents: List[Document], api_key: str) -> bool:
        import time
        logger.info("🧠 Creating vector database...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
        
        if self.db_dir.exists(): shutil.rmtree(self.db_dir)
        db = Chroma(persist_directory=str(self.db_dir), embedding_function=embeddings)
        
        batch_size = 50
        total = len(documents)
        total_batches = (total + batch_size - 1) // batch_size
        logger.info(f"📦 Embedding {total} chunks in {total_batches} batches")
        
        for batch_num, i in enumerate(range(0, total, batch_size), start=1):
            batch = documents[i:i + batch_size]
            try:
                db.add_documents(batch)
                logger.info(f"   ✅ Batch {batch_num}/{total_batches} ({i + len(batch)}/{total} chunks)")
            except Exception as e:
                logger.error(f"   ❌ Batch {batch_num} failed: {e} — retrying...")
                time.sleep(10)
                db.add_documents(batch)
            if batch_num < total_batches: time.sleep(2)
                
        self.stats["chunks_stored"] = len(db.get()["ids"])
        return True
        
    def _generate_report(self, chunks: List[ParsedChunk]):
        report = {
            "build_date": datetime.now().isoformat(),
            "statistics": {
                "chunks_parsed": self.stats["chunks_parsed"],
                "chunks_rejected": self.stats["chunks_rejected"],
                "chunks_stored": self.stats["chunks_stored"],
            },
            "rejected_chunks": [{"id": cid, "reason": r} for cid, r in self.parser.rejected_chunks]
        }
        report_path = self.db_dir / "build_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if self.parser.rejected_chunks:
            logger.warning(f"⚠️  {len(self.parser.rejected_chunks)} chunks rejected (see build_report.json)")

if __name__ == "__main__":
    builder = KnowledgeBaseBuilder(knowledge_dir="./knowledge_base", db_dir="./chroma_db")
    import sys
    sys.exit(0 if builder.build() else 1)