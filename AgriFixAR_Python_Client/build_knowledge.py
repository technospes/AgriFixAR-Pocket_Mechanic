"""
build_knowledge.py — Optimized RAG Knowledge Base Builder
Agricultural Machinery Diagnostic Assistant

Improvements over v1:
  - Structured semantic chunking (detects manual section boundaries)
  - section_type metadata per chunk (cause_fix, fix, cause, problem_header, general)
  - problem_categories stored as list (not CSV string) for $contains filtering
  - Normalized whitespace hashing to prevent duplicate chunks
  - Improved brand extraction with stop-word filter
  - Machine type detection scans first 10 000 characters
  - No private Chroma APIs (_collection.count removed)
"""

import os
import re
import json
import hashlib
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
import logging
from datetime import datetime
import sys as _sys

from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

# ── Logging — UTF-8 safe on Windows ──────────────────────────────────────────
_file_handler = logging.FileHandler("knowledge_build.log", encoding="utf-8")
try:
    _stream_handler = logging.StreamHandler(
        stream=open(_sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
    )
except Exception:
    _stream_handler = logging.StreamHandler()
    _stream_handler.stream = _sys.stdout

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[_file_handler, _stream_handler],
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Words that look like brand names but are generic — skip during brand detection
BRAND_IGNORE_WORDS: Set[str] = {
    "Engine", "Motor", "System", "Repair", "Service", "Manual",
    "Guide", "Handbook", "Safety", "Warning", "Caution", "Note",
    "Chapter", "Section", "Page", "Figure", "Table", "Appendix",
    "Tractor", "Pump", "Harvester", "Thresher", "Generator",
    "Check", "Inspect", "Replace", "Install", "Remove", "Clean",
    "First", "Second", "Third", "Step", "Part", "Tool", "Unit",
}

# Structured manual section boundary patterns (regex, case-insensitive)
SECTION_PATTERNS = [
    # PROBLEM N — title  (your format with emoji or without)
    (r"(?:[\U0001F7E6\U0001F7E7\U0001F7E8\U0001F7E9\U0001F7EA\U0001F7EB]?\s*)?PROBLEM\s+\d+\s*[—–-]", "problem_header"),
    # Chunk N — title
    (r"Chunk\s+\d+\s*[—–-]", "cause_fix"),
    # Cause / Possible Cause
    (r"(?:Possible\s+)?CAUSE\s*:", "cause"),
    # Fix / Corrective Action / Repair
    (r"(?:CORRECTIVE\s+ACTION|FIX|REPAIR\s+STEPS?)\s*:", "fix"),
    # What's Happening / Symptom
    (r"(?:What'?s?\s+Happening|SYMPTOM)\s*:", "cause"),
    # What to Do / Steps
    (r"(?:What\s+to\s+Do|STEPS?)\s*:", "fix"),
]

FALLBACK_CHUNK_SIZE = 600
FALLBACK_CHUNK_OVERLAP = 100


# =============================================================================
# SEMANTIC CHUNKER
# =============================================================================

class SemanticManualChunker:
    """
    Splits structured troubleshooting manuals into semantically complete chunks.

    Each chunk aims to contain both the CAUSE and FIX so the LLM always
    receives actionable context. Falls back to RecursiveCharacterTextSplitter
    when no structured markers are found.
    """

    def split_document(self, doc: Document) -> List[Document]:
        """Split a single Document into structured chunks with section_type metadata."""
        text = doc.page_content
        structured_chunks = self._split_structured(text)

        if not structured_chunks:
            # Fallback: naive recursive split
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=FALLBACK_CHUNK_SIZE,
                chunk_overlap=FALLBACK_CHUNK_OVERLAP,
            )
            raw_chunks = splitter.split_text(text)
            structured_chunks = [{"text": c, "section_type": "general"} for c in raw_chunks if c.strip()]

        result = []
        for item in structured_chunks:
            content = item["text"].strip()
            if not content:
                continue

            # Normalize whitespace for stable hashing
            normalized = re.sub(r"\s+", " ", content).strip()
            content_hash = hashlib.md5(normalized.encode()).hexdigest()

            child_meta = dict(doc.metadata)
            child_meta["section_type"] = item["section_type"]
            child_meta["content_hash"] = content_hash

            result.append(Document(page_content=content, metadata=child_meta))

        return result

    def _split_structured(self, text: str) -> List[Dict]:
        """
        Detect manual boundaries and group lines into semantic blocks.

        Strategy:
          1. Walk through lines looking for PROBLEM or Chunk headers.
          2. When a new PROBLEM header is found, flush the previous block.
          3. Chunk N blocks stay together (cause + fix in one document).
          4. Returns [] if no structure detected.
        """
        # Check whether any structural markers exist at all
        combined_pattern = "|".join(p for p, _ in SECTION_PATTERNS)
        if not re.search(combined_pattern, text, re.IGNORECASE):
            return []

        chunks: List[Dict] = []
        current_lines: List[str] = []
        current_type: str = "general"

        lines = text.split("\n")

        def flush():
            nonlocal current_lines, current_type
            block = "\n".join(current_lines).strip()
            if block:
                chunks.append({"text": block, "section_type": current_type})
            current_lines = []
            current_type = "general"

        for line in lines:
            matched_type = None
            for pattern, sec_type in SECTION_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    matched_type = sec_type
                    break

            if matched_type == "problem_header":
                # New top-level problem — flush previous chunk
                flush()
                current_lines = [line]
                current_type = "problem_header"
            elif matched_type == "cause_fix":
                # A "Chunk N —" boundary signals a self-contained cause+fix block
                flush()
                current_lines = [line]
                current_type = "cause_fix"
            elif matched_type in ("cause", "fix"):
                # Sub-section inside current block — keep together to preserve context
                current_lines.append(line)
                # Upgrade section_type if we find both cause and fix signals
                if current_type == "cause" and matched_type == "fix":
                    current_type = "cause_fix"
                elif current_type == "general":
                    current_type = matched_type
            else:
                current_lines.append(line)

        flush()
        return chunks

    def split_documents(self, docs: List[Document]) -> List[Document]:
        """Split a list of Documents."""
        result = []
        for doc in docs:
            result.extend(self.split_document(doc))
        return result


# =============================================================================
# KNOWLEDGE BASE BUILDER
# =============================================================================

class KnowledgeBaseBuilder:
    """
    Builds a ChromaDB vector store from troubleshooting manuals.

    Key improvements:
      - Uses SemanticManualChunker instead of naive character splitting
      - Stores problem_categories as a proper list (supports $contains filter)
      - Scans 10 000 chars for machine type detection
      - Skips brand stop-words to reduce false positives
      - Uses len(chunks) instead of db._collection.count()
    """

    def __init__(self, knowledge_dir: str = "./knowledge_base", db_dir: str = "./chroma_db"):
        self.knowledge_dir = Path(knowledge_dir)
        self.db_dir = Path(db_dir)
        self.failed_files: List[Tuple[Path, str]] = []
        self.discovered_brands: Set[str] = set()
        self.discovered_machines: Set[str] = set()
        self.discovered_problems: Set[str] = set()

    # ── Environment ───────────────────────────────────────────────────────────

    def load_environment(self) -> str:
        load_dotenv()
        api_key = os.getenv("GOOGLE_AI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_AI_API_KEY not found in .env file")
        return api_key

    # ── Problem keyword extraction ────────────────────────────────────────────

    def extract_problem_keywords(self, content: str, filename: str) -> List[str]:
        """
        Detect problem categories from content + filename.
        Returns a plain Python list (not a CSV string).
        """
        keywords: Set[str] = set()
        # Scan first 10 000 characters for speed
        combined = f"{filename.lower()} {content.lower()[:10000]}"

        problem_indicators: Dict[str, List[str]] = {
            "not_starting":  ["not start", "won't start", "no start", "dead", "not working", "won't turn on"],
            "noise":         ["noise", "sound", "knocking", "clicking", "grinding", "rattling", "squealing", "humming"],
            "leaking":       ["leak", "leaking", "dripping", "seeping", "oil leak", "water leak", "fuel leak"],
            "overheating":   ["hot", "overheat", "temperature", "boiling", "heat", "overheat"],
            "vibration":     ["vibrat", "shaking", "wobbl", "unsteady"],
            "smoke":         ["smoke", "smoking", "fumes", "exhaust"],
            "power_loss":    ["weak", "low power", "no power", "poor performance", "sluggish", "power loss"],
            "electrical":    ["battery", "starter", "wiring", "electrical", "fuse", "spark", "short circuit"],
            "fuel":          ["fuel", "diesel", "petrol", "gas", "carburetor", "injector", "fuel pump"],
            "hydraulic":     ["hydraulic", "oil pressure", "lift", "hydraulic fluid"],
            "mechanical":    ["gear", "clutch", "belt", "chain", "bearing", "worn"],
            "water_flow":    ["water", "pump", "flow", "pressure", "discharge", "no water", "low water"],
            "cooling":       ["radiator", "coolant", "cooling", "water temperature"],
            "engine":        ["engine", "piston", "cylinder", "compression"],
        }

        for category, terms in problem_indicators.items():
            if any(term in combined for term in terms):
                keywords.add(category)

        return sorted(keywords)  # sorted list for deterministic output

    # ── Brand extraction ──────────────────────────────────────────────────────

    def extract_brands_universal(self, text: str) -> Set[str]:
        """
        Detect brand names. Filters out generic machinery vocabulary
        defined in BRAND_IGNORE_WORDS to reduce false positives.
        """
        brands: Set[str] = set()

        # Pattern 1: BrandName + Model number  (e.g., "Mahindra 575")
        for match in re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:\d+|[A-Z]{2,})", text):
            word = match.strip().split()[0].title()
            if word not in BRAND_IGNORE_WORDS and len(word) > 3:
                brands.add(word)

        # Pattern 2: BrandName + machine keyword  (e.g., "Kirloskar Pump")
        for match in re.findall(
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:Tractor|Pump|Motor|Engine|Harvester|Cultivator|Plough)",
            text, re.IGNORECASE
        ):
            word = match.strip().split()[0].title()
            if word not in BRAND_IGNORE_WORDS and len(word) > 3:
                brands.add(word)

        # Pattern 3: Capitalized words that appear 3+ times (likely a brand name)
        word_freq: Dict[str, int] = {}
        for word in re.findall(r"\b([A-Z][a-z]{3,})\b", text):
            if word not in BRAND_IGNORE_WORDS and word not in {"The", "This", "That", "When", "Where", "What", "Which"}:
                word_freq[word] = word_freq.get(word, 0) + 1
        brands.update(w for w, freq in word_freq.items() if freq >= 3)

        return brands

    def extract_brand_from_filename(self, filename: str) -> Optional[str]:
        """Best-effort brand extraction from filename."""
        clean = filename.replace("_", " ").replace("-", " ")
        for word in clean.split():
            if word[0].isupper() and len(word) > 3 and word not in BRAND_IGNORE_WORDS:
                return word
        return None

    # ── Machine type detection ────────────────────────────────────────────────

    def extract_machine_type_universal(self, filename: str, content: str) -> str:
        """
        Detect machine type by scanning filename + first 10 000 content chars.
        Scanning 10 000 chars (up from 2 000) improves accuracy on longer manuals.
        """
        filename_lower = filename.lower()
        # Scan more of the document for better accuracy
        content_lower = content.lower()[:10000]

        machine_keywords = [
            "tractor", "pump", "motor", "engine", "harvester", "combine",
            "thresher", "cultivator", "plough", "plow", "seeder", "drill",
            "sprayer", "spreader", "trailer", "rotavator", "tiller",
            "weeder", "reaper", "baler", "loader", "digger",
            "submersible", "centrifugal", "generator", "alternator",
        ]

        # Filename first — most reliable signal
        for kw in machine_keywords:
            if kw in filename_lower:
                return kw

        # Content scan
        for kw in machine_keywords:
            if kw in content_lower:
                return kw

        # Directory path
        for part in Path(filename).parts:
            part_lower = part.lower()
            for kw in machine_keywords:
                if kw in part_lower:
                    return kw

        # Contextual pattern ("this pump is..." / "the tractor has...")
        ctx_match = re.search(r"(?:this|the)\s+([a-z]+)\s+(?:is|has|provides|operates)", content_lower)
        if ctx_match:
            return ctx_match.group(1)

        return "agricultural_equipment"

    # ── Model extraction ──────────────────────────────────────────────────────

    def extract_model_universal(self, text: str) -> Optional[str]:
        patterns = [
            r"Model[:\s]+([A-Z0-9\-\/\s]{2,20})",
            r"Model\s+No[.:\s]+([A-Z0-9\-\/\s]{2,20})",
            r"\b([A-Z]{2,4}\s*\d{2,4}[A-Z]*)\b",
            r"\b(\d{3,4}\s*[A-Z]{1,3})\b",
            r"\b([A-Z]+[-/]\d+[-/]?[A-Z]*)\b",
            r"Type[:\s]+([A-Z0-9\-\/\s]{2,20})",
        ]
        for pattern in patterns:
            for match in re.findall(pattern, text, re.IGNORECASE):
                clean = match.strip()
                if 2 <= len(clean) <= 20:
                    return clean
        return None

    # ── Misc metadata ─────────────────────────────────────────────────────────

    def extract_metadata_from_content(self, content: str, file_path: Path) -> Dict:
        meta: Dict = {}
        year_match = re.search(r"\b(19|20)\d{2}\b", content)
        if year_match:
            meta["year"] = year_match.group()
        hindi_chars = re.findall(r"[\u0900-\u097F]", content)
        meta["language"] = "hindi" if len(hindi_chars) > 10 else "english"
        return meta

    def calculate_content_hash(self, content: str) -> str:
        """Normalize whitespace before hashing to catch near-duplicate chunks."""
        normalized = re.sub(r"\s+", " ", content).strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    # ── Document loaders ──────────────────────────────────────────────────────

    def _build_doc_metadata(
        self,
        docs: List[Document],
        file_path: Path,
        file_type: str,
    ) -> List[Document]:
        """Attach universal metadata to every raw page/document."""
        full_text = " ".join(d.page_content for d in docs)

        brands = self.extract_brands_universal(full_text)
        machine_type = self.extract_machine_type_universal(file_path.name, full_text)
        model = self.extract_model_universal(full_text)
        extra_meta = self.extract_metadata_from_content(full_text, file_path)
        # Returns list — stored as list, not CSV
        problem_categories = self.extract_problem_keywords(full_text, file_path.name)

        self.discovered_brands.update(brands)
        self.discovered_machines.add(machine_type)
        self.discovered_problems.update(problem_categories)

        primary_brand = (
            next(iter(brands), None) or self.extract_brand_from_filename(file_path.name) or "Unknown"
        )

        for doc in docs:
            doc.metadata.update(
                {
                    "source_file": file_path.name,
                    "file_path": str(file_path),
                    "brand": primary_brand,
                    "all_brands": ",".join(sorted(brands)),
                    "model": model,
                    "machine_type": machine_type,
                    # ★ Stored as a list for ChromaDB $contains filter support
                    "problem_categories": problem_categories,
                    "content_hash": self.calculate_content_hash(doc.page_content),
                    "processing_date": datetime.now().isoformat(),
                    "total_pages": len(docs),
                    "file_type": file_type,
                    "is_troubleshooting": any(
                        w in doc.page_content.lower()
                        for w in ["problem", "symptom", "solution", "fix", "repair", "troubleshoot"]
                    ),
                    # section_type will be overwritten by SemanticManualChunker
                    "section_type": "general",
                }
            )
            doc.metadata.update(extra_meta)
        return docs

    def load_pdf_document(self, file_path: Path) -> Tuple[List[Document], Optional[str]]:
        try:
            try:
                docs = PyPDFLoader(str(file_path)).load()
            except Exception as e:
                logger.warning(f"PyPDFLoader failed, trying Unstructured: {e}")
                docs = UnstructuredPDFLoader(str(file_path)).load()
            return self._build_doc_metadata(docs, file_path, "pdf"), None
        except Exception as e:
            return [], f"Failed to load PDF {file_path}: {e}"

    def load_txt_document(self, file_path: Path) -> Tuple[List[Document], Optional[str]]:
        try:
            docs = TextLoader(str(file_path), encoding="utf-8", autodetect_encoding=True).load()
            return self._build_doc_metadata(docs, file_path, "txt"), None
        except Exception as e:
            return [], f"Failed to load TXT {file_path}: {e}"

    # ── Directory scanner ─────────────────────────────────────────────────────

    def scan_knowledge_directory(self) -> Dict:
        pdf_files = list(self.knowledge_dir.rglob("*.pdf"))
        txt_files = list(self.knowledge_dir.rglob("*.txt"))
        return {
            "total_pdfs": len(pdf_files),
            "total_txt": len(txt_files),
            "all_files": pdf_files + txt_files,
        }

    # ── Main build ────────────────────────────────────────────────────────────

    def build_knowledge_base(self) -> bool:
        try:
            api_key = self.load_environment()
            logger.info("=== Starting Knowledge Base Build (Optimized) ===")
            logger.info(f"Knowledge dir : {self.knowledge_dir}")
            logger.info(f"DB dir        : {self.db_dir}")
            scan = self.scan_knowledge_directory()
            if not scan["all_files"]:
                logger.error("No files found in knowledge directory!")
                return False

            logger.info(f"Found {scan['total_pdfs']} PDF(s) + {scan['total_txt']} TXT(s)")

            all_documents: List[Document] = []
            seen_hashes: Set[str] = set()

            for file_path in scan["all_files"]:
                loader_fn = (
                    self.load_pdf_document
                    if file_path.suffix.lower() == ".pdf"
                    else self.load_txt_document
                )
                docs, error = loader_fn(file_path)
                if error:
                    self.failed_files.append((file_path, error))
                    logger.error(error)
                    continue

                for doc in docs:
                    h = doc.metadata["content_hash"]
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        all_documents.append(doc)

                logger.info(f"  ✓ {file_path.name} → {len(docs)} page(s)")

            if not all_documents:
                logger.error("No documents were successfully processed!")
                return False

            logger.info(f"\n📊 Auto-discovered:")
            logger.info(f"   Brands  : {sorted(self.discovered_brands)}")
            logger.info(f"   Machines: {sorted(self.discovered_machines)}")
            logger.info(f"   Problems: {sorted(self.discovered_problems)}\n")

            # ── Structured semantic chunking ──────────────────────────────────
            logger.info("Running structured semantic chunker...")
            chunker = SemanticManualChunker()
            chunks = chunker.split_documents(all_documents)

            # Final deduplication at chunk level
            final_chunks: List[Document] = []
            chunk_hashes: Set[str] = set()
            for chunk in chunks:
                h = chunk.metadata["content_hash"]
                if h not in chunk_hashes:
                    chunk_hashes.add(h)
                    final_chunks.append(chunk)

            logger.info(f"Created {len(final_chunks)} unique chunks")

            # ── Embed + store ─────────────────────────────────────────────────
            logger.info("Creating vector database...")
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=api_key,
                client_options={"api_endpoint": "generativelanguage.googleapis.com"},
            )

            if self.db_dir.exists():
                logger.info("Clearing existing database...")
                shutil.rmtree(self.db_dir)

            db = Chroma.from_documents(
                documents=final_chunks,
                embedding=embeddings,
                persist_directory=str(self.db_dir),
            )

            # ★ Use public API instead of db._collection.count()
            stored_ids = db.get()["ids"]
            logger.info(f"✅ Database created — {len(stored_ids)} chunks stored")

            self._generate_build_report(all_documents, final_chunks)
            logger.info("=== Knowledge Base Build Completed ===")
            return True

        except Exception as e:
            logger.error(f"Knowledge base build failed: {e}", exc_info=True)
            return False

    # ── Build report ──────────────────────────────────────────────────────────

    def _generate_build_report(self, documents: List[Document], chunks: List[Document]):
        section_type_counts: Dict[str, int] = {}
        machine_counts: Dict[str, int] = {}
        brand_counts: Dict[str, int] = {}

        for chunk in chunks:
            st = chunk.metadata.get("section_type", "general")
            section_type_counts[st] = section_type_counts.get(st, 0) + 1
            mt = chunk.metadata.get("machine_type", "unknown")
            machine_counts[mt] = machine_counts.get(mt, 0) + 1
            br = chunk.metadata.get("brand", "Unknown")
            brand_counts[br] = brand_counts.get(br, 0) + 1

        report = {
            "build_date": datetime.now().isoformat(),
            "total_source_documents": len(documents),
            "total_chunks": len(chunks),
            "section_type_distribution": section_type_counts,
            "machine_type_distribution": machine_counts,
            "brand_distribution": brand_counts,
            "failed_files": [(str(f), str(e)) for f, e in self.failed_files],
            "discovered_brands": sorted(self.discovered_brands),
            "discovered_machines": sorted(self.discovered_machines),
            "discovered_problems": sorted(self.discovered_problems),
            "avg_chunk_chars": (
                sum(len(c.page_content) for c in chunks) / len(chunks) if chunks else 0
            ),
        }

        report_path = self.db_dir / "build_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Build report → {report_path}")
        logger.info(f"Section types: {section_type_counts}")
        if self.failed_files:
            logger.warning(f"Failed files: {len(self.failed_files)}")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    builder = KnowledgeBaseBuilder(
        knowledge_dir="./knowledge_base",
        db_dir="./chroma_db",
    )
    success = builder.build_knowledge_base()
    if success:
        print("\n✅ Knowledge base built successfully!")
        print("   Check chroma_db/build_report.json for details.")
        return 0
    else:
        print("\n❌ Knowledge base build failed!")
        print("   Check knowledge_build.log for details.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())