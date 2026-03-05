import os
import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
import logging
from datetime import datetime

# LangChain imports
from langchain_community.document_loaders import (
    PyPDFLoader, 
    UnstructuredPDFLoader,
    TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

# Configure logging — force UTF-8 on Windows to handle emoji/checkmarks
import sys as _sys
_file_handler = logging.FileHandler('knowledge_build.log', encoding='utf-8')
try:
    _stream_handler = logging.StreamHandler(
        stream=open(_sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
    )
except Exception:
    _stream_handler = logging.StreamHandler()
    _stream_handler.stream = _sys.stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[_file_handler, _stream_handler]
)
logger = logging.getLogger(__name__)

class KnowledgeBaseBuilder:
    """Universal knowledge base builder - Auto-detects ANY agricultural machine"""
    
    def __init__(self, knowledge_dir: str = "./knowledge_base", db_dir: str = "./chroma_db"):
        self.knowledge_dir = Path(knowledge_dir)
        self.db_dir = Path(db_dir)
        self.documents = []
        self.failed_files = []
        
        # Auto-discovered brands and machines during processing
        self.discovered_brands = set()
        self.discovered_machines = set()
        self.discovered_problems = set()

    def extract_problem_keywords(self, content: str, filename: str) -> List[str]:
        """Universal problem keyword extraction - auto-detects categories"""
        keywords = set()
        content_lower = content.lower()
        filename_lower = filename.lower()
        
        # Universal problem indicators (common across ALL machinery)
        problem_indicators = {
            'not_starting': ['not start', 'won\'t start', 'no start', 'dead', 'not working'],
            'noise': ['noise', 'sound', 'knocking', 'clicking', 'grinding', 'rattling', 'squealing', 'humming'],
            'leaking': ['leak', 'leaking', 'dripping', 'seeping', 'oil leak', 'water leak', 'fuel leak'],
            'overheating': ['hot', 'overheat', 'temperature', 'boiling', 'heat'],
            'vibration': ['vibrat', 'shaking', 'wobbl', 'unsteady'],
            'smoke': ['smoke', 'smoking', 'fumes', 'exhaust'],
            'power_loss': ['weak', 'low power', 'no power', 'poor performance', 'sluggish'],
            'electrical': ['battery', 'starter', 'wiring', 'electrical', 'fuse', 'spark'],
            'fuel': ['fuel', 'diesel', 'petrol', 'gas', 'carburetor', 'injector'],
            'hydraulic': ['hydraulic', 'oil pressure', 'lift'],
            'mechanical': ['gear', 'clutch', 'belt', 'chain', 'bearing'],
            'water': ['water', 'pump', 'flow', 'pressure', 'discharge'],
            'cooling': ['radiator', 'coolant', 'cooling'],
            'engine': ['engine', 'motor', 'piston', 'cylinder']
        }
        
        # Check filename and content for problem indicators
        combined_text = f"{filename_lower} {content_lower[:2000]}"
        
        for category, terms in problem_indicators.items():
            if any(term in combined_text for term in terms):
                keywords.add(category)
        
        return list(keywords)

    def extract_brands_universal(self, text: str) -> Set[str]:
        """Auto-detect brand names from text (any brand, not just predefined)"""
        brands = set()
        
        # Common Indian/International brand patterns
        # Pattern 1: Brand + Model pattern (e.g., "Mahindra 575", "Kirloskar KP4")
        brand_model_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:\d+|[A-Z]{2,})'
        matches = re.findall(brand_model_pattern, text)
        brands.update(matches)
        
        # Pattern 2: "Brand" + Keyword (e.g., "Mahindra Tractor", "Kirloskar Pump")
        brand_keyword_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:Tractor|Pump|Motor|Engine|Harvester|Cultivator|Plough)'
        matches = re.findall(brand_keyword_pattern, text, re.IGNORECASE)
        brands.update([m.title() for m in matches])
        
        # Pattern 3: Capitalized words that appear multiple times (likely brands)
        capitalized_words = re.findall(r'\b([A-Z][a-z]{3,})\b', text)
        word_freq = {}
        for word in capitalized_words:
            if word not in ['The', 'This', 'That', 'When', 'Where', 'What', 'Which']:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Add words that appear 3+ times (likely important brands/models)
        for word, freq in word_freq.items():
            if freq >= 3:
                brands.add(word)
        
        return brands

    def extract_machine_type_universal(self, filename: str, content: str) -> str:
        """Universal machine type detection - works for ANY agricultural equipment"""
        filename_lower = filename.lower()
        content_lower = content.lower()[:2000]
        
        # Universal machine keywords (expanded dynamically)
        machine_keywords = [
            'tractor', 'pump', 'motor', 'engine', 'harvester', 'combine',
            'thresher', 'cultivator', 'plough', 'plow', 'seeder', 'drill',
            'sprayer', 'spreader', 'trailer', 'rotavator', 'tiller',
            'weeder', 'reaper', 'baler', 'loader', 'digger',
            'submersible', 'centrifugal', 'generator', 'alternator'
        ]
        
        # Check filename first (most reliable)
        for keyword in machine_keywords:
            if keyword in filename_lower:
                return keyword
        
        # Check content
        for keyword in machine_keywords:
            if keyword in content_lower:
                return keyword
        
        # Check directory structure (e.g., "knowledge_base/Pumps/file.pdf")
        parts = Path(filename).parts
        for part in parts:
            part_lower = part.lower()
            for keyword in machine_keywords:
                if keyword in part_lower:
                    return keyword
        
        # Try to extract from context (look for "This [machine] is...")
        context_pattern = r'(?:this|the)\s+([a-z]+)\s+(?:is|has|provides|operates)'
        matches = re.findall(context_pattern, content_lower)
        if matches:
            return matches[0]
        
        return "agricultural_equipment"  # Generic fallback

    def extract_model_universal(self, text: str) -> Optional[str]:
        """Universal model number extraction - works for any format"""
        # Model patterns (very flexible)
        patterns = [
            r'Model[:\s]+([A-Z0-9\-\/\s]{2,20})',  # Model: ABC-123
            r'Model\s+No[.:\s]+([A-Z0-9\-\/\s]{2,20})',  # Model No. 123
            r'\b([A-Z]{2,4}\s*\d{2,4}[A-Z]*)\b',  # AB 1234, XYZ123
            r'\b(\d{3,4}\s*[A-Z]{1,3})\b',  # 575 DI, 1234X
            r'\b([A-Z]+[-/]\d+[-/]?[A-Z]*)\b',  # ABC-123, XY/456/Z
            r'Type[:\s]+([A-Z0-9\-\/\s]{2,20})',  # Type: 123
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Return first reasonable match (length 2-20 chars)
                for match in matches:
                    clean_match = match.strip()
                    if 2 <= len(clean_match) <= 20:
                        return clean_match
        
        return None

    def load_environment(self) -> str:
        """Load and validate environment variables"""
        load_dotenv()
        api_key = os.getenv("GOOGLE_AI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_AI_API_KEY not found in .env file")
        return api_key

    def extract_metadata_from_content(self, content: str, file_path: Path) -> Dict:
        """Universal metadata extraction"""
        metadata = {}
        
        # Extract model
        model = self.extract_model_universal(content)
        if model:
            metadata['model_reference'] = model
        
        # Extract year
        year_match = re.search(r'\b(19|20)\d{2}\b', content)
        if year_match:
            metadata['year'] = year_match.group()
        
        # Detect language
        hindi_chars = re.findall(r'[\u0900-\u097F]', content)
        if len(hindi_chars) > 10:
            metadata['language'] = 'hindi'
        else:
            metadata['language'] = 'english'
        
        return metadata

    def calculate_content_hash(self, content: str) -> str:
        """Calculate hash of content for duplicate detection"""
        return hashlib.md5(content.encode()).hexdigest()

    def load_pdf_document(self, file_path: Path) -> Tuple[List[Document], Optional[str]]:
        """Load PDF document with universal metadata extraction"""
        try:
            try:
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()
            except Exception as e:
                logger.warning(f"PyPDFLoader failed, trying UnstructuredPDFLoader: {e}")
                loader = UnstructuredPDFLoader(str(file_path))
                docs = loader.load()
            
            full_text = " ".join([doc.page_content for doc in docs])
            
            # Universal extraction
            brands = self.extract_brands_universal(full_text)
            machine_type = self.extract_machine_type_universal(file_path.name, full_text)
            model = self.extract_model_universal(full_text)
            content_metadata = self.extract_metadata_from_content(full_text, file_path)
            problem_keywords = self.extract_problem_keywords(full_text, file_path.name)
            
            # Track discoveries
            self.discovered_brands.update(brands)
            self.discovered_machines.add(machine_type)
            self.discovered_problems.update(problem_keywords)
            
            # Primary brand (first discovered or from filename)
            primary_brand = list(brands)[0] if brands else self.extract_brand_from_filename(file_path.name)
            
            for doc in docs:
                doc.metadata.update({
                    "source_file": file_path.name,
                    "file_path": str(file_path),
                    "brand": primary_brand or "Unknown",
                    "all_brands": ','.join(brands) if brands else "",
                    "model": model,
                    "machine_type": machine_type,
                    "problem_categories": ','.join(problem_keywords),
                    "content_hash": self.calculate_content_hash(doc.page_content),
                    "processing_date": datetime.now().isoformat(),
                    "total_pages": len(docs),
                    "file_type": "pdf",
                    "is_troubleshooting": any(word in doc.page_content.lower() 
                                            for word in ['problem', 'symptom', 'solution', 'fix', 'repair', 'troubleshoot'])
                })
                doc.metadata.update(content_metadata)
            
            return docs, None
            
        except Exception as e:
            error_msg = f"Failed to load PDF {file_path}: {e}"
            return [], error_msg

    def load_txt_document(self, file_path: Path) -> Tuple[List[Document], Optional[str]]:
        """Load TXT document with universal metadata extraction"""
        try:
            loader = TextLoader(str(file_path), encoding="utf-8", autodetect_encoding=True)
            docs = loader.load()
            
            full_text = " ".join([doc.page_content for doc in docs])
            
            # Universal extraction
            brands = self.extract_brands_universal(full_text)
            machine_type = self.extract_machine_type_universal(file_path.name, full_text)
            model = self.extract_model_universal(full_text)
            content_metadata = self.extract_metadata_from_content(full_text, file_path)
            problem_keywords = self.extract_problem_keywords(full_text, file_path.name)
            
            # Track discoveries
            self.discovered_brands.update(brands)
            self.discovered_machines.add(machine_type)
            self.discovered_problems.update(problem_keywords)
            
            primary_brand = list(brands)[0] if brands else self.extract_brand_from_filename(file_path.name)
            
            for doc in docs:
                doc.metadata.update({
                    "source_file": file_path.name,
                    "file_path": str(file_path),
                    "brand": primary_brand or "Unknown",
                    "all_brands": ','.join(brands) if brands else "",
                    "model": model,
                    "machine_type": machine_type,
                    "problem_categories": ','.join(problem_keywords),
                    "content_hash": self.calculate_content_hash(doc.page_content),
                    "processing_date": datetime.now().isoformat(),
                    "total_pages": 1,
                    "file_type": "txt",
                    "content_type": "troubleshooting_guide",
                    "is_troubleshooting": any(word in doc.page_content.lower() 
                                            for word in ['problem', 'symptom', 'solution', 'fix', 'repair', 'troubleshoot'])
                })
                doc.metadata.update(content_metadata)
            
            return docs, None
            
        except Exception as e:
            error_msg = f"Failed to load TXT {file_path}: {e}"
            return [], error_msg

    def extract_brand_from_filename(self, filename: str) -> Optional[str]:
        """Extract brand from filename using common patterns"""
        # Try to find capitalized words in filename
        filename_clean = filename.replace('_', ' ').replace('-', ' ')
        words = filename_clean.split()
        
        for word in words:
            if word[0].isupper() and len(word) > 3:
                return word
        
        return None

    def scan_knowledge_directory(self) -> Dict:
        """Scan knowledge directory"""
        pdf_files = list(self.knowledge_dir.rglob("*.pdf"))
        txt_files = list(self.knowledge_dir.rglob("*.txt"))
        
        return {
            "total_pdfs": len(pdf_files),
            "total_txt": len(txt_files),
            "all_files": pdf_files + txt_files
        }

    def build_knowledge_base(self) -> bool:
        """Main method to build the knowledge base"""
        try:
            api_key = self.load_environment()
            
            logger.info("=== Starting UNIVERSAL Knowledge Base Build ===")
            logger.info(f"Knowledge directory: {self.knowledge_dir}")
            logger.info(f"Database directory: {self.db_dir}")
            
            scan_results = self.scan_knowledge_directory()
            
            if not scan_results["all_files"]:
                logger.error("No files found in knowledge directory!")
                return False
            
            logger.info(f"Found {scan_results['total_pdfs']} PDF files and {scan_results['total_txt']} TXT files")
            
            all_documents = []
            unique_content_hashes = set()
            
            for file_path in scan_results["all_files"]:
                if file_path.suffix.lower() == '.pdf':
                    docs, error = self.load_pdf_document(file_path)
                    
                    if error:
                        self.failed_files.append((file_path, error))
                        logger.error(error)
                    else:
                        for doc in docs:
                            if doc.metadata["content_hash"] not in unique_content_hashes:
                                unique_content_hashes.add(doc.metadata["content_hash"])
                                all_documents.append(doc)
                        
                        logger.info(f"✓ Processed PDF {file_path.name} -> {len(docs)} pages")
                
                elif file_path.suffix.lower() == '.txt':
                    docs, error = self.load_txt_document(file_path)
                    
                    if error:
                        self.failed_files.append((file_path, error))
                        logger.error(error)
                    else:
                        for doc in docs:
                            if doc.metadata["content_hash"] not in unique_content_hashes:
                                unique_content_hashes.add(doc.metadata["content_hash"])
                                all_documents.append(doc)
                        
                        logger.info(f"✓ Processed TXT {file_path.name} -> {len(docs)} documents")
            
            if not all_documents:
                logger.error("No documents were successfully processed!")
                return False
            
            logger.info(f"\n🔍 AUTO-DISCOVERED:")
            logger.info(f"  Brands: {sorted(self.discovered_brands)}")
            logger.info(f"  Machines: {sorted(self.discovered_machines)}")
            logger.info(f"  Problem Types: {sorted(self.discovered_problems)}\n")
            
            # Split documents into chunks
            logger.info(f"Splitting {len(all_documents)} documents into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )
            
            chunks = text_splitter.split_documents(all_documents)
            logger.info(f"Created {len(chunks)} searchable chunks")
            
            # Create vector store
            logger.info("Creating vector database...")
            # Use api_version="v1" so text-embedding-004 is found
            # (langchain-google-genai 2.0 defaults to v1beta which doesn't have this model)
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=api_key,
                client_options={"api_endpoint": "generativelanguage.googleapis.com"}
            )
            
            if self.db_dir.exists():
                logger.info("Clearing existing database...")
                import shutil
                shutil.rmtree(self.db_dir)
            
            db = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=str(self.db_dir)
            )
            
            collection_count = db._collection.count()
            logger.info(f"Database created successfully! Contains {collection_count} chunks.")
            
            self.generate_build_report(all_documents, chunks)
            
            logger.info("=== Knowledge Base Build Completed Successfully ===")
            return True
            
        except Exception as e:
            logger.error(f"Knowledge base build failed: {e}")
            return False

    def generate_build_report(self, documents: List[Document], chunks: List[Document]):
        """Generate a detailed build report"""
        report = {
            "build_date": datetime.now().isoformat(),
            "total_documents": len(documents),
            "total_chunks": len(chunks),
            "failed_files": [(str(f[0]), str(f[1])) for f in self.failed_files],
            "discovered_brands": sorted(list(self.discovered_brands)),
            "discovered_machines": sorted(list(self.discovered_machines)),
            "discovered_problems": sorted(list(self.discovered_problems)),
            "brands_processed": {},
            "machine_types_processed": {},
            "problem_categories_processed": {},
            "statistics": {
                "avg_chunk_size": sum(len(chunk.page_content) for chunk in chunks) / len(chunks) if chunks else 0,
                "total_content_size": sum(len(doc.page_content) for doc in documents)
            }
        }
        
        for doc in documents:
            brand = doc.metadata.get("brand", "Unknown")
            machine_type = doc.metadata.get("machine_type", "unknown")
            problem_cats = doc.metadata.get("problem_categories", "").split(',')
            
            report["brands_processed"][brand] = report["brands_processed"].get(brand, 0) + 1
            report["machine_types_processed"][machine_type] = report["machine_types_processed"].get(machine_type, 0) + 1
            
            for cat in problem_cats:
                if cat:
                    report["problem_categories_processed"][cat] = report["problem_categories_processed"].get(cat, 0) + 1
        
        report_path = self.db_dir / "build_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Build report saved to: {report_path}")
        
        logger.info("\n=== BUILD SUMMARY ===")
        logger.info(f"Total documents: {len(documents)}")
        logger.info(f"Total chunks: {len(chunks)}")
        logger.info(f"Auto-discovered brands: {report['discovered_brands']}")
        logger.info(f"Auto-discovered machines: {report['discovered_machines']}")
        logger.info(f"Auto-discovered problems: {report['discovered_problems'][:10]}...")  # Show first 10
        if self.failed_files:
            logger.warning(f"Failed files: {len(self.failed_files)}")

def main():
    """Main execution function"""
    builder = KnowledgeBaseBuilder(
        knowledge_dir="./knowledge_base",
        db_dir="./chroma_db"
    )
    
    success = builder.build_knowledge_base()
    
    if success:
        print("\n✅ Knowledge base built successfully!")
        print("   The system auto-detected all brands, machines, and problem types.")
        print("   Check build_report.json for details.")
        return 0
    else:
        print("\n❌ Knowledge base build failed!")
        print("   Check knowledge_build.log for details.")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())