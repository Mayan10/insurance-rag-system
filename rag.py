import json
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import pickle
import logging
from pathlib import Path

# External libraries (install via pip)
# Initialize variables to None to avoid NameError
torch = None
AutoTokenizer = None
AutoModel = None
pipeline = None
SentenceTransformer = None
faiss = None
cosine_similarity = None
spacy = None
RecursiveCharacterTextSplitter = None
Document = None
PyPDF2 = None
DocxDocument = None
email = None
policy = None
fitz = None

try:
    import torch
    from transformers import AutoTokenizer, AutoModel, pipeline
    from sentence_transformers import SentenceTransformer
    import faiss
    from sklearn.metrics.pairwise import cosine_similarity
    import spacy
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.docstore.document import Document
    import PyPDF2
    from docx import Document as DocxDocument
    import email
    from email import policy
    import fitz  # PyMuPDF for better PDF processing
except ImportError as e:
    print(f"Missing dependencies. Install with: pip install {e.name}")
    print("Some features may not work without these dependencies.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QueryStructured:
    """Structured representation of parsed query"""
    age: Optional[int] = None
    gender: Optional[str] = None
    procedure: Optional[str] = None
    location: Optional[str] = None
    policy_duration: Optional[str] = None
    policy_age_months: Optional[int] = None
    raw_query: str = ""
    extracted_entities: Dict[str, Any] = None

@dataclass
class RetrievedClause:
    """Retrieved document clause with metadata"""
    content: str
    source_document: str
    page_number: Optional[int] = None
    section: Optional[str] = None
    relevance_score: float = 0.0
    clause_type: Optional[str] = None

@dataclass
class DecisionResponse:
    """Final decision response structure"""
    decision: str  # "approved", "rejected", "requires_review"
    amount: Optional[float] = None
    confidence: float = 0.0
    justification: str = ""
    relevant_clauses: List[RetrievedClause] = None
    reasoning_chain: List[str] = None

class AdvancedQueryParser:
    """Advanced NLP-based query parser using multiple techniques"""
    
    def __init__(self):
        # Load spaCy model for NER
        self.nlp = None
        if spacy is not None:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            except Exception as e:
                logger.warning(f"spaCy not available: {e}")
        else:
            logger.warning("spaCy not installed. Install with: pip install spacy")
        
        # Initialize medical NER pipeline
        self.medical_ner = None
        if pipeline is not None:
            try:
                self.medical_ner = pipeline("ner", 
                                          model="d4data/biomedical-ner-all",
                                          aggregation_strategy="simple")
            except Exception as e:
                logger.warning(f"Medical NER model not available: {e}")
        else:
            logger.warning("Transformers pipeline not available. Install with: pip install transformers")
    
    def parse_query(self, query: str) -> QueryStructured:
        """Parse natural language query into structured format"""
        structured = QueryStructured(raw_query=query)
        
        # Basic regex patterns for extracting information
        age_pattern = r'(\d+)[-\s]*(?:year|yr|y)?\s*(?:old)?(?:\s*(?:male|female|M|F))?'
        gender_pattern = r'\b(?:male|female|M|F|man|woman)\b'
        location_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        policy_duration_pattern = r'(\d+)[-\s]*(?:month|mon|m|year|yr|y)[-\s]*(?:old)?\s*(?:policy|insurance)'
        
        # Extract age from query
        age_match = re.search(age_pattern, query, re.IGNORECASE)
        if age_match:
            structured.age = int(age_match.group(1))
        
        # Extract gender information
        gender_match = re.search(gender_pattern, query, re.IGNORECASE)
        if gender_match:
            gender = gender_match.group().lower()
            structured.gender = 'male' if gender in ['male', 'm', 'man'] else 'female'
        else:
            # Check for single letter gender indicators
            if re.search(r'\bM\b', query):
                structured.gender = 'male'
            elif re.search(r'\bF\b', query):
                structured.gender = 'female'
        
        # Extract policy duration information
        policy_match = re.search(policy_duration_pattern, query, re.IGNORECASE)
        if policy_match:
            duration = int(policy_match.group(1))
            structured.policy_age_months = duration
            structured.policy_duration = f"{duration} months"
        
        # Use spaCy for advanced entity extraction
        if self.nlp:
            doc = self.nlp(query)
            entities = {}
            for ent in doc.ents:
                entities[ent.label_] = ent.text
            structured.extracted_entities = entities
        
        # Use medical NER for procedure extraction
        if self.medical_ner:
            try:
                medical_entities = self.medical_ner(query)
                for entity in medical_entities:
                    if entity['entity_group'] in ['TREATMENT', 'PROCEDURE']:
                        structured.procedure = entity['word']
                        break
            except Exception as e:
                logger.warning(f"Medical NER failed: {e}")
        
        # Fallback procedure extraction using pattern matching
        if not structured.procedure:
            medical_terms = ['surgery', 'operation', 'procedure', 'treatment', 'therapy']
            query_lower = query.lower()
            for term in medical_terms:
                if term in query_lower:
                    # Extract surrounding context for better procedure identification
                    words = query_lower.split()
                    try:
                        idx = words.index(term)
                        if idx > 0:
                            # Try to get more context (2 words before)
                            if idx > 1:
                                structured.procedure = f"{words[idx-2]} {words[idx-1]} {term}"
                            else:
                                structured.procedure = f"{words[idx-1]} {term}"
                        else:
                            structured.procedure = term
                    except ValueError:
                        # If exact word not found, use the term as is
                        structured.procedure = term
                    break
        
        # Extract location information (improved)
        words = query.split()
        potential_locations = []
        for word in words:
            if word[0].isupper() and len(word) > 2 and not any(char.isdigit() for char in word):
                potential_locations.append(word)
        
        if potential_locations:
            structured.location = potential_locations[0]  # Take first capitalized word
        
        return structured

class HybridEmbeddingRetriever:
    """Advanced retrieval system using multiple embedding models and techniques"""
    
    def __init__(self, embedding_models: List[str] = None):
        if embedding_models is None:
            # Use only one model to reduce memory usage
            embedding_models = [
                'all-MiniLM-L6-v2'  # Fast and efficient - single model
            ]
        
        self.embedding_models = {}
        self.vector_stores = {}
        
        # Load embedding models with memory optimization
        if SentenceTransformer is not None:
            for model_name in embedding_models:
                try:
                    # Use device='cpu' to avoid GPU memory issues
                    self.embedding_models[model_name] = SentenceTransformer(model_name, device='cpu')
                    logger.info(f"Loaded embedding model: {model_name}")
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {e}")
        else:
            logger.warning("SentenceTransformer not available. Install with: pip install sentence-transformers")
        
        # Text splitter for chunking documents
        self.text_splitter = None
        if RecursiveCharacterTextSplitter is not None:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=512,
                chunk_overlap=50,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
        else:
            logger.warning("RecursiveCharacterTextSplitter not available. Install with: pip install langchain")
        
        self.documents = []
        self.chunks = []
        self.chunk_embeddings = {}
    
    def load_documents(self, document_paths: List[str]):
        """Load and process documents from various formats with enhanced PDF processing"""
        if self.text_splitter is None:
            logger.error("Text splitter not available. Cannot load documents.")
            return
            
        for path in document_paths:
            path_obj = Path(path)
            
            if path_obj.suffix.lower() == '.pdf':
                # Enhanced PDF processing for large documents
                self._extract_pdf_content_enhanced(path)
            elif path_obj.suffix.lower() in ['.docx', '.doc']:
                content = self._extract_docx_content(path)
                self._process_document_content(content, path)
            elif path_obj.suffix.lower() == '.txt':
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self._process_document_content(content, path)
            else:
                logger.warning(f"Unsupported file format: {path}")
                continue
        
        logger.info(f"Loaded {len(self.documents)} documents, {len(self.chunks)} chunks")
        
        # Create embeddings for all chunks
        self._create_embeddings()
    
    def _process_document_content(self, content: str, path: str):
        """Process document content and create chunks"""
        if Document is not None:
            doc = Document(page_content=content, metadata={'source': path})
            self.documents.append(doc)
            
            # Split into chunks
            chunks = self.text_splitter.split_documents([doc])
            self.chunks.extend(chunks)
        else:
            logger.error("Document class not available. Cannot process documents.")
    
    def _extract_pdf_content_enhanced(self, path: str):
        """Enhanced PDF extraction that processes all pages"""
        if fitz is None:
            logger.error("PyMuPDF not available. Install with: pip install PyMuPDF")
            return
            
        try:
            doc = fitz.open(path)
            total_pages = len(doc)
            
            logger.info(f"Processing PDF with {total_pages} pages...")
            
            # Process each page individually for better chunking
            for page_num in range(total_pages):
                page = doc[page_num]
                text = page.get_text()
                
                if text.strip():  # Only process non-empty pages
                    # Create document for this page
                    if Document is not None:
                        page_doc = Document(
                            page_content=text, 
                            metadata={
                                'source': path,
                                'page_number': page_num + 1,
                                'total_pages': total_pages
                            }
                        )
                        self.documents.append(page_doc)
                        
                        # Split page into chunks
                        page_chunks = self.text_splitter.split_documents([page_doc])
                        self.chunks.extend(page_chunks)
                        
                        # Show progress for large documents
                        if total_pages > 20:
                            if (page_num + 1) % 10 == 0 or page_num + 1 == total_pages:
                                logger.info(f"   Processed {page_num + 1}/{total_pages} pages ({len(self.chunks)} chunks)")
                        else:
                            logger.info(f"   Page {page_num + 1}: {len(text)} characters, {len(page_chunks)} chunks")
            
            doc.close()
            logger.info(f"PDF processing completed: {len(self.documents)} page documents, {len(self.chunks)} total chunks")
            
        except Exception as e:
            logger.error(f"Failed to extract PDF content: {e}")
            return
    
    def _extract_pdf_content(self, path: str) -> str:
        """Extract text from PDF using PyMuPDF"""
        if fitz is None:
            logger.error("PyMuPDF not available. Install with: pip install PyMuPDF")
            return ""
            
        try:
            doc = fitz.open(path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Failed to extract PDF content: {e}")
            return ""
    
    def _extract_docx_content(self, path: str) -> str:
        """Extract text from DOCX"""
        if DocxDocument is None:
            logger.error("python-docx not available. Install with: pip install python-docx")
            return ""
            
        try:
            doc = DocxDocument(path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Failed to extract DOCX content: {e}")
            return ""
    
    def _create_embeddings(self):
        """Create embeddings for all chunks using multiple models with memory optimization"""
        if not self.embedding_models:
            logger.warning("No embedding models available. Cannot create embeddings.")
            return
            
        if faiss is None:
            logger.warning("FAISS not available. Install with: pip install faiss-cpu")
            return
            
        chunk_texts = [chunk.page_content for chunk in self.chunks]
        total_chunks = len(chunk_texts)
        
        logger.info(f"Creating embeddings for {total_chunks} chunks...")
        
        for model_name, model in self.embedding_models.items():
            try:
                logger.info(f"Processing model: {model_name}")
                
                # Process in batches for large documents
                batch_size = 50  # Process 50 chunks at a time
                all_embeddings = []
                
                for i in range(0, total_chunks, batch_size):
                    batch = chunk_texts[i:i + batch_size]
                    batch_embeddings = model.encode(batch, show_progress_bar=False)
                    all_embeddings.append(batch_embeddings)
                    
                    # Show progress
                    if total_chunks > 100:
                        progress = min((i + batch_size) / total_chunks * 100, 100)
                        logger.info(f"   Progress: {progress:.1f}% ({i + len(batch)}/{total_chunks} chunks)")
                
                # Combine all embeddings
                embeddings = np.vstack(all_embeddings)
                
                # Create FAISS index
                dimension = embeddings.shape[1]
                index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
                
                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(embeddings)
                index.add(embeddings.astype(np.float32))
                
                self.vector_stores[model_name] = {
                    'index': index,
                    'embeddings': embeddings
                }
                
                logger.info(f"Created FAISS index for {model_name} with {total_chunks} chunks")
                
            except Exception as e:
                logger.error(f"Failed to create embeddings for {model_name}: {e}")
                # Continue with other models if one fails
                continue
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 10) -> List[RetrievedClause]:
        """Retrieve relevant chunks using ensemble of embedding models"""
        if not self.embedding_models:
            logger.warning("No embedding models available. Returning empty results.")
            return []
            
        if not self.vector_stores:
            logger.warning("No vector stores available. Documents may not be loaded. Returning empty results.")
            return []
            
        if faiss is None:
            logger.warning("FAISS not available. Cannot perform similarity search.")
            return []
            
        all_results = []
        
        # Enhanced query processing
        query_terms = query.lower().split()
        medical_terms = ['surgery', 'treatment', 'procedure', 'knee', 'dental', 'cardiac', 'heart']
        query_medical_terms = [term for term in query_terms if term in medical_terms]
        
        for model_name, model in self.embedding_models.items():
            if model_name not in self.vector_stores:
                continue
            
            # Encode query
            query_embedding = model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search
            vector_store = self.vector_stores[model_name]
            scores, indices = vector_store['index'].search(
                query_embedding.astype(np.float32), top_k * 2  # Get more results for filtering
            )
            
            # Collect results with enhanced scoring
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    content_lower = chunk.page_content.lower()
                    
                    # Boost score for medical term matches
                    boosted_score = float(score)
                    for term in query_medical_terms:
                        if term in content_lower:
                            boosted_score += 0.1  # Boost for medical term matches
                    
                    # Boost for coverage-related terms
                    coverage_terms = ['covered', 'eligible', 'approved', 'excluded', 'not covered']
                    for term in coverage_terms:
                        if term in content_lower:
                            boosted_score += 0.05
                    
                    clause = RetrievedClause(
                        content=chunk.page_content,
                        source_document=chunk.metadata.get('source', ''),
                        relevance_score=boosted_score,
                        clause_type=model_name
                    )
                    all_results.append(clause)
        
        # Remove duplicates and sort by relevance
        unique_results = {}
        for result in all_results:
            key = result.content[:200]  # Use first 200 chars as key for better deduplication
            if key not in unique_results or result.relevance_score > unique_results[key].relevance_score:
                unique_results[key] = result
        
        # Sort by relevance score
        final_results = sorted(unique_results.values(), 
                             key=lambda x: x.relevance_score, reverse=True)
        
        return final_results[:top_k]

class LLMDecisionEngine:
    """Advanced decision engine using chain-of-thought reasoning"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        # Initialize your preferred LLM here
        # This is a placeholder - replace with actual LLM initialization
        
    def make_decision(self, query: QueryStructured, 
                     relevant_clauses: List[RetrievedClause]) -> DecisionResponse:
        """Make decision using advanced chain-of-thought reasoning"""
        
        # Create context from relevant clauses
        context = self._create_context(relevant_clauses)
        
        # Generate reasoning chain
        reasoning_chain = self._generate_reasoning_chain(query, context)
        
        # Make final decision
        decision = self._evaluate_decision(query, context, reasoning_chain)
        
        return decision
    
    def _create_context(self, clauses: List[RetrievedClause]) -> str:
        """Create structured context from retrieved clauses"""
        context_parts = []
        for i, clause in enumerate(clauses[:5]):  # Top 5 most relevant
            context_parts.append(f"Clause {i+1} (Relevance: {clause.relevance_score:.3f}):")
            context_parts.append(clause.content)
            context_parts.append(f"Source: {clause.source_document}")
            context_parts.append("---")
        
        return "\n".join(context_parts)
    
    def _generate_reasoning_chain(self, query: QueryStructured, context: str) -> List[str]:
        """Generate step-by-step reasoning chain"""
        reasoning_steps = []
        
        # Step 1: Query analysis
        reasoning_steps.append(f"Query Analysis: {query.age}Y {query.gender}, {query.procedure} in {query.location}, {query.policy_duration} policy")
        
        # Step 2: Policy eligibility check
        if query.policy_age_months and query.policy_age_months < 12:
            reasoning_steps.append(f"Policy Age Check: Policy is {query.policy_age_months} months old (less than 1 year)")
        
        # Step 3: Procedure coverage check
        if query.procedure:
            reasoning_steps.append(f"Procedure Check: Evaluating coverage for {query.procedure}")
        
        # Step 4: Location coverage check
        if query.location:
            reasoning_steps.append(f"Location Check: Treatment location is {query.location}")
        
        # Step 5: Context evaluation
        reasoning_steps.append("Context Evaluation: Analyzing relevant policy clauses")
        
        return reasoning_steps
    
    def _evaluate_decision(self, query: QueryStructured, context: str, 
                          reasoning_chain: List[str]) -> DecisionResponse:
        """Evaluate final decision based on document content analysis"""
        
        decision = "requires_review"  # Default
        amount = None
        confidence = 0.5
        justification = "Manual review required"
        
        context_lower = context.lower()
        query_procedure = query.procedure.lower() if query.procedure else ""
        
        # Debug: Show what we're analyzing
        print(f"\nDEBUG: Analyzing context for procedure: '{query_procedure}'")
        print(f"   Context length: {len(context)} characters")
        print(f"   Context preview: {context[:200]}...")
        
        # Analyze document content for coverage patterns
        coverage_indicators = {
            'covered': 0,
            'eligible': 0,
            'approved': 0,
            'excluded': 0,
            'not covered': 0,
            'waiting period': 0,
            'pre-existing': 0,
            'knee': 0,
            'surgery': 0,
            'treatment': 0
        }
        
        # Count coverage indicators in context
        for indicator in coverage_indicators:
            coverage_indicators[indicator] = context_lower.count(indicator)
        
        print(f"   Coverage indicators found: {coverage_indicators}")
        
        # Check for specific procedure mentions
        procedure_mentioned = query_procedure in context_lower if query_procedure else False
        surgery_mentioned = 'surgery' in context_lower
        knee_mentioned = 'knee' in context_lower
        
        print(f"   Procedure mentioned: {procedure_mentioned}")
        print(f"   Surgery mentioned: {surgery_mentioned}")
        print(f"   Knee mentioned: {knee_mentioned}")
        
        # Decision logic based on document content
        if procedure_mentioned or (query_procedure and any(term in context_lower for term in query_procedure.split())):
            # Procedure is mentioned in document
            if coverage_indicators['excluded'] > 0 or coverage_indicators['not covered'] > 0:
                decision = "rejected"
                confidence = 0.8
                justification = "Procedure is explicitly excluded in the policy"
            elif coverage_indicators['covered'] > 0 or coverage_indicators['eligible'] > 0:
                decision = "approved"
                confidence = 0.85
                justification = "Procedure is explicitly covered in the policy"
            else:
                decision = "requires_review"
                confidence = 0.6
                justification = "Procedure mentioned but coverage status unclear"
        
        # Check for surgery-specific coverage
        elif surgery_mentioned:
            if coverage_indicators['covered'] > 0:
                decision = "approved"
                confidence = 0.8
                justification = "Surgery is covered under the policy"
            elif coverage_indicators['excluded'] > 0:
                decision = "rejected"
                confidence = 0.75
                justification = "Surgery is excluded from coverage"
            else:
                decision = "requires_review"
                confidence = 0.5
                justification = "Surgery mentioned but coverage unclear"
        
        # Check for knee-specific coverage
        elif knee_mentioned and 'knee' in query_procedure:
            if coverage_indicators['covered'] > 0:
                decision = "approved"
                confidence = 0.9
                justification = "Knee surgery is covered under the policy"
            else:
                decision = "requires_review"
                confidence = 0.6
                justification = "Knee surgery mentioned but coverage unclear"
        
        # Extract amounts from context
        amount_patterns = [
            r'coverage.*?(\d+(?:,\d+)*(?:\.\d+)?)',
            r'amount.*?(\d+(?:,\d+)*(?:\.\d+)?)',
            r'up to.*?(\d+(?:,\d+)*(?:\.\d+)?)',
            r'maximum.*?(\d+(?:,\d+)*(?:\.\d+)?)'
        ]
        
        for pattern in amount_patterns:
            amount_match = re.search(pattern, context_lower)
            if amount_match:
                try:
                    amount = float(amount_match.group(1).replace(',', ''))
                    break
                except ValueError:
                    continue
        
        # Policy age considerations
        if query.policy_age_months and query.policy_age_months < 6:
            if coverage_indicators['waiting period'] > 0:
                confidence *= 0.7
                justification += " (Waiting period may apply)"
        
        # Age-based adjustments
        if query.age and query.age > 65:
            confidence *= 0.9
            justification += " (Senior citizen considerations)"
        
        print(f"   Final decision: {decision} (confidence: {confidence:.2f})")
        print(f"   Justification: {justification}")
        
        return DecisionResponse(
            decision=decision,
            amount=amount,
            confidence=confidence,
            justification=justification,
            relevant_clauses=None,
            reasoning_chain=reasoning_chain
        )

class AdvancedRAGSystem:
    """Main RAG system orchestrating all components"""
    
    def __init__(self):
        self.query_parser = AdvancedQueryParser()
        self.retriever = HybridEmbeddingRetriever()
        self.decision_engine = LLMDecisionEngine()
        
        logger.info("Advanced RAG System initialized")
    
    def load_documents(self, document_paths: List[str]):
        """Load documents into the system"""
        self.retriever.load_documents(document_paths)
    
    def process_query(self, query: str, debug: bool = False) -> Dict[str, Any]:
        """Process a natural language query and return structured decision"""
        
        # Step 1: Parse query
        structured_query = self.query_parser.parse_query(query)
        logger.info(f"Parsed query: {structured_query}")
        
        # Step 2: Retrieve relevant information
        relevant_clauses = self.retriever.retrieve_relevant_chunks(query, top_k=10)
        logger.info(f"Retrieved {len(relevant_clauses)} relevant clauses")
        
        # Debug: Show retrieved content
        if debug and relevant_clauses:
            print(f"\nDEBUG: Retrieved Content:")
            for i, clause in enumerate(relevant_clauses[:3], 1):
                print(f"   Clause {i} (Score: {clause.relevance_score:.3f}):")
                content_preview = clause.content[:200].replace('\n', ' ').strip()
                print(f"      {content_preview}...")
        elif debug:
            print(f"\nDEBUG: No relevant clauses found!")
            print(f"   Total chunks available: {len(self.retriever.chunks)}")
            print(f"   Query: '{query}'")
            
            # Show some sample chunks to see what's in the document
            if self.retriever.chunks:
                print(f"\nSample document chunks:")
                for i, chunk in enumerate(self.retriever.chunks[:3], 1):
                    preview = chunk.page_content[:100].replace('\n', ' ').strip()
                    print(f"   Chunk {i}: {preview}...")
        
        # Step 3: Make decision
        decision = self.decision_engine.make_decision(structured_query, relevant_clauses)
        logger.info(f"Decision: {decision.decision} (confidence: {decision.confidence:.2f})")
        
        # Step 4: Format response with better structure
        response = {
            "decision": decision.decision,
            "amount": decision.amount,
            "confidence": decision.confidence,
            "justification": decision.justification,
            "reasoning_chain": decision.reasoning_chain,
            "query_analysis": {
                "age": structured_query.age,
                "gender": structured_query.gender,
                "procedure": structured_query.procedure,
                "location": structured_query.location,
                "policy_duration": structured_query.policy_duration,
                "policy_age_months": structured_query.policy_age_months,
                "extracted_entities": structured_query.extracted_entities
            },
            "relevant_clauses": [
                {
                    "content": clause.content[:300] + "..." if len(clause.content) > 300 else clause.content,
                    "source": clause.source_document,
                    "relevance_score": clause.relevance_score,
                    "clause_type": clause.clause_type
                }
                for clause in relevant_clauses[:5]  # Top 5 clauses
            ],
            "summary": {
                "total_clauses_retrieved": len(relevant_clauses),
                "top_relevance_score": max([c.relevance_score for c in relevant_clauses]) if relevant_clauses else 0,
                "average_relevance_score": sum([c.relevance_score for c in relevant_clauses]) / len(relevant_clauses) if relevant_clauses else 0
            }
        }
        
        return response
    
    def batch_process(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Process multiple queries in batch"""
        results = []
        for query in queries:
            try:
                result = self.process_query(query)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
                results.append({
                    "error": str(e),
                    "query": query
                })
        return results

# Example usage and testing
def create_sample_documents():
    """Create sample insurance policy documents for testing"""
    sample_docs = [
        {
            "content": """
            INSURANCE POLICY TERMS AND CONDITIONS
            
            Coverage Details:
            - Knee surgery is covered for patients aged 18-65
            - Waiting period: 6 months for major surgeries
            - Maximum coverage amount: $50,000 for orthopedic procedures
            - Pre-authorization required for surgeries over $10,000
            
            Exclusions:
            - Cosmetic procedures not covered
            - Experimental treatments excluded
            - Pre-existing conditions have 12-month waiting period
            
            Network Hospitals:
            - All major hospitals in Pune, Mumbai, Delhi
            - Cashless treatment available at network hospitals
            """,
            "source": "policy_terms.txt"
        },
        {
            "content": """
            DENTAL COVERAGE POLICY
            
            Covered Procedures:
            - Routine dental checkups (2 per year)
            - Dental fillings and extractions
            - Root canal treatment (up to $2,000)
            - Dental surgery for medical necessity
            
            Coverage Limits:
            - Annual maximum: $3,000
            - Waiting period: 3 months for major procedures
            - Co-pay: 20% for specialist consultations
            
            Network Dentists:
            - All registered dental clinics
            - Emergency dental care covered
            """,
            "source": "dental_policy.txt"
        },
        {
            "content": """
            CARDIAC CARE COVERAGE
            
            Heart Surgery Coverage:
            - Bypass surgery: Up to $100,000
            - Angioplasty: Up to $75,000
            - Heart valve replacement: Up to $150,000
            - Emergency cardiac procedures: Full coverage
            
            Eligibility:
            - Age 18-70 for major cardiac procedures
            - Pre-existing heart conditions: 24-month waiting period
            - Second opinion required for surgeries over $50,000
            
            Network Cardiac Centers:
            - All accredited cardiac hospitals
            - Emergency air ambulance coverage
            """,
            "source": "cardiac_policy.txt"
        }
    ]
    return sample_docs

def load_single_document(file_path: str):
    """Load a single document from file"""
    path_obj = Path(file_path)
    
    if path_obj.suffix.lower() == '.pdf':
        content = _extract_pdf_content(file_path)
    elif path_obj.suffix.lower() in ['.docx', '.doc']:
        content = _extract_docx_content(file_path)
    elif path_obj.suffix.lower() == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    return Document(page_content=content, metadata={'source': file_path})

def _extract_pdf_content(path: str) -> str:
    """Extract text from PDF using PyMuPDF"""
    if fitz is None:
        raise ImportError("PyMuPDF not available. Install with: pip install PyMuPDF")
        
    try:
        doc = fitz.open(path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        raise Exception(f"Failed to extract PDF content: {e}")

def _extract_docx_content(path: str) -> str:
    """Extract text from DOCX"""
    if DocxDocument is None:
        raise ImportError("python-docx not available. Install with: pip install python-docx")
        
    try:
        doc = DocxDocument(path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        raise Exception(f"Failed to extract DOCX content: {e}")

def print_formatted_result(result: Dict[str, Any]):
    """Print results in a conversational format"""
    
    # Generate conversational response
    response = generate_conversational_response(result)
    print(f"\nRESPONSE: {response}")
    
    # Show confidence and key details
    confidence = result['confidence']
    decision = result['decision']
    
    if confidence < 0.5:
        confidence_note = " (Low confidence - manual review recommended)"
    elif confidence < 0.8:
        confidence_note = " (Moderate confidence)"
    else:
        confidence_note = " (High confidence)"
    
    print(f"Confidence: {confidence:.1%}{confidence_note}")
    
    # Show amount if available
    if result.get('amount'):
        print(f"Coverage Amount: ${result['amount']:,.2f}")
    
    # Show key reasoning if confidence is low
    if confidence < 0.7:
        print(f"\nKey Factors:")
        analysis = result["query_analysis"]
        if analysis.get("procedure"):
            print(f"   • Procedure: {analysis['procedure']}")
        if analysis.get("policy_age_months") and analysis['policy_age_months'] < 12:
            print(f"   • Policy Age: {analysis['policy_age_months']} months (may affect coverage)")
        if analysis.get("age") and analysis['age'] > 65:
            print(f"   • Age: {analysis['age']} years (senior citizen considerations)")
    
    print("-" * 60)

def generate_conversational_response(result: Dict[str, Any]) -> str:
    """Generate a conversational response based on the decision"""
    decision = result['decision']
    analysis = result['query_analysis']
    procedure = analysis.get('procedure', 'the procedure')
    
    if decision == "approved":
        if procedure and procedure != "surgery":
            return f"Yes, {procedure} is covered under the policy."
        else:
            return "Yes, this procedure is covered under the policy."
    
    elif decision == "rejected":
        if procedure and procedure != "surgery":
            return f"No, {procedure} is not covered under the policy."
        else:
            return "No, this procedure is not covered under the policy."
    
    elif decision == "requires_review":
        if procedure and procedure != "surgery":
            return f"This {procedure} requires manual review. Please contact customer service for detailed assessment."
        else:
            return "This procedure requires manual review. Please contact customer service for detailed assessment."
    
    else:
        return "Unable to determine coverage. Please contact customer service for assistance."

def main(document_path: str = None):
    """Example usage of the Advanced RAG System"""
    
    # Initialize system
    rag_system = AdvancedRAGSystem()
    
    if document_path:
        # Load user's document using enhanced processing
        print(f"Loading document: {document_path}")
        try:
            # Use the enhanced document loading
            rag_system.retriever.load_documents([document_path])
            print(f"Document loaded successfully!")
        except Exception as e:
            print(f"Error loading document: {e}")
            return
    else:
        # Create and load sample documents
        print("Loading sample documents...")
        sample_docs = create_sample_documents()
        
        # Convert sample documents to Document objects and load them
        if Document is not None:
            documents = []
            for doc in sample_docs:
                documents.append(Document(page_content=doc["content"], metadata={'source': doc["source"]}))
            
            # Load sample documents
            rag_system.retriever.documents = documents
            rag_system.retriever.chunks = documents  # For sample docs, use full documents
    
    # Interactive query testing
    print("\n" + "="*60)
    print("RAG SYSTEM READY FOR TESTING")
    print("="*60)
    print("Enter your queries (type 'quit' to exit):")
    
    while True:
        try:
            query = input("\nEnter your query: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            if not query:
                continue
                
            print(f"\nProcessing: {query}")
            print("-" * 50)
            
            result = rag_system.process_query(query, debug=True)
            print_formatted_result(result)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Use document path from command line argument
        main(sys.argv[1])
    else:
        # Use sample documents
        main()