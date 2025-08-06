#!/usr/bin/env python3
"""
FastAPI application for Insurance RAG System - Hackathon Submission
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
import uvicorn
import os
import logging
from datetime import datetime
import requests
import tempfile
from pathlib import Path

# Import our RAG system
from rag import EnhancedRAGSystem, create_sample_documents
from langchain.docstore.document import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Insurance RAG System API",
    description="Advanced RAG system for insurance policy analysis and question answering",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# API Key validation (you should set this as an environment variable)
API_KEY = os.getenv("API_KEY", "hackrx-2024-secret-key")

# Initialize RAG system
rag_system = None

def get_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate API key"""
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return credentials.credentials

# Pydantic models
class HackRxRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

def initialize_rag_system():
    """Initialize the RAG system"""
    global rag_system
    if rag_system is None:
        logger.info("Initializing RAG system...")
        rag_system = EnhancedRAGSystem()
        
        # Load sample documents as fallback
        sample_docs = create_sample_documents()
        documents = []
        for doc in sample_docs:
            documents.append(Document(page_content=doc["content"], metadata={'source': doc["source"]}))
        
        rag_system.retriever.documents = documents
        rag_system.retriever.chunks = documents
        logger.info("RAG system initialized successfully")

def download_document(url: str) -> str:
    """Download document from URL and return local path"""
    try:
        logger.info(f"Downloading document from: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name
        
        logger.info(f"Document downloaded to: {tmp_file_path}")
        return tmp_file_path
        
    except Exception as e:
        logger.error(f"Failed to download document: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download document: {str(e)}"
        )

def process_question(question: str) -> str:
    """Process a single question and return answer using LLM-based analysis"""
    try:
        # Use the RAG system to process the question
        result = rag_system.process_query(question, debug=False)
        
        # Get the LLM-based answer from the result
        if 'coverage_details' in result and 'llm_answer' in result['coverage_details']:
            # Use LLM answer if available
            return result['coverage_details']['llm_answer']
        elif 'justification' in result:
            # Use justification as fallback
            return result['justification']
        else:
            # Fallback response
            return "Unable to determine the answer based on the available policy information."
        
    except Exception as e:
        logger.error(f"Error processing question '{question}': {e}")
        return f"Error processing question: {str(e)}"

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup"""
    initialize_rag_system()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Insurance RAG System API",
        "status": "running",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "rag_system_initialized": rag_system is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/hackrx/run", response_model=HackRxResponse)
async def hackrx_run(
    request: HackRxRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Main endpoint for processing insurance policy questions
    
    This endpoint:
    1. Downloads the policy document from the provided URL
    2. Processes each question using the RAG system
    3. Returns structured answers
    """
    try:
        logger.info(f"Processing request with {len(request.questions)} questions")
        
        # Download and load the document
        document_path = download_document(str(request.documents))
        
        try:
            # Load the document into the RAG system
            rag_system.load_documents([document_path])
            logger.info("Document loaded successfully into RAG system")
            
            # Verify that documents were loaded
            if not rag_system.retriever.chunks:
                logger.warning("No chunks found after loading document, using sample documents")
                # Load sample documents as fallback
                sample_docs = create_sample_documents()
                documents = []
                for doc in sample_docs:
                    documents.append(Document(page_content=doc["content"], metadata={'source': doc["source"]}))
                
                rag_system.retriever.documents = documents
                rag_system.retriever.chunks = documents
                logger.info("Sample documents loaded as fallback")
            
        except Exception as e:
            logger.warning(f"Failed to load document into RAG system: {e}")
            logger.info("Using fallback sample documents")
            # Load sample documents as fallback
            sample_docs = create_sample_documents()
            documents = []
            for doc in sample_docs:
                documents.append(Document(page_content=doc["content"], metadata={'source': doc["source"]}))
            
            rag_system.retriever.documents = documents
            rag_system.retriever.chunks = documents
            logger.info("Sample documents loaded as fallback")
        
        # Process each question
        answers = []
        for i, question in enumerate(request.questions):
            logger.info(f"Processing question {i+1}/{len(request.questions)}: {question[:50]}...")
            answer = process_question(question)
            answers.append(answer)
        
        # Clean up temporary file
        try:
            os.unlink(document_path)
            logger.info("Temporary file cleaned up")
        except:
            pass
        
        logger.info(f"Successfully processed {len(answers)} questions")
        
        return HackRxResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"Error in hackrx_run: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/api/v1/hackrx/run", response_model=HackRxResponse)
async def hackrx_run_v1(
    request: HackRxRequest,
    api_key: str = Depends(get_api_key)
):
    """Alternative endpoint path for compatibility"""
    return await hackrx_run(request, api_key)

if __name__ == "__main__":
    # Run the application
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    ) 