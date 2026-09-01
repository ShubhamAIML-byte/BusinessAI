"""
rag_pipeline.py
---------------
Retrieval-Augmented Generation (RAG) Pipeline for BusinessAI

This module provides:
- Document ingestion (PDF, DOCX, TXT, CSV)
- Vector embeddings using sentence-transformers
- ChromaDB vector storage and retrieval
- Semantic search capabilities
- Context-aware response generation
- Integration with existing LLM backend

The RAG pipeline enhances the system with:
1. Knowledge base management
2. Semantic document search
3. Context-aware query answering
4. Document upload and processing
5. Persistent vector storage
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Document loaders
try:
    import pypdf
    from docx import Document as DocxDocument
except ImportError:
    pypdf = None
    DocxDocument = None

import pandas as pd

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Document Processing
# -----------------------------------------------------------------------------

class DocumentProcessor:
    """Process various document types into text chunks."""
    
    @staticmethod
    def read_pdf(file_path: str) -> str:
        """Extract text from PDF file."""
        if pypdf is None:
            raise ImportError("pypdf not installed. Install with: pip install pypdf")
        
        text = []
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = pypdf.PdfReader(f)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
            return "\n\n".join(text)
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}")
            return ""
    
    @staticmethod
    def read_docx(file_path: str) -> str:
        """Extract text from DOCX file."""
        if DocxDocument is None:
            raise ImportError("python-docx not installed. Install with: pip install python-docx")
        
        try:
            doc = DocxDocument(file_path)
            text = []
            for para in doc.paragraphs:
                if para.text.strip():
                    text.append(para.text)
            return "\n\n".join(text)
        except Exception as e:
            logger.error(f"Error reading DOCX {file_path}: {e}")
            return ""
    
    @staticmethod
    def read_txt(file_path: str) -> str:
        """Read plain text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error reading TXT {file_path}: {e}")
                return ""
    
    @staticmethod
    def read_csv(file_path: str) -> str:
        """Convert CSV to readable text format."""
        try:
            df = pd.read_csv(file_path)
            # Convert to markdown-like table format
            text = f"CSV File: {Path(file_path).name}\n\n"
            text += df.to_string(index=False)
            return text
        except Exception as e:
            logger.error(f"Error reading CSV {file_path}: {e}")
            return ""
    
    @classmethod
    def read_document(cls, file_path: str) -> str:
        """Read document based on file extension."""
        file_path_lower = file_path.lower()
        
        if file_path_lower.endswith('.pdf'):
            return cls.read_pdf(file_path)
        elif file_path_lower.endswith('.docx'):
            return cls.read_docx(file_path)
        elif file_path_lower.endswith('.txt'):
            return cls.read_txt(file_path)
        elif file_path_lower.endswith('.csv'):
            return cls.read_csv(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_path}")
            return ""


class TextChunker:
    """Split text into semantic chunks for embedding."""
    
    @staticmethod
    def chunk_by_sentences(
        text: str,
        max_chunk_size: int = 500,
        overlap: int = 50
    ) -> List[str]:
        """
        Split text into chunks by sentences with overlap.
        
        Args:
            text: Input text to chunk
            max_chunk_size: Maximum characters per chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If single sentence is too long, split it
            if sentence_length > max_chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split long sentence into smaller parts
                words = sentence.split()
                temp_chunk = []
                temp_length = 0
                
                for word in words:
                    if temp_length + len(word) + 1 > max_chunk_size:
                        if temp_chunk:
                            chunks.append(' '.join(temp_chunk))
                        temp_chunk = [word]
                        temp_length = len(word)
                    else:
                        temp_chunk.append(word)
                        temp_length += len(word) + 1
                
                if temp_chunk:
                    chunks.append(' '.join(temp_chunk))
                continue
            
            # Add sentence to current chunk if it fits
            if current_length + sentence_length + 1 <= max_chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length + 1
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                
                # Add overlap from previous chunk
                if overlap > 0 and current_chunk:
                    overlap_text = ' '.join(current_chunk)[-overlap:]
                    current_chunk = [overlap_text, sentence]
                    current_length = len(overlap_text) + sentence_length + 1
                else:
                    current_chunk = [sentence]
                    current_length = sentence_length
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return [chunk.strip() for chunk in chunks if chunk.strip()]


# -----------------------------------------------------------------------------
# Vector Store
# -----------------------------------------------------------------------------

@dataclass
class RAGPipeline:
    """
    Complete RAG pipeline with document processing, embedding, and retrieval.
    
    Features:
    - Document ingestion from multiple formats
    - Semantic chunking
    - Vector embeddings using sentence-transformers
    - ChromaDB vector storage
    - Semantic search and retrieval
    """
    
    collection_name: str = "business_documents"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    chroma_persist_dir: str = "./chroma_db"
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    # Internal state (initialized in __post_init__)
    embedding_model: Any = field(default=None, init=False, repr=False)
    chroma_client: Any = field(default=None, init=False, repr=False)
    collection: Any = field(default=None, init=False, repr=False)
    doc_processor: DocumentProcessor = field(default_factory=DocumentProcessor, init=False)
    text_chunker: TextChunker = field(default_factory=TextChunker, init=False)
    
    def __post_init__(self):
        """Initialize embedding model and vector store."""
        self._init_embedding_model()
        self._init_chroma()
    
    def _init_embedding_model(self):
        """Initialize sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _init_chroma(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Create persist directory if it doesn't exist
            os.makedirs(self.chroma_persist_dir, exist_ok=True)
            
            # Initialize ChromaDB client with persistence
            self.chroma_client = chromadb.PersistentClient(
                path=self.chroma_persist_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"ChromaDB initialized. Collection: {self.collection_name}")
            logger.info(f"Current document count: {self.collection.count()}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.embedding_model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return []
    
    def ingest_document(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ingest a document into the RAG system.
        
        Args:
            file_path: Path to document file
            metadata: Optional metadata to attach to document chunks
            
        Returns:
            Dictionary with ingestion results
        """
        try:
            # Read document
            logger.info(f"Reading document: {file_path}")
            text = self.doc_processor.read_document(file_path)
            
            if not text:
                return {
                    "success": False,
                    "error": "Failed to extract text from document",
                    "chunks_added": 0
                }
            
            # Chunk text
            logger.info(f"Chunking document: {Path(file_path).name}")
            chunks = self.text_chunker.chunk_by_sentences(
                text,
                max_chunk_size=self.chunk_size,
                overlap=self.chunk_overlap
            )
            
            if not chunks:
                return {
                    "success": False,
                    "error": "No chunks created from document",
                    "chunks_added": 0
                }
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            embeddings = self.embed_texts(chunks)
            
            if not embeddings:
                return {
                    "success": False,
                    "error": "Failed to generate embeddings",
                    "chunks_added": 0
                }
            
            # Prepare metadata
            file_name = Path(file_path).name
            doc_metadata = metadata or {}
            doc_metadata.update({
                "source": file_name,
                "file_path": file_path,
                "total_chunks": len(chunks)
            })
            
            # Create unique IDs for chunks
            base_id = f"{file_name}_{hash(file_path)}"
            ids = [f"{base_id}_chunk_{i}" for i in range(len(chunks))]
            
            # Add metadata to each chunk
            chunk_metadatas = [
                {**doc_metadata, "chunk_index": i}
                for i in range(len(chunks))
            ]
            
            # Add to ChromaDB
            logger.info(f"Adding {len(chunks)} chunks to vector store")
            self.collection.add(
                documents=chunks,
                embeddings=embeddings,
                metadatas=chunk_metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully ingested {file_name}: {len(chunks)} chunks")
            
            return {
                "success": True,
                "file_name": file_name,
                "chunks_added": len(chunks),
                "total_characters": len(text)
            }
            
        except Exception as e:
            logger.error(f"Error ingesting document {file_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "chunks_added": 0
            }
    
    def ingest_text(
        self,
        text: str,
        source_name: str = "direct_input",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ingest raw text directly into the RAG system.
        
        Args:
            text: Raw text to ingest
            source_name: Name to identify this text source
            metadata: Optional metadata
            
        Returns:
            Dictionary with ingestion results
        """
        try:
            # Chunk text
            chunks = self.text_chunker.chunk_by_sentences(
                text,
                max_chunk_size=self.chunk_size,
                overlap=self.chunk_overlap
            )
            
            if not chunks:
                return {
                    "success": False,
                    "error": "No chunks created from text",
                    "chunks_added": 0
                }
            
            # Generate embeddings
            embeddings = self.embed_texts(chunks)
            
            # Prepare metadata
            doc_metadata = metadata or {}
            doc_metadata.update({
                "source": source_name,
                "total_chunks": len(chunks)
            })
            
            # Create unique IDs
            base_id = f"{source_name}_{hash(text)}"
            ids = [f"{base_id}_chunk_{i}" for i in range(len(chunks))]
            
            chunk_metadatas = [
                {**doc_metadata, "chunk_index": i}
                for i in range(len(chunks))
            ]
            
            # Add to ChromaDB
            self.collection.add(
                documents=chunks,
                embeddings=embeddings,
                metadatas=chunk_metadatas,
                ids=ids
            )
            
            return {
                "success": True,
                "source_name": source_name,
                "chunks_added": len(chunks),
                "total_characters": len(text)
            }
            
        except Exception as e:
            logger.error(f"Error ingesting text: {e}")
            return {
                "success": False,
                "error": str(e),
                "chunks_added": 0
            }
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search for relevant document chunks.
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of search results with text, metadata, and scores
        """
        try:
            # Generate query embedding
            query_embedding = self.embed_texts([query])[0]
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_metadata
            )
            
            # Format results
            search_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    search_results.append({
                        "text": doc,
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                        "distance": results['distances'][0][i] if results['distances'] else 0.0,
                        "similarity": 1 - results['distances'][0][i] if results['distances'] else 1.0
                    })
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []
    
    def retrieve_context(
        self,
        query: str,
        n_results: int = 3,
        max_context_length: int = 2000
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: User query
            n_results: Number of chunks to retrieve
            max_context_length: Maximum characters in context
            
        Returns:
            Tuple of (context_string, list_of_sources)
        """
        results = self.search(query, n_results=n_results)
        
        if not results:
            return "", []
        
        # Build context string
        context_parts = []
        sources = []
        total_length = 0
        
        for i, result in enumerate(results, 1):
            chunk_text = result['text']
            chunk_length = len(chunk_text)
            
            if total_length + chunk_length > max_context_length:
                break
            
            context_parts.append(f"[Source {i}]\n{chunk_text}")
            sources.append({
                "source": result['metadata'].get('source', 'Unknown'),
                "chunk_index": result['metadata'].get('chunk_index', 0),
                "similarity": result['similarity']
            })
            
            total_length += chunk_length
        
        context = "\n\n".join(context_parts)
        return context, sources
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system."""
        try:
            total_docs = self.collection.count()
            
            # Get unique sources
            if total_docs > 0:
                all_results = self.collection.get()
                sources = set()
                if all_results['metadatas']:
                    sources = {
                        meta.get('source', 'Unknown')
                        for meta in all_results['metadatas']
                    }
            else:
                sources = set()
            
            return {
                "total_chunks": total_docs,
                "unique_sources": len(sources),
                "sources": list(sources),
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model_name
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                "total_chunks": 0,
                "unique_sources": 0,
                "sources": [],
                "error": str(e)
            }
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection."""
        try:
            # Delete the collection
            self.chroma_client.delete_collection(self.collection_name)
            
            # Recreate it
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"Cleared collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False
    
    def delete_by_source(self, source_name: str) -> bool:
        """Delete all chunks from a specific source."""
        try:
            # Get all IDs for this source
            results = self.collection.get(
                where={"source": source_name}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} chunks from {source_name}")
                return True
            else:
                logger.info(f"No chunks found for source: {source_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting source {source_name}: {e}")
            return False


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def create_rag_pipeline(
    collection_name: str = "business_documents",
    embedding_model: str = "all-MiniLM-L6-v2",
    persist_dir: str = "./chroma_db"
) -> RAGPipeline:
    """
    Create and initialize a RAG pipeline.
    
    Args:
        collection_name: Name for the ChromaDB collection
        embedding_model: Name of sentence-transformer model
        persist_dir: Directory for ChromaDB persistence
        
    Returns:
        Initialized RAGPipeline instance
    """
    return RAGPipeline(
        collection_name=collection_name,
        embedding_model_name=embedding_model,
        chroma_persist_dir=persist_dir
    )
