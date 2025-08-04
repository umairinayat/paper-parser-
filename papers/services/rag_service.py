"""
RAG (Retrieval-Augmented Generation) service for paper question-answering.
Uses Groq API for LLM inference and vector embeddings for document retrieval.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

# PDF processing
import PyPDF2
import fitz  # PyMuPDF

# Text processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Groq API
from groq import Groq
from django.conf import settings
from django.core.cache import cache

logger = logging.getLogger(__name__)

class RAGService:
    """Service for Retrieval-Augmented Generation on uploaded papers."""
    
    def __init__(self):
        self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file using PyMuPDF for better accuracy."""
        try:
            text = ""
            with fitz.open(pdf_path) as doc:
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text += page.get_text()
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
            # Fallback to PyPDF2
            try:
                text = ""
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                return text.strip()
            except Exception as e2:
                logger.error(f"Fallback PDF extraction also failed: {str(e2)}")
                raise e2
    
    def process_paper(self, paper) -> str:
        """Process a paper and create vector embeddings for RAG."""
        try:
            # Extract text from the uploaded file
            if not paper.file:
                raise ValueError("No file attached to paper")
            
            file_path = paper.file.path
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Paper file not found: {file_path}")
            
            # Extract text based on file type
            if paper.file_extension == 'pdf':
                text_content = self.extract_text_from_pdf(file_path)
            else:
                raise ValueError(f"Unsupported file type: {paper.file_extension}")
            
            if not text_content or len(text_content.strip()) < 100:
                raise ValueError("Insufficient text content extracted from paper")
            
            # Update paper with extracted text
            paper.full_text = text_content
            paper.save()
            
            # Create text chunks
            documents = self.text_splitter.split_text(text_content)
            
            # Create Document objects with metadata
            docs = []
            for i, chunk in enumerate(documents):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        'paper_id': str(paper.id),
                        'paper_title': paper.title,
                        'chunk_id': i,
                        'paper_authors': paper.authors,
                        'paper_abstract': paper.abstract[:500] if paper.abstract else ""
                    }
                )
                docs.append(doc)
            
            # Create vector store
            vectorstore = FAISS.from_documents(docs, self.embeddings)
            
            # Save vectorstore to disk
            vector_path = self._get_vector_path(paper.id)
            os.makedirs(os.path.dirname(vector_path), exist_ok=True)
            vectorstore.save_local(vector_path)
            
            logger.info(f"Successfully processed paper {paper.id} with {len(docs)} chunks")
            return f"Successfully processed paper with {len(docs)} text chunks"
            
        except Exception as e:
            logger.error(f"Error processing paper {paper.id}: {str(e)}")
            raise e
    
    def _get_vector_path(self, paper_id: str) -> str:
        """Get the path for storing vector embeddings."""
        return os.path.join(settings.MEDIA_ROOT, 'vectors', str(paper_id))
    
    def load_vectorstore(self, paper_id: str) -> Optional[FAISS]:
        """Load vector store for a paper."""
        try:
            vector_path = self._get_vector_path(paper_id)
            if os.path.exists(vector_path):
                return FAISS.load_local(vector_path, self.embeddings, allow_dangerous_deserialization=True)
            return None
        except Exception as e:
            logger.error(f"Error loading vectorstore for paper {paper_id}: {str(e)}")
            return None
    
    def answer_question(self, paper, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Answer a question about a specific paper using RAG."""
        try:
            # Load or create vectorstore
            vectorstore = self.load_vectorstore(paper.id)
            if not vectorstore:
                # Process paper if vectorstore doesn't exist
                self.process_paper(paper)
                vectorstore = self.load_vectorstore(paper.id)
                if not vectorstore:
                    raise ValueError("Failed to create or load vectorstore")
            
            # Retrieve relevant documents
            relevant_docs = vectorstore.similarity_search(question, k=top_k)
            
            # Prepare context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Create the prompt
            prompt = self._create_qa_prompt(
                paper_title=paper.title,
                paper_abstract=paper.abstract,
                context=context,
                question=question
            )
            
            # Get answer from Groq
            response = self.groq_client.chat.completions.create(
                model=settings.PRIMARY_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant specialized in analyzing research papers. Provide accurate, concise answers based on the given context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=settings.MAX_TOKENS // 2,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content
            
            # Prepare sources information
            sources = []
            for doc in relevant_docs:
                sources.append({
                    'chunk_id': doc.metadata.get('chunk_id', 0),
                    'content_preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                })
            
            result = {
                'answer': answer,
                'sources': sources,
                'question': question,
                'paper_title': paper.title,
                'timestamp': datetime.now().isoformat(),
                'model_used': settings.PRIMARY_MODEL
            }
            
            # Cache the result for 1 hour
            cache_key = f"qa_{paper.id}_{hash(question)}"
            cache.set(cache_key, result, 3600)
            
            return result
            
        except Exception as e:
            logger.error(f"Error answering question for paper {paper.id}: {str(e)}")
            raise e
    
    def _create_qa_prompt(self, paper_title: str, paper_abstract: str, context: str, question: str) -> str:
        """Create a prompt for question answering."""
        return f"""
Based on the following research paper information, please answer the question accurately and concisely.

Paper Title: {paper_title}

Paper Abstract: {paper_abstract}

Relevant Context from Paper:
{context}

Question: {question}

Instructions:
1. Answer the question based solely on the provided context
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Provide specific details and quotes when relevant
4. Keep your answer concise but comprehensive
5. If the question asks for data, numbers, or specific findings, extract them accurately from the context

Answer:
"""
    
    def get_paper_summary(self, paper) -> Dict[str, Any]:
        """Generate a comprehensive summary of the paper using RAG."""
        try:
            if not paper.full_text:
                self.process_paper(paper)
            
            # Use the abstract and beginning of the paper for summary
            text_for_summary = (paper.abstract or "") + "\n\n" + paper.full_text[:3000]
            
            prompt = f"""
Please provide a comprehensive summary of this research paper:

Title: {paper.title}
Authors: {paper.authors}

Content:
{text_for_summary}

Please provide:
1. Main research question or objective
2. Key methodology used
3. Main findings and results
4. Significance and implications
5. Limitations mentioned by authors

Keep each section concise but informative.
"""
            
            response = self.groq_client.chat.completions.create(
                model=settings.PRIMARY_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert researcher who excels at summarizing academic papers clearly and accurately."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=settings.MAX_TOKENS // 2,
                temperature=0.1
            )
            
            summary = response.choices[0].message.content
            
            return {
                'summary': summary,
                'paper_title': paper.title,
                'generated_at': datetime.now().isoformat(),
                'model_used': settings.PRIMARY_MODEL
            }
            
        except Exception as e:
            logger.error(f"Error generating summary for paper {paper.id}: {str(e)}")
            raise e
    
    def suggest_questions(self, paper) -> List[str]:
        """Suggest relevant questions that can be asked about the paper."""
        try:
            # Use abstract and title to suggest questions
            content = f"Title: {paper.title}\n\nAbstract: {paper.abstract}"
            
            prompt = f"""
Based on the following research paper information, suggest 8-10 insightful questions that someone might want to ask about this paper:

{content}

Generate questions that would help someone understand:
- The research methodology
- Key findings and results
- Practical applications
- Limitations and future work
- Technical details
- Significance of the work

Format as a numbered list of questions.
"""
            
            response = self.groq_client.chat.completions.create(
                model=settings.PRIMARY_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert researcher who asks insightful questions about academic papers."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            suggestions_text = response.choices[0].message.content
            
            # Extract questions from the response
            questions = []
            for line in suggestions_text.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    # Remove numbering and clean up
                    question = line
                    for prefix in ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', '-', '•']:
                        if question.startswith(prefix):
                            question = question[len(prefix):].strip()
                            break
                    if question and question.endswith('?'):
                        questions.append(question)
            
            return questions[:10]  # Return max 10 questions
            
        except Exception as e:
            logger.error(f"Error generating question suggestions for paper {paper.id}: {str(e)}")
            return [
                "What is the main research question addressed in this paper?",
                "What methodology was used in this study?",
                "What are the key findings of this research?",
                "What are the practical implications of this work?",
                "What limitations does this study have?"
            ]
    
    def generate_followup_questions(self, paper, last_question: str, last_answer: str, max_questions: int = 3) -> List[str]:
        """Generate follow-up questions based on the previous Q&A"""
        try:
            prompt = f"""
Based on the following research paper Q&A exchange, suggest {max_questions} relevant follow-up questions that would help deepen understanding of the topic.

Paper Title: {paper.title}
Paper Abstract: {paper.abstract[:500] if paper.abstract else "No abstract available"}

Previous Question: {last_question}
Previous Answer: {last_answer[:1000]}...

Generate {max_questions} specific, insightful follow-up questions that:
1. Build upon the previous answer
2. Explore related aspects not yet covered
3. Help understand the broader context or implications
4. Are specific to this research paper

Format as a numbered list of questions.
"""
            
            response = self.groq_client.chat.completions.create(
                model=settings.PRIMARY_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert researcher who asks insightful follow-up questions to deepen understanding of academic papers."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7  # Slightly higher for creative questions
            )
            
            followup_text = response.choices[0].message.content
            
            # Extract questions from the response
            questions = []
            for line in followup_text.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    # Remove numbering and clean up
                    question = line
                    for prefix in ['1.', '2.', '3.', '4.', '5.', '-', '•']:
                        if question.startswith(prefix):
                            question = question[len(prefix):].strip()
                            break
                    if question and question.endswith('?'):
                        questions.append(question)
            
            return questions[:max_questions]
            
        except Exception as e:
            logger.error(f"Error generating follow-up questions for paper {paper.id}: {str(e)}")
            return []