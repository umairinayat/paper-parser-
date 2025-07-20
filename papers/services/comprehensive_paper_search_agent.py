import os
import json
import csv
import asyncio
import re
import requests
import logging
import traceback
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import pandas as pd
import contextlib
import arxiv
from pydantic import BaseModel, Field, validator
from urllib.parse import quote_plus
import time
from functools import wraps

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities.arxiv import ArxivAPIWrapper

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool

from django.conf import settings

# Configure comprehensive logging
def setup_logging():
    """Setup comprehensive logging configuration"""
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(settings.BASE_DIR if hasattr(settings, 'BASE_DIR') else '.', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create handlers
    file_handler = logging.FileHandler(
        os.path.join(logs_dir, f'paper_search_{datetime.now().strftime("%Y%m%d")}.log')
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Setup main logger
    logger = logging.getLogger('paper_search')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Prevent duplicate logs
    logger.propagate = False
    
    return logger

# Initialize logger
logger = setup_logging()

def log_execution_time(func):
    """Decorator to log function execution time"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        func_name = func.__name__
        logger.info(f"Starting {func_name}")
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Completed {func_name} in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed {func_name} after {execution_time:.2f} seconds: {str(e)}")
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        func_name = func.__name__
        logger.info(f"Starting {func_name}")
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Completed {func_name} in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed {func_name} after {execution_time:.2f} seconds: {str(e)}")
            raise
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

def log_paper_retrieval(source_name: str):
    """Decorator to log paper retrieval from specific sources"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"Retrieving papers from {source_name}")
            try:
                result = func(*args, **kwargs)
                paper_count = len(result) if isinstance(result, list) else 0
                logger.info(f"Successfully retrieved {paper_count} papers from {source_name}")
                return result
            except Exception as e:
                logger.error(f"Failed to retrieve papers from {source_name}: {str(e)}")
                logger.debug(f"Error details: {traceback.format_exc()}")
                return []
        return wrapper
    return decorator

# Define comprehensive state structure for LangGraph with proper Pydantic validation
class ExecutionMetadata(BaseModel):
    """Metadata about the execution process"""
    search_duration: Optional[float] = None
    total_papers_found: Optional[int] = None
    unique_papers: Optional[int] = None
    sources_searched: List[str] = Field(default_factory=list)
    analysis_start_time: Optional[str] = None
    analysis_end_time: Optional[str] = None

class PaperModel(BaseModel):
    """Pydantic model for individual papers"""
    title: str = ""
    authors: str = ""
    abstract: str = ""
    year: str = ""
    source: str = ""
    doi: Optional[str] = None
    categories: Optional[str] = None
    url: Optional[str] = None
    published: Optional[str] = None
    retrieval_timestamp: Optional[str] = None
    query_used: Optional[str] = None
    main_findings: Optional[str] = None
    methodology: Optional[str] = None
    interventions: Optional[str] = None
    outcomes: Optional[str] = None
    limitations: Optional[str] = None
    contributions: Optional[str] = None
    research_questions: Optional[str] = None
    hypotheses: Optional[str] = None
    statistical_data: Optional[str] = None
    sample_size: Optional[str] = None
    experimental_design: Optional[str] = None
    data_collection: Optional[str] = None
    analysis_techniques: Optional[str] = None
    research_gaps: Optional[str] = None
    future_research: Optional[str] = None
    practical_implications: Optional[str] = None
    theoretical_contributions: Optional[str] = None

class AnalysisResult(BaseModel):
    """Pydantic model for analysis results"""
    papers_analyzed: int = 0
    analysis_summary: str = ""
    papers: List[Dict[str, Any]] = Field(default_factory=list)
    themes: List[str] = Field(default_factory=list)
    research_gaps: List[str] = Field(default_factory=list)
    future_research: str = ""
    impact_assessment: Dict[str, Any] = Field(default_factory=dict)
    comprehensive_synthesis: Optional[str] = None
    detailed_content: Optional[Dict[str, Any]] = None
    analysis_metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class ComprehensiveSearchState(BaseModel):
    """Main state model for the comprehensive search workflow"""
    query: str
    papers: List[PaperModel] = Field(default_factory=list)
    analysis_result: Optional[AnalysisResult] = None
    search_sources: List[str] = Field(default_factory=list)
    current_step: str = "started"
    error: Optional[str] = None
    detailed_analysis: Optional[Dict[str, Any]] = None
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    methodology_comparison: Optional[Dict[str, Any]] = None
    statistical_data: List[Dict[str, Any]] = Field(default_factory=list)
    cross_references: List[Dict[str, Any]] = Field(default_factory=list)
    source_results: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    execution_metadata: ExecutionMetadata = Field(default_factory=ExecutionMetadata)

    class Config:
        arbitrary_types_allowed = True
        
    @validator('papers', pre=True)
    def validate_papers(cls, v):
        """Ensure papers are properly converted to PaperModel instances"""
        if isinstance(v, list):
            validated_papers = []
            for paper in v:
                if isinstance(paper, dict):
                    validated_papers.append(PaperModel(**paper))
                elif isinstance(paper, PaperModel):
                    validated_papers.append(paper)
                else:
                    logger.warning(f"Invalid paper type: {type(paper)}")
            return validated_papers
        return v or []

class MultiSourceSearchWrapper:
    """Enhanced wrapper to search across multiple academic sources with comprehensive logging"""
    
    def __init__(self):
        logger.info("Initializing MultiSourceSearchWrapper")
        try:
            self.search_tool = DuckDuckGoSearchRun()
            self.arxiv_client = arxiv.Client()
            self.wikipedia_tool = WikipediaAPIWrapper()
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            logger.info("Successfully initialized all search tools")
        except Exception as e:
            logger.error(f"Failed to initialize search tools: {str(e)}")
            raise
    
    @log_paper_retrieval("Google Scholar")
    def search_google_scholar(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search Google Scholar for academic papers with enhanced logging"""
        papers = []
        search_query = f"site:scholar.google.com {query}"
        logger.debug(f"Google Scholar search query: {search_query}")
        
        try:
            results = self.search_tool.run(search_query)
            logger.debug(f"Google Scholar raw results length: {len(results)}")
            
            # Parse results with better error handling
            lines = results.split('\n')
            current_paper = {}
            
            for i, line in enumerate(lines):
                try:
                    if any(keyword in line.lower() for keyword in ['title', 'paper', 'research']):
                        if current_paper:
                            papers.append(current_paper)
                            logger.debug(f"Added paper: {current_paper.get('title', 'Unknown')[:50]}...")
                        
                        current_paper = {
                            'title': line.strip(),
                            'source': 'Google Scholar',
                            'abstract': 'Abstract not available from Google Scholar',
                            'authors': 'Authors not specified',
                            'retrieval_timestamp': datetime.now().isoformat(),
                            'query_used': query
                        }
                except Exception as line_error:
                    logger.debug(f"Error processing line {i}: {str(line_error)}")
                    continue
            
            if current_paper:
                papers.append(current_paper)
                
        except Exception as e:
            logger.error(f"Google Scholar search failed: {str(e)}")
            logger.debug(f"Google Scholar error details: {traceback.format_exc()}")
        
        return papers[:max_results]
    
    @log_paper_retrieval("Nature")
    def search_nature(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search Nature journal articles with enhanced logging and improved extraction"""
        papers = []
        search_query = f"site:nature.com {query}"
        logger.debug(f"Nature search query: {search_query}")
        
        try:
            results = self.search_tool.run(search_query)
            logger.debug(f"Nature raw results: {results}")  # Log the full results
            
            lines = results.split('\n')
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                # Loosen the filter: add any non-empty line as a paper
                paper = {
                    'title': line,
                    'source': 'Nature',
                    'abstract': 'Abstract not available from Nature search',
                    'authors': 'Authors not specified',
                    'retrieval_timestamp': datetime.now().isoformat(),
                    'query_used': query
                }
                papers.append(paper)
                logger.debug(f"Added Nature paper: {paper['title'][:50]}...")
        except Exception as e:
            logger.error(f"Nature search failed: {str(e)}")
            logger.debug(f"Nature error details: {traceback.format_exc()}")
        
        return papers[:max_results]
    
    @log_paper_retrieval("PubMed")
    def search_pubmed(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search PubMed for medical and scientific papers with enhanced logging"""
        papers = []
        search_query = f"site:pubmed.ncbi.nlm.nih.gov {query}"
        logger.debug(f"PubMed search query: {search_query}")
        
        try:
            results = self.search_tool.run(search_query)
            logger.debug(f"PubMed raw results length: {len(results)}")
            
            lines = results.split('\n')
            for i, line in enumerate(lines):
                try:
                    if any(keyword in line.lower() for keyword in ['research', 'study', 'clinical', 'trial']):
                        paper = {
                            'title': line.strip(),
                            'source': 'PubMed',
                            'abstract': 'Abstract not available from PubMed search',
                            'authors': 'Authors not specified',
                            'retrieval_timestamp': datetime.now().isoformat(),
                            'query_used': query
                        }
                        papers.append(paper)
                        logger.debug(f"Added PubMed paper: {paper['title'][:50]}...")
                except Exception as line_error:
                    logger.debug(f"Error processing PubMed line {i}: {str(line_error)}")
                    continue
                    
        except Exception as e:
            logger.error(f"PubMed search failed: {str(e)}")
            logger.debug(f"PubMed error details: {traceback.format_exc()}")
        
        return papers[:max_results]
    
    @log_paper_retrieval("ScienceDirect")
    def search_science_direct(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search ScienceDirect for academic papers with enhanced logging"""
        papers = []
        search_query = f"site:sciencedirect.com {query}"
        logger.debug(f"ScienceDirect search query: {search_query}")
        
        try:
            results = self.search_tool.run(search_query)
            logger.debug(f"ScienceDirect raw results length: {len(results)}")
            
            lines = results.split('\n')
            for i, line in enumerate(lines):
                try:
                    if any(keyword in line.lower() for keyword in ['research', 'study', 'paper', 'article']):
                        paper = {
                            'title': line.strip(),
                            'source': 'ScienceDirect',
                            'abstract': 'Abstract not available from ScienceDirect search',
                            'authors': 'Authors not specified',
                            'retrieval_timestamp': datetime.now().isoformat(),
                            'query_used': query
                        }
                        papers.append(paper)
                        logger.debug(f"Added ScienceDirect paper: {paper['title'][:50]}...")
                except Exception as line_error:
                    logger.debug(f"Error processing ScienceDirect line {i}: {str(line_error)}")
                    continue
                    
        except Exception as e:
            logger.error(f"ScienceDirect search failed: {str(e)}")
            logger.debug(f"ScienceDirect error details: {traceback.format_exc()}")
        
        return papers[:max_results]
    
    @log_paper_retrieval("IEEE Xplore")
    def search_ieee(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search IEEE Xplore for technical papers with enhanced logging"""
        papers = []
        search_query = f"site:ieeexplore.ieee.org/Xplore/home.jsp {query}"
        logger.debug(f"IEEE search query: {search_query}")
        
        try:
            results = self.search_tool.run(search_query)
            logger.debug(f"IEEE raw results length: {len(results)}")
            
            lines = results.split('\n')
            for i, line in enumerate(lines):
                try:
                    if any(keyword in line.lower() for keyword in ['research', 'study', 'paper', 'conference']):
                        paper = {
                            'title': line.strip(),
                            'source': 'IEEE Xplore',
                            'abstract': 'Abstract not available from IEEE search',
                            'authors': 'Authors not specified',
                            'retrieval_timestamp': datetime.now().isoformat(),
                            'query_used': query
                        }
                        papers.append(paper)
                        logger.debug(f"Added IEEE paper: {paper['title'][:50]}...")
                except Exception as line_error:
                    logger.debug(f"Error processing IEEE line {i}: {str(line_error)}")
                    continue
                    
        except Exception as e:
            logger.error(f"IEEE search failed: {str(e)}")
            logger.debug(f"IEEE error details: {traceback.format_exc()}")
        
        return papers[:max_results]
    
    @log_paper_retrieval("ArXiv")
    def search_arxiv(self, query: str, max_results: int = 15) -> List[PaperModel]:
        """Enhanced ArXiv search with more metadata and comprehensive logging"""
        papers = []
        logger.debug(f"ArXiv search query: {query}, max_results: {max_results}")
        
        try:
            with contextlib.suppress(ResourceWarning):
                search = arxiv.Search(query=query, max_results=max_results)
                results = list(self.arxiv_client.results(search))
                
                if not results:
                    logger.warning("No results found from ArXiv")
                    return []
                
                logger.info(f"Found {len(results)} results from ArXiv")
                
                for i, result in enumerate(results):
                    try:
                        paper_data = {
                            'title': result.title,
                            'authors': ', '.join(author.name for author in result.authors),
                            'abstract': result.summary,
                            'year': str(result.published.year),
                            'source': 'ArXiv',
                            'doi': result.entry_id,
                            'categories': ', '.join(result.categories),
                            'url': result.entry_id,
                            'published': str(result.published),
                            'retrieval_timestamp': datetime.now().isoformat(),
                            'query_used': query
                        }
                        paper = PaperModel(**paper_data)
                        papers.append(paper)
                        logger.debug(f"Added ArXiv paper {i+1}: {paper.title[:50]}...")
                    except Exception as paper_error:
                        logger.error(f"Error processing ArXiv paper {i}: {str(paper_error)}")
                        continue
                
        except Exception as e:
            logger.error(f"ArXiv search failed: {str(e)}")
            logger.debug(f"ArXiv error details: {traceback.format_exc()}")
        
        return papers
    
    @log_paper_retrieval("Web Academic")
    def search_web_academic(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search web for academic papers using DuckDuckGo with enhanced logging"""
        papers = []
        search_query = f"research papers {query} filetype:pdf"
        logger.debug(f"Web academic search query: {search_query}")
        
        try:
            results = self.search_tool.run(search_query)
            logger.debug(f"Web academic raw results length: {len(results)}")
            
            lines = results.split('\n')
            for i, line in enumerate(lines):
                try:
                    if any(keyword in line.lower() for keyword in ['paper', 'research', 'study', 'journal', 'conference']):
                        paper = {
                            'title': line.strip(),
                            'source': 'Web Search',
                            'abstract': 'Abstract not available from web search',
                            'authors': 'Authors not specified',
                            'retrieval_timestamp': datetime.now().isoformat(),
                            'query_used': query
                        }
                        papers.append(paper)
                        logger.debug(f"Added web academic paper: {paper['title'][:50]}...")
                except Exception as line_error:
                    logger.debug(f"Error processing web academic line {i}: {str(line_error)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Web academic search failed: {str(e)}")
            logger.debug(f"Web academic error details: {traceback.format_exc()}")
        
        return papers[:max_results]

class ComprehensivePaperSearchAgent:
    """
    Comprehensive AI Agent for searching and analyzing research papers across multiple sources
    with enhanced logging and error handling
    """
    
    def __init__(self):
        logger.info("Initializing ComprehensivePaperSearchAgent")
        try:
            self.groq_client = ChatGroq(
                api_key=settings.GROQ_API_KEY,
                model_name=settings.PRIMARY_MODEL
            )
            logger.info(f"Initialized Groq client with model: {settings.PRIMARY_MODEL}")
            
            # Fallback models
            self.fallback_models = [
                "llama-3.3-70b-versatile",
                "gemma2-9b-it",
                "mistral-saba-24b"
            ]
            logger.info(f"Fallback models configured: {self.fallback_models}")
            
            # Initialize multi-source search wrapper
            self.search_wrapper = MultiSourceSearchWrapper()
            logger.info("Initialized MultiSourceSearchWrapper")
            
            # Create the comprehensive LangGraph workflow
            self.workflow = self._create_workflow()
            logger.info("Created LangGraph workflow")
            
        except Exception as e:
            logger.error(f"Failed to initialize ComprehensivePaperSearchAgent: {str(e)}")
            logger.debug(f"Initialization error details: {traceback.format_exc()}")
            raise
    
    def _create_workflow(self):
        """Create comprehensive LangGraph workflow for multi-source paper search and analysis"""
        logger.info("Creating LangGraph workflow")
        
        try:
            # Define the state graph
            workflow = StateGraph(ComprehensiveSearchState)
            
            # Add comprehensive nodes
            workflow.add_node("search_multiple_sources", self._search_multiple_sources_node)
            workflow.add_node("analyze_papers", self._analyze_papers_node)
            workflow.add_node("extract_detailed_content", self._extract_detailed_content_node)
            workflow.add_node("analyze_citations", self._analyze_citations_node)
            workflow.add_node("compare_methodologies", self._compare_methodologies_node)
            workflow.add_node("extract_statistical_data", self._extract_statistical_data_node)
            workflow.add_node("find_cross_references", self._find_cross_references_node)
            workflow.add_node("assess_impact", self._assess_impact_node)
            workflow.add_node("synthesize_results", self._synthesize_results_node)
            workflow.add_node("handle_error", self._handle_error_node)
            
            # Define comprehensive workflow
            workflow.set_entry_point("search_multiple_sources")
            
            # Add conditional edges for error handling
            workflow.add_conditional_edges(
                "search_multiple_sources",
                self._should_continue,
                {
                    "continue": "analyze_papers",
                    "error": "handle_error"
                }
            )
            
            workflow.add_conditional_edges(
                "analyze_papers",
                self._should_continue,
                {
                    "continue": "extract_detailed_content",
                    "error": "handle_error"
                }
            )
            
            workflow.add_conditional_edges(
                "extract_detailed_content",
                self._should_continue,
                {
                    "continue": "analyze_citations",
                    "error": "handle_error"
                }
            )
            
            workflow.add_conditional_edges(
                "analyze_citations",
                self._should_continue,
                {
                    "continue": "compare_methodologies",
                    "error": "handle_error"
                }
            )
            
            workflow.add_conditional_edges(
                "compare_methodologies",
                self._should_continue,
                {
                    "continue": "extract_statistical_data",
                    "error": "handle_error"
                }
            )
            
            workflow.add_conditional_edges(
                "extract_statistical_data",
                self._should_continue,
                {
                    "continue": "find_cross_references",
                    "error": "handle_error"
                }
            )
            
            workflow.add_conditional_edges(
                "find_cross_references",
                self._should_continue,
                {
                    "continue": "assess_impact",
                    "error": "handle_error"
                }
            )
            
            workflow.add_conditional_edges(
                "assess_impact",
                self._should_continue,
                {
                    "continue": "synthesize_results",
                    "error": "handle_error"
                }
            )
            
            workflow.add_edge("synthesize_results", END)
            workflow.add_edge("handle_error", END)
            
            logger.info("Successfully created LangGraph workflow")
            return workflow.compile()
            
        except Exception as e:
            logger.error(f"Failed to create workflow: {str(e)}")
            logger.debug(f"Workflow creation error details: {traceback.format_exc()}")
            raise
    
    def _should_continue(self, state: ComprehensiveSearchState) -> str:
        """Determine if workflow should continue or handle error"""
        if state.error:
            logger.warning(f"Workflow error detected: {state.error}")
            return "error"
        logger.debug("Workflow continuing to next step")
        return "continue"
    
    def _convert_papers_to_models(self, papers: List[Union[Dict, PaperModel]]) -> List[PaperModel]:
        """Convert papers to PaperModel instances"""
        converted_papers = []
        for paper in papers:
            try:
                if isinstance(paper, dict):
                    converted_papers.append(PaperModel(**paper))
                elif isinstance(paper, PaperModel):
                    converted_papers.append(paper)
                else:
                    logger.warning(f"Skipping invalid paper type: {type(paper)}")
            except Exception as e:
                logger.error(f"Error converting paper to model: {str(e)}")
                continue
        return converted_papers
    
    def _convert_papers_to_dicts(self, papers: List[PaperModel]) -> List[Dict[str, Any]]:
        """Convert PaperModel instances to dictionaries for analysis"""
        return [paper.dict() for paper in papers]
    
    @log_execution_time
    async def _search_multiple_sources_node(self, state: ComprehensiveSearchState) -> ComprehensiveSearchState:
        """Search for papers across multiple academic sources with comprehensive logging"""
        try:
            query = state.query
            all_papers = []
            search_sources = []
            source_results = {}
            execution_start = time.time()
            
            logger.info(f"Starting comprehensive search for query: '{query}'")
            
            # Search ArXiv
            try:
                logger.info("Searching ArXiv...")
                arxiv_start = time.time()
                arxiv_papers = self.search_wrapper.search_arxiv(query, max_results=15)
                arxiv_time = time.time() - arxiv_start
                all_papers.extend(arxiv_papers)
                search_sources.append("ArXiv")
                source_results["arxiv"] = [paper.dict() for paper in arxiv_papers]
                logger.info(f"ArXiv search completed in {arxiv_time:.2f}s - Found {len(arxiv_papers)} papers")
            except Exception as e:
                logger.error(f"ArXiv search failed: {str(e)}")
            
            # Search other sources (convert dict results to PaperModel)
            other_sources = [
                ("Google Scholar", "search_google_scholar", "google_scholar", 10),
                ("Nature", "search_nature", "nature", 8),
                ("PubMed", "search_pubmed", "pubmed", 8),
                ("ScienceDirect", "search_science_direct", "sciencedirect", 8),
                ("IEEE Xplore", "search_ieee", "ieee", 8),
                ("Web Academic", "search_web_academic", "web_academic", 10)
            ]
            
            for source_name, method_name, result_key, max_results in other_sources:
                try:
                    logger.info(f"Searching {source_name}...")
                    source_start = time.time()
                    method = getattr(self.search_wrapper, method_name)
                    source_papers_dict = method(query, max_results=max_results)
                    source_time = time.time() - source_start
                    
                    # Convert dict results to PaperModel
                    source_papers = self._convert_papers_to_models(source_papers_dict)
                    all_papers.extend(source_papers)
                    search_sources.append(source_name)
                    source_results[result_key] = [paper.dict() for paper in source_papers]
                    logger.info(f"{source_name} search completed in {source_time:.2f}s - Found {len(source_papers)} papers")
                except Exception as e:
                    logger.error(f"{source_name} search failed: {str(e)}")
            
            # Remove duplicates and limit results
            logger.info("Processing and deduplicating papers...")
            unique_papers = self._deduplicate_papers(all_papers)
            limit = getattr(settings, 'DEFAULT_SEARCH_LIMIT', 30)
            limited_papers = unique_papers[:limit]
            
            total_time = time.time() - execution_start
            logger.info(f"Multi-source search completed in {total_time:.2f}s")
            logger.info(f"Total papers found: {len(all_papers)}, Unique papers: {len(unique_papers)}, Limited to: {len(limited_papers)}")
            logger.info(f"Sources used: {', '.join(search_sources)}")
            
            # Create execution metadata
            execution_metadata = ExecutionMetadata(
                search_duration=total_time,
                total_papers_found=len(all_papers),
                unique_papers=len(unique_papers),
                sources_searched=search_sources,
                analysis_start_time=datetime.fromtimestamp(execution_start).isoformat(),
                analysis_end_time=datetime.now().isoformat()
            )
            
            return ComprehensiveSearchState(
                query=state.query,
                papers=limited_papers,
                search_sources=search_sources,
                source_results=source_results,
                current_step="multi_source_search_completed",
                execution_metadata=execution_metadata
            )
            
        except Exception as e:
            error_msg = f"Multi-source search failed: {str(e)}"
            logger.error(error_msg)
            logger.debug(f"Multi-source search error details: {traceback.format_exc()}")
            return ComprehensiveSearchState(
                query=state.query,
                error=error_msg,
                current_step="multi_source_search_failed"
            )
    
    @log_execution_time
    async def _analyze_papers_node(self, state: ComprehensiveSearchState) -> ComprehensiveSearchState:
        """Enhanced analysis of papers using AI with comprehensive logging"""
        try:
            papers = state.papers
            if not papers:
                error_msg = "No papers found to analyze"
                logger.warning(error_msg)
                return ComprehensiveSearchState(
                    query=state.query,
                    papers=state.papers,
                    error=error_msg,
                    current_step="analysis_failed",
                    search_sources=state.search_sources,
                    source_results=state.source_results,
                    execution_metadata=state.execution_metadata
                )
            
            logger.info(f"Starting AI analysis of {len(papers)} papers")
            
            # Create enhanced analysis prompt
            analysis_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert research paper analyst. Analyze the provided papers and extract comprehensive information.
                
                For each paper, extract:
                - Abstract summary (detailed, not truncated)
                - Main findings (key discoveries and results)
                - Methodology (research approach and methods used)
                - Interventions/approaches used
                - Outcomes/results with specific data
                - Limitations and constraints
                - Key contributions to the field
                - Research questions addressed
                - Hypotheses tested
                - Statistical significance and p-values
                - Sample sizes and demographics
                - Control groups and experimental design
                - Data collection methods
                - Analysis techniques used
                - Research gaps identified
                - Future research directions
                - Practical implications
                - Theoretical contributions
                
                Also provide:
                - Overall synthesis of findings
                - Relationships between papers
                - Key themes and patterns
                - Recommendations for further research
                - Impact assessment
                
                Return the analysis in JSON format with the following structure:
                {{
                    "papers_analyzed": number,
                    "analysis_summary": "overall summary",
                    "papers": [
                        {{
                            "title": "paper title",
                            "authors": "authors",
                            "year": "year",
                            "abstract": "full abstract without truncation",
                            "main_findings": "detailed findings",
                            "methodology": "detailed methodology",
                            "interventions": "interventions",
                            "outcomes": "detailed outcomes",
                            "limitations": "limitations",
                            "contributions": "contributions",
                            "research_questions": "questions",
                            "hypotheses": "hypotheses",
                            "statistical_data": "statistical info",
                            "sample_size": "sample info",
                            "experimental_design": "design details",
                            "data_collection": "data methods",
                            "analysis_techniques": "analysis methods",
                            "research_gaps": "gaps",
                            "future_research": "future directions",
                            "practical_implications": "practical impact",
                            "theoretical_contributions": "theoretical impact"
                        }}
                    ],
                    "themes": ["theme1", "theme2"],
                    "research_gaps": ["gap1", "gap2"],
                    "future_research": "future research directions",
                    "impact_assessment": "overall impact"
                }}
                """),
                ("human", "Analyze these papers: {papers}")
            ])
            
            # Run analysis with retry logic
            chain = analysis_prompt | self.groq_client | JsonOutputParser()
            papers_text = self._format_papers_for_analysis(papers)
            
            logger.debug(f"Formatted papers text length: {len(papers_text)}")
            
            try:
                analysis_result_data = await chain.ainvoke({"papers": papers_text})
                analysis_result = AnalysisResult(**analysis_result_data) if isinstance(analysis_result_data, dict) else None
                logger.info("AI analysis completed successfully")
                logger.debug(f"Analysis result type: {type(analysis_result)}")
            except Exception as ai_error:
                logger.warning(f"Primary AI analysis failed: {str(ai_error)}")
                logger.info("Attempting fallback analysis")
                analysis_result_data = self._basic_analysis(papers)
                analysis_result = AnalysisResult(**analysis_result_data)
            
            return ComprehensiveSearchState(
                query=state.query,
                papers=state.papers,
                analysis_result=analysis_result,
                search_sources=state.search_sources,
                source_results=state.source_results,
                execution_metadata=state.execution_metadata,
                current_step="analysis_completed"
            )
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            logger.error(error_msg)
            logger.debug(f"Analysis error details: {traceback.format_exc()}")
            return ComprehensiveSearchState(
                query=state.query,
                papers=state.papers,
                error=error_msg,
                current_step="analysis_failed",
                search_sources=state.search_sources,
                source_results=state.source_results,
                execution_metadata=state.execution_metadata
            )
    
    @log_execution_time
    async def _extract_detailed_content_node(self, state: ComprehensiveSearchState) -> ComprehensiveSearchState:
        """Extract detailed content from papers including specific results and findings"""
        try:
            analysis_result = state.analysis_result
            papers = state.papers
            
            logger.info(f"Extracting detailed content from {len(papers)} papers")
            
            # Enhanced content extraction prompt
            extraction_prompt = ChatPromptTemplate.from_messages([
                ("system", """Extract detailed content from research papers including:
                
                1. Specific Results and Findings:
                - Numerical results and statistics
                - Effect sizes and confidence intervals
                - Success rates and performance metrics
                - Comparative results between groups
                
                2. Detailed Methodology:
                - Experimental procedures
                - Data collection protocols
                - Analysis methods with specific techniques
                - Quality control measures
                
                3. Key Insights:
                - Novel discoveries
                - Unexpected findings
                - Practical applications
                - Theoretical implications
                
                4. Limitations and Constraints:
                - Sample size limitations
                - Methodological constraints
                - External validity issues
                - Potential biases
                
                Return as JSON with detailed extraction for each paper.
                """),
                ("human", "Extract detailed content from: {papers}")
            ])
            
            chain = extraction_prompt | self.groq_client | JsonOutputParser()
            papers_text = self._format_papers_for_analysis(papers)
            
            try:
                detailed_content = await chain.ainvoke({"papers": papers_text})
                logger.info("Detailed content extraction completed successfully")
            except Exception as ai_error:
                logger.warning(f"Detailed content extraction failed: {str(ai_error)}")
                detailed_content = {"error": "AI extraction failed", "papers": []}
            
            # Update analysis result with detailed content
            if analysis_result:
                analysis_result["detailed_content"] = detailed_content
            
            return ComprehensiveSearchState(
                query=state.query,
                papers=state.papers,
                analysis_result=analysis_result,
                search_sources=state.search_sources,
                source_results=state.source_results,
                execution_metadata=state.execution_metadata,
                detailed_analysis=detailed_content,
                current_step="detailed_extraction_completed"
            )
            
        except Exception as e:
            error_msg = f"Detailed extraction failed: {str(e)}"
            logger.error(error_msg)
            logger.debug(f"Detailed extraction error details: {traceback.format_exc()}")
            return ComprehensiveSearchState(
                query=state.query,
                papers=state.papers,
                error=error_msg,
                current_step="detailed_extraction_failed"
            )
    
    @log_execution_time
    async def _analyze_citations_node(self, state: ComprehensiveSearchState) -> ComprehensiveSearchState:
        """Analyze citations and references between papers"""
        try:
            papers = state.papers
            logger.info(f"Analyzing citations for {len(papers)} papers")
            
            # Citation analysis prompt
            citation_prompt = ChatPromptTemplate.from_messages([
                ("system", """Analyze citations and references between research papers:
                
                1. Identify common references across papers
                2. Find papers that cite each other
                3. Identify influential papers in the field
                4. Map the research landscape and connections
                5. Find gaps in citation patterns
                
                Return as JSON with citation analysis.
                """),
                ("human", "Analyze citations for: {papers}")
            ])
            
            chain = citation_prompt | self.groq_client | JsonOutputParser()
            papers_text = self._format_papers_for_analysis(papers)
            
            try:
                citation_analysis = await chain.ainvoke({"papers": papers_text})
                logger.info("Citation analysis completed successfully")
            except Exception as ai_error:
                logger.warning(f"Citation analysis failed: {str(ai_error)}")
                citation_analysis = {"citations": [], "error": "AI citation analysis failed"}
            
            return ComprehensiveSearchState(
                query=state.query,
                papers=state.papers,
                analysis_result=state.analysis_result,
                search_sources=state.search_sources,
                source_results=state.source_results,
                execution_metadata=state.execution_metadata,
                detailed_analysis=state.detailed_analysis,
                citations=citation_analysis.get("citations", []),
                current_step="citation_analysis_completed"
            )
            
        except Exception as e:
            error_msg = f"Citation analysis failed: {str(e)}"
            logger.error(error_msg)
            logger.debug(f"Citation analysis error details: {traceback.format_exc()}")
            return ComprehensiveSearchState(
                query=state.query,
                papers=state.papers,
                error=error_msg,
                current_step="citation_analysis_failed"
            )
    
    @log_execution_time
    async def _compare_methodologies_node(self, state: ComprehensiveSearchState) -> ComprehensiveSearchState:
        """Compare methodologies across papers"""
        try:
            papers = state.papers
            logger.info(f"Comparing methodologies across {len(papers)} papers")
            
            # Methodology comparison prompt
            methodology_prompt = ChatPromptTemplate.from_messages([
                ("system", """Compare research methodologies across papers:
                
                1. Identify common methodological approaches
                2. Compare strengths and weaknesses of different methods
                3. Find methodological innovations
                4. Identify methodological gaps
                5. Assess methodological rigor
                6. Compare sample sizes and demographics
                7. Analyze experimental designs
                8. Compare data collection methods
                
                Return as JSON with methodology comparison.
                """),
                ("human", "Compare methodologies for: {papers}")
            ])
            
            chain = methodology_prompt | self.groq_client | JsonOutputParser()
            papers_text = self._format_papers_for_analysis(papers)
            
            try:
                methodology_comparison = await chain.ainvoke({"papers": papers_text})
                logger.info("Methodology comparison completed successfully")
            except Exception as ai_error:
                logger.warning(f"Methodology comparison failed: {str(ai_error)}")
                methodology_comparison = {"error": "AI methodology comparison failed"}
            
            return ComprehensiveSearchState(
                query=state.query,
                papers=state.papers,
                analysis_result=state.analysis_result,
                search_sources=state.search_sources,
                source_results=state.source_results,
                execution_metadata=state.execution_metadata,
                detailed_analysis=state.detailed_analysis,
                citations=state.citations,
                methodology_comparison=methodology_comparison,
                current_step="methodology_comparison_completed"
            )
            
        except Exception as e:
            error_msg = f"Methodology comparison failed: {str(e)}"
            logger.error(error_msg)
            logger.debug(f"Methodology comparison error details: {traceback.format_exc()}")
            return ComprehensiveSearchState(
                query=state.query,
                papers=state.papers,
                error=error_msg,
                current_step="methodology_comparison_failed"
            )
    
    @log_execution_time
    async def _extract_statistical_data_node(self, state: ComprehensiveSearchState) -> ComprehensiveSearchState:
        """Extract statistical data and numerical results from papers"""
        try:
            papers = state.papers
            logger.info(f"Extracting statistical data from {len(papers)} papers")
            
            # Statistical data extraction prompt
            stats_prompt = ChatPromptTemplate.from_messages([
                ("system", """Extract statistical data and numerical results from research papers:
                
                1. Sample sizes and demographics
                2. Statistical tests used
                3. P-values and significance levels
                4. Effect sizes and confidence intervals
                5. Correlation coefficients
                6. Regression results
                7. Mean, median, standard deviation
                8. Success rates and performance metrics
                9. Comparative statistics
                10. Statistical power and effect sizes
                
                Return as JSON with statistical data for each paper.
                """),
                ("human", "Extract statistical data from: {papers}")
            ])
            
            chain = stats_prompt | self.groq_client | JsonOutputParser()
            papers_text = self._format_papers_for_analysis(papers)
            
            try:
                statistical_data = await chain.ainvoke({"papers": papers_text})
                logger.info("Statistical data extraction completed successfully")
            except Exception as ai_error:
                logger.warning(f"Statistical data extraction failed: {str(ai_error)}")
                statistical_data = {"statistical_data": [], "error": "AI statistical extraction failed"}
            
            return ComprehensiveSearchState(
                query=state.query,
                papers=state.papers,
                analysis_result=state.analysis_result,
                search_sources=state.search_sources,
                source_results=state.source_results,
                execution_metadata=state.execution_metadata,
                detailed_analysis=state.detailed_analysis,
                citations=state.citations,
                methodology_comparison=state.methodology_comparison,
                statistical_data=statistical_data.get("statistical_data", []),
                current_step="statistical_extraction_completed"
            )
            
        except Exception as e:
            error_msg = f"Statistical extraction failed: {str(e)}"
            logger.error(error_msg)
            logger.debug(f"Statistical extraction error details: {traceback.format_exc()}")
            return ComprehensiveSearchState(
                query=state.query,
                papers=state.papers,
                error=error_msg,
                current_step="statistical_extraction_failed"
            )
    
    @log_execution_time
    async def _find_cross_references_node(self, state: ComprehensiveSearchState) -> ComprehensiveSearchState:
        """Find cross-references and connections between papers"""
        try:
            papers = state.papers
            logger.info(f"Finding cross-references for {len(papers)} papers")
            
            # Cross-reference analysis prompt
            cross_ref_prompt = ChatPromptTemplate.from_messages([
                ("system", """Find cross-references and connections between research papers:
                
                1. Papers that build on each other
                2. Conflicting findings
                3. Complementary research
                4. Research gaps that could be filled
                5. Potential collaborations
                6. Research trends and patterns
                7. Emerging themes
                8. Future research opportunities
                
                Return as JSON with cross-reference analysis.
                """),
                ("human", "Find cross-references for: {papers}")
            ])
            
            chain = cross_ref_prompt | self.groq_client | JsonOutputParser()
            papers_text = self._format_papers_for_analysis(papers)
            
            try:
                cross_references = await chain.ainvoke({"papers": papers_text})
                logger.info("Cross-reference analysis completed successfully")
            except Exception as ai_error:
                logger.warning(f"Cross-reference analysis failed: {str(ai_error)}")
                cross_references = {"cross_references": [], "error": "AI cross-reference analysis failed"}
            
            return ComprehensiveSearchState(
                query=state.query,
                papers=state.papers,
                analysis_result=state.analysis_result,
                search_sources=state.search_sources,
                source_results=state.source_results,
                execution_metadata=state.execution_metadata,
                detailed_analysis=state.detailed_analysis,
                citations=state.citations,
                methodology_comparison=state.methodology_comparison,
                statistical_data=state.statistical_data,
                cross_references=cross_references.get("cross_references", []),
                current_step="cross_reference_analysis_completed"
            )
            
        except Exception as e:
            error_msg = f"Cross-reference analysis failed: {str(e)}"
            logger.error(error_msg)
            logger.debug(f"Cross-reference analysis error details: {traceback.format_exc()}")
            return ComprehensiveSearchState(
                query=state.query,
                papers=state.papers,
                error=error_msg,
                current_step="cross_reference_analysis_failed"
            )
    
    @log_execution_time
    async def _assess_impact_node(self, state: ComprehensiveSearchState) -> ComprehensiveSearchState:
        """Assess the impact and significance of research findings"""
        try:
            papers = state.papers
            logger.info(f"Assessing impact for {len(papers)} papers")
            
            # Impact assessment prompt
            impact_prompt = ChatPromptTemplate.from_messages([
                ("system", """Assess the impact and significance of research findings:
                
                1. Theoretical impact on the field
                2. Practical applications and implications
                3. Policy implications
                4. Industry relevance
                5. Societal impact
                6. Innovation potential
                7. Commercial viability
                8. Long-term significance
                9. Potential for follow-up research
                10. Risk assessment and limitations
                
                Return as JSON with impact assessment for each paper.
                """),
                ("human", "Assess impact for: {papers}")
            ])
            
            chain = impact_prompt | self.groq_client | JsonOutputParser()
            papers_text = self._format_papers_for_analysis(papers)
            
            try:
                impact_assessment = await chain.ainvoke({"papers": papers_text})
                logger.info("Impact assessment completed successfully")
            except Exception as ai_error:
                logger.warning(f"Impact assessment failed: {str(ai_error)}")
                impact_assessment = {"error": "AI impact assessment failed"}
            
            # Update analysis result with impact assessment
            if state.analysis_result:
                state.analysis_result["impact_assessment"] = impact_assessment
            
            return ComprehensiveSearchState(
                query=state.query,
                papers=state.papers,
                analysis_result=state.analysis_result,
                search_sources=state.search_sources,
                source_results=state.source_results,
                execution_metadata=state.execution_metadata,
                detailed_analysis=state.detailed_analysis,
                citations=state.citations,
                methodology_comparison=state.methodology_comparison,
                statistical_data=state.statistical_data,
                cross_references=state.cross_references,
                current_step="impact_assessment_completed"
            )
            
        except Exception as e:
            error_msg = f"Impact assessment failed: {str(e)}"
            logger.error(error_msg)
            logger.debug(f"Impact assessment error details: {traceback.format_exc()}")
            return ComprehensiveSearchState(
                query=state.query,
                papers=state.papers,
                error=error_msg,
                current_step="impact_assessment_failed"
            )
    
    @log_execution_time
    async def _synthesize_results_node(self, state: ComprehensiveSearchState) -> ComprehensiveSearchState:
        """Synthesize all analysis results into comprehensive summary"""
        try:
            logger.info("Synthesizing all analysis results")
            
            # Create comprehensive synthesis
            synthesis_prompt = ChatPromptTemplate.from_messages([
                ("system", """Create a comprehensive synthesis of all research analysis including:
                
                1. Executive Summary
                2. Key Findings and Results
                3. Methodology Insights
                4. Statistical Evidence
                5. Citation Network Analysis
                6. Cross-Reference Patterns
                7. Impact Assessment
                8. Research Gaps and Opportunities
                9. Recommendations for Future Research
                10. Practical Applications
                
                Provide actionable insights and clear conclusions.
                """),
                ("human", "Synthesize all analysis results: {analysis}")
            ])
            
            chain = synthesis_prompt | self.groq_client
            
            try:
                synthesis = await chain.ainvoke({"analysis": json.dumps(state.analysis_result, indent=2)})
                logger.info("Synthesis completed successfully")
            except Exception as ai_error:
                logger.warning(f"AI synthesis failed: {str(ai_error)}")
                synthesis = type('obj', (object,), {'content': 'Synthesis failed due to AI error'})
            
            # Create final comprehensive result
            final_result = {
                **state.analysis_result,
                "comprehensive_synthesis": synthesis.content,
                "analysis_metadata": {
                    "total_papers": len(state.papers),
                    "sources_used": state.search_sources,
                    "source_results": state.source_results,
                    "analysis_date": datetime.now().isoformat(),
                    "query": state.query,
                    "detailed_analysis": state.detailed_analysis,
                    "citations": state.citations,
                    "methodology_comparison": state.methodology_comparison,
                    "statistical_data": state.statistical_data,
                    "cross_references": state.cross_references,
                    "execution_metadata": state.execution_metadata
                }
            }
            
            logger.info("Final comprehensive result created")
            
            return ComprehensiveSearchState(
                query=state.query,
                papers=state.papers,
                analysis_result=final_result,
                search_sources=state.search_sources,
                source_results=state.source_results,
                execution_metadata=state.execution_metadata,
                detailed_analysis=state.detailed_analysis,
                citations=state.citations,
                methodology_comparison=state.methodology_comparison,
                statistical_data=state.statistical_data,
                cross_references=state.cross_references,
                current_step="synthesis_completed"
            )
            
        except Exception as e:
            error_msg = f"Synthesis failed: {str(e)}"
            logger.error(error_msg)
            logger.debug(f"Synthesis error details: {traceback.format_exc()}")
            return ComprehensiveSearchState(
                query=state.query,
                papers=state.papers,
                error=error_msg,
                current_step="synthesis_failed"
            )
    
    @log_execution_time
    async def _handle_error_node(self, state: ComprehensiveSearchState) -> ComprehensiveSearchState:
        """Handle errors in the workflow"""
        error = state.error
        logger.warning(f"Handling workflow error: {error}")
        
        # Try fallback analysis
        try:
            papers = state.papers
            if papers:
                logger.info("Attempting fallback basic analysis")
                basic_analysis = self._basic_analysis(papers)
                logger.info("Fallback analysis completed")
                return ComprehensiveSearchState(
                    query=state.query,
                    papers=state.papers,
                    analysis_result=basic_analysis,
                    search_sources=state.search_sources,
                    source_results=state.source_results,
                    execution_metadata=state.execution_metadata,
                    current_step="error_handled_with_fallback"
                )
        except Exception as fallback_error:
            logger.error(f"Fallback analysis also failed: {str(fallback_error)}")
        
        logger.error("All analysis attempts failed")
        return ComprehensiveSearchState(
            query=state.query,
            papers=state.papers,
            analysis_result={
                "error": error,
                "papers": [],
                "analysis_summary": f"Analysis failed: {error}"
            },
            search_sources=state.search_sources,
            source_results=state.source_results,
            execution_metadata=state.execution_metadata,
            current_step="error_handled"
        )
    
    def _deduplicate_papers(self, papers: List[PaperModel]) -> List[PaperModel]:
        """Remove duplicate papers based on title similarity with logging"""
        logger.info(f"Deduplicating {len(papers)} papers")
        unique_papers = []
        seen_titles = set()
        duplicates_count = 0
        
        for paper in papers:
            title = paper.title.lower().strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_papers.append(paper)
            else:
                duplicates_count += 1
                logger.debug(f"Duplicate paper detected: {title[:50]}...")
        
        logger.info(f"Removed {duplicates_count} duplicates, {len(unique_papers)} unique papers remain")
        return unique_papers
    
    def _format_papers_for_analysis(self, papers: List[PaperModel]) -> str:
        """Enhanced format papers data for analysis with full abstracts and logging"""
        logger.debug(f"Formatting {len(papers)} papers for analysis")
        formatted_papers = []
        
        for i, paper in enumerate(papers, 1):
            try:
                paper_text = f"Paper {i}:\n"
                paper_text += f"Title: {paper.title}\n"
                paper_text += f"Authors: {paper.authors}\n"
                paper_text += f"Year: {paper.year}\n"
                paper_text += f"Abstract: {paper.abstract}\n"
                paper_text += f"Source: {paper.source}\n"
                if paper.doi:
                    paper_text += f"DOI: {paper.doi}\n"
                if paper.categories:
                    paper_text += f"Categories: {paper.categories}\n"
                if paper.url:
                    paper_text += f"URL: {paper.url}\n"
                paper_text += "-" * 50 + "\n"
                
                formatted_papers.append(paper_text)
            except Exception as format_error:
                logger.warning(f"Error formatting paper {i}: {str(format_error)}")
                continue
        
        formatted_text = "\n".join(formatted_papers)
        logger.debug(f"Formatted papers text length: {len(formatted_text)} characters")
        return formatted_text
    
    def _basic_analysis(self, papers: List[PaperModel]) -> Dict[str, Any]:
        """Enhanced basic analysis when AI models fail with logging"""
        logger.info(f"Performing basic analysis on {len(papers)} papers")
        
        try:
            # Convert papers to dicts for compatibility
            papers_dict = self._convert_papers_to_dicts(papers)
            
            # Extract basic statistics
            sources = [paper.source for paper in papers]
            source_counts = {}
            for source in sources:
                source_counts[source] = source_counts.get(source, 0) + 1
            
            years = [paper.year for paper in papers if paper.year and paper.year.isdigit()]
            unique_years = list(set(years))
            
            basic_result = {
                "papers_analyzed": len(papers),
                "analysis_summary": f"Basic analysis completed for {len(papers)} papers from {len(source_counts)} sources",
                "papers": papers_dict,
                "extraction_date": datetime.now().isoformat(),
                "error": "AI analysis failed, showing basic results",
                "source_distribution": source_counts,
                "years_covered": sorted(unique_years),
                "themes": [],
                "research_gaps": [],
                "future_research": "Further analysis required",
                "impact_assessment": {}
            }
            
            logger.info("Basic analysis completed successfully")
            return basic_result
            
        except Exception as e:
            logger.error(f"Basic analysis failed: {str(e)}")
            return {
                "papers_analyzed": 0,
                "analysis_summary": "Analysis completely failed",
                "papers": [],
                "error": f"All analysis methods failed: {str(e)}",
                "themes": [],
                "research_gaps": [],
                "future_research": "",
                "impact_assessment": {}
            }
    
    @log_execution_time
    async def run_full_analysis(self, query: str, max_papers: int = 30) -> Dict[str, Any]:
        """Run complete comprehensive analysis workflow with logging"""
        logger.info(f"Starting full analysis for query: '{query}' with max_papers: {max_papers}")
        state = ComprehensiveSearchState(query=query)
        
        try:
            final_state = await self.workflow.ainvoke(state)
            logger.info(f"Workflow returned type: {type(final_state)}")
            # Ensure final_state is a ComprehensiveSearchState
            if isinstance(final_state, dict):
                logger.warning("Workflow returned a dict, converting to ComprehensiveSearchState using parse_obj.")
                final_state = ComprehensiveSearchState.parse_obj(final_state)
            logger.info("Full analysis workflow completed successfully")
            # Log final statistics
            if final_state.analysis_result:
                logger.info(f"Analysis result: {final_state.analysis_result}")
                total_papers = final_state.analysis_result.analysis_metadata.get('total_papers', 0) if final_state.analysis_result.analysis_metadata else 0
                sources_used = final_state.analysis_result.analysis_metadata.get('sources_used', []) if final_state.analysis_result.analysis_metadata else []
                logger.info(f"Analysis completed: {total_papers} papers from sources: {', '.join(sources_used)}")
            return final_state.analysis_result.dict() if final_state.analysis_result else {}
            
        except Exception as e:
            error_msg = f"Workflow failed: {str(e)}"
            logger.error(error_msg)
            logger.debug(f"Workflow error details: {traceback.format_exc()}")
            return {
                "error": error_msg,
                "papers": [],
                "analysis_summary": "Analysis failed",
                "themes": [],
                "research_gaps": [],
                "future_research": "",
                "impact_assessment": {}
            }
    
    @log_execution_time
    def export_to_csv(self, analysis_result: Dict[str, Any], filename: str = None) -> str:
        """
        Enhanced export analysis results to CSV file with detailed information and logging
        """
        logger.info("Starting CSV export")
        
        try:
            if not analysis_result:
                logger.error("Analysis result is None or empty")
                raise ValueError("No analysis result to export")
            
            if not filename:
                # Use just the query name for the filename
                query_name = analysis_result.get('query', 'research').replace(' ', '_').replace('/', '_').replace('\\', '_')
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{query_name}_{timestamp}.csv"
            
            # Ensure exports directory exists
            exports_dir = getattr(settings, 'EXPORTS_DIR', './exports')
            filepath = os.path.join(exports_dir, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            logger.info(f"Exporting to file: {filepath}")
            
            # Prepare enhanced data for CSV export
            csv_data = []
            
            # Add paper details with enhanced information
            papers = analysis_result.get('papers', [])
            logger.info(f"Processing {len(papers)} papers for CSV export")
            
            for i, paper in enumerate(papers, 1):
                try:
                    # Handle different paper formats
                    if isinstance(paper, dict):
                        paper_dict = paper
                    elif hasattr(paper, 'dict'):
                        paper_dict = paper.dict()
                    else:
                        paper_dict = paper.__dict__ if hasattr(paper, '__dict__') else {}
                    
                    # Ensure paper_dict is not None
                    if paper_dict is None:
                        paper_dict = {}
                    
                    row = {
                        'Paper_Number': i,
                        'Title': paper_dict.get('title', ''),
                        'Authors': paper_dict.get('authors', ''),
                        'Year': paper_dict.get('year', ''),
                        'Abstract': paper_dict.get('abstract', ''),
                        'Source': paper_dict.get('source', ''),
                        'DOI': paper_dict.get('doi', ''),
                        'Categories': paper_dict.get('categories', ''),
                        'URL': paper_dict.get('url', ''),
                        'Main_Findings': paper_dict.get('main_findings', ''),
                        'Methodology': paper_dict.get('methodology', ''),
                        'Interventions': paper_dict.get('interventions', ''),
                        'Outcomes': paper_dict.get('outcomes', ''),
                        'Limitations': paper_dict.get('limitations', ''),
                        'Contributions': paper_dict.get('contributions', ''),
                        'Research_Questions': paper_dict.get('research_questions', ''),
                        'Hypotheses': paper_dict.get('hypotheses', ''),
                        'Statistical_Data': paper_dict.get('statistical_data', ''),
                        'Sample_Size': paper_dict.get('sample_size', ''),
                        'Experimental_Design': paper_dict.get('experimental_design', ''),
                        'Data_Collection': paper_dict.get('data_collection', ''),
                        'Analysis_Techniques': paper_dict.get('analysis_techniques', ''),
                        'Research_Gaps': paper_dict.get('research_gaps', ''),
                        'Future_Research': paper_dict.get('future_research', ''),
                        'Practical_Implications': paper_dict.get('practical_implications', ''),
                        'Theoretical_Contributions': paper_dict.get('theoretical_contributions', ''),
                        'Retrieval_Timestamp': paper_dict.get('retrieval_timestamp', ''),
                        'Query_Used': paper_dict.get('query_used', '')
                    }
                    csv_data.append(row)
                    logger.debug(f"Processed paper {i}: {paper_dict.get('title', 'Unknown')[:50]}...")
                except Exception as paper_error:
                    logger.warning(f"Error processing paper {i} for CSV: {str(paper_error)}")
                    continue
            
            # Write to CSV with proper error handling
            try:
                with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                    if csv_data:
                        fieldnames = csv_data[0].keys()
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(csv_data)
                        logger.info(f"Successfully exported {len(csv_data)} papers to {filepath}")
                    else:
                        logger.warning("No data to export to CSV")
                        # Write empty CSV with headers
                        fieldnames = ['Title', 'Authors', 'Year', 'Abstract', 'Source', 'Error']
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerow({'Error': 'No papers found to export'})
            except Exception as write_error:
                logger.error(f"Failed to write CSV file: {str(write_error)}")
                raise
            
            # Log export summary with null checks
            metadata = analysis_result.get('analysis_metadata', {}) or {}
            total_papers = metadata.get('total_papers', len(papers)) if metadata else len(papers)
            sources_used = metadata.get('sources_used', []) if metadata else []
            
            logger.info(f"CSV Export Summary:")
            logger.info(f"  - Total papers exported: {len(csv_data)}")
            logger.info(f"  - Original papers found: {total_papers}")
            logger.info(f"  - Sources used: {', '.join(sources_used) if sources_used else 'Unknown'}")
            logger.info(f"  - Export file: {filepath}")
            
            return filepath
            
        except Exception as e:
            error_msg = f"CSV export failed: {str(e)}"
            logger.error(error_msg)
            logger.debug(f"CSV export error details: {traceback.format_exc()}")
            raise Exception(error_msg)
    
    @log_execution_time
    def export_detailed_report(self, analysis_result: Dict[str, Any], filename: str = None) -> str:
        """
        Export a detailed analysis report to a text file with comprehensive logging
        """
        logger.info("Starting detailed report export")
        
        try:
            if not filename:
                query_name = analysis_result.get('query', 'research').replace(' ', '_').replace('/', '_').replace('\\', '_')
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{query_name}_detailed_report_{timestamp}.txt"
            
            # Ensure exports directory exists
            exports_dir = getattr(settings, 'EXPORTS_DIR', './exports')
            filepath = os.path.join(exports_dir, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            logger.info(f"Exporting detailed report to: {filepath}")
            
            # Create detailed report content
            report_content = []
            
            # Header
            report_content.append("="*80)
            report_content.append("COMPREHENSIVE RESEARCH PAPER ANALYSIS REPORT")
            report_content.append("="*80)
            report_content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_content.append(f"Query: {analysis_result.get('query', 'Unknown')}")
            report_content.append("")
            
            # Metadata
            metadata = analysis_result.get('analysis_metadata', {})
            report_content.append("ANALYSIS METADATA")
            report_content.append("-" * 40)
            report_content.append(f"Total Papers: {metadata.get('total_papers', 'Unknown')}")
            report_content.append(f"Sources Used: {', '.join(metadata.get('sources_used', []))}")
            report_content.append(f"Analysis Date: {metadata.get('analysis_date', 'Unknown')}")
            
            # Execution metadata
            exec_metadata = metadata.get('execution_metadata', {})
            if exec_metadata:
                report_content.append(f"Search Duration: {exec_metadata.get('search_duration', 'Unknown')} seconds")
                report_content.append(f"Unique Papers Found: {exec_metadata.get('unique_papers', 'Unknown')}")
            report_content.append("")
            
            # Executive Summary
            synthesis = analysis_result.get('comprehensive_synthesis', '')
            if synthesis:
                report_content.append("EXECUTIVE SUMMARY")
                report_content.append("-" * 40)
                report_content.append(synthesis)
                report_content.append("")
            
            # Analysis Summary
            analysis_summary = analysis_result.get('analysis_summary', '')
            if analysis_summary:
                report_content.append("ANALYSIS SUMMARY")
                report_content.append("-" * 40)
                report_content.append(analysis_summary)
                report_content.append("")
            
            # Papers Details
            papers = analysis_result.get('papers', [])
            if papers:
                report_content.append("DETAILED PAPER ANALYSIS")
                report_content.append("-" * 40)
                
                for i, paper in enumerate(papers, 1):
                    report_content.append(f"\nPAPER {i}:")
                    report_content.append(f"Title: {paper.get('title', 'Unknown')}")
                    report_content.append(f"Authors: {paper.get('authors', 'Unknown')}")
                    report_content.append(f"Year: {paper.get('year', 'Unknown')}")
                    report_content.append(f"Source: {paper.get('source', 'Unknown')}")
                    
                    if paper.get('abstract'):
                        report_content.append(f"Abstract: {paper.get('abstract')}")
                    
                    if paper.get('main_findings'):
                        report_content.append(f"Main Findings: {paper.get('main_findings')}")
                    
                    if paper.get('methodology'):
                        report_content.append(f"Methodology: {paper.get('methodology')}")
                    
                    if paper.get('limitations'):
                        report_content.append(f"Limitations: {paper.get('limitations')}")
                    
                    report_content.append("-" * 30)
            
            # Themes and Patterns
            themes = analysis_result.get('themes', [])
            if themes:
                report_content.append("\nKEY THEMES AND PATTERNS")
                report_content.append("-" * 40)
                for theme in themes:
                    report_content.append(f"• {theme}")
                report_content.append("")
            
            # Research Gaps
            gaps = analysis_result.get('research_gaps', [])
            if gaps:
                report_content.append("RESEARCH GAPS IDENTIFIED")
                report_content.append("-" * 40)
                for gap in gaps:
                    report_content.append(f"• {gap}")
                report_content.append("")
            
            # Future Research
            future_research = analysis_result.get('future_research', '')
            if future_research:
                report_content.append("FUTURE RESEARCH DIRECTIONS")
                report_content.append("-" * 40)
                report_content.append(future_research)
                report_content.append("")
            
            # Impact Assessment
            impact = analysis_result.get('impact_assessment', {})
            if impact and not impact.get('error'):
                report_content.append("IMPACT ASSESSMENT")
                report_content.append("-" * 40)
                report_content.append(json.dumps(impact, indent=2))
                report_content.append("")
            
            # Source Distribution
            source_results = metadata.get('source_results', {})
            if source_results:
                report_content.append("SOURCE DISTRIBUTION")
                report_content.append("-" * 40)
                for source, papers_list in source_results.items():
                    report_content.append(f"{source.title()}: {len(papers_list)} papers")
                report_content.append("")
            
            # Error Information
            if analysis_result.get('error'):
                report_content.append("ERRORS AND WARNINGS")
                report_content.append("-" * 40)
                report_content.append(f"Error: {analysis_result.get('error')}")
                report_content.append("")
            
            # Footer
            report_content.append("="*80)
            report_content.append("END OF REPORT")
            report_content.append("="*80)
            
            # Write report to file
            try:
                with open(filepath, 'w', encoding='utf-8') as report_file:
                    report_file.write('\n'.join(report_content))
                logger.info(f"Successfully exported detailed report to {filepath}")
            except Exception as write_error:
                logger.error(f"Failed to write report file: {str(write_error)}")
                raise
            
            # Log export summary
            logger.info(f"Detailed Report Export Summary:")
            logger.info(f"  - Report sections: {len([line for line in report_content if line.startswith('-')])}")
            logger.info(f"  - Total content length: {len(''.join(report_content))} characters")
            logger.info(f"  - Export file: {filepath}")
            
            return filepath
            
        except Exception as e:
            error_msg = f"Detailed report export failed: {str(e)}"
            logger.error(error_msg)
            logger.debug(f"Report export error details: {traceback.format_exc()}")
            raise Exception(error_msg)
    
    @log_execution_time
    def export_json(self, analysis_result: Dict[str, Any], filename: str = None) -> str:
        """
        Export analysis results to JSON file with comprehensive logging
        """
        logger.info("Starting JSON export")
        
        try:
            if not filename:
                query_name = analysis_result.get('query', 'research').replace(' ', '_').replace('/', '_').replace('\\', '_')
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{query_name}_analysis_{timestamp}.json"
            
            # Ensure exports directory exists
            exports_dir = getattr(settings, 'EXPORTS_DIR', './exports')
            filepath = os.path.join(exports_dir, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            logger.info(f"Exporting JSON to: {filepath}")
            
            # Add export metadata
            export_data = {
                **analysis_result,
                "export_metadata": {
                    "export_timestamp": datetime.now().isoformat(),
                    "export_format": "JSON",
                    "export_version": "1.0"
                }
            }
            
            # Write JSON file
            try:
                with open(filepath, 'w', encoding='utf-8') as json_file:
                    json.dump(export_data, json_file, indent=2, ensure_ascii=False, default=str)
                logger.info(f"Successfully exported JSON to {filepath}")
            except Exception as write_error:
                logger.error(f"Failed to write JSON file: {str(write_error)}")
                raise
            
            # Log export summary
            file_size = os.path.getsize(filepath)
            logger.info(f"JSON Export Summary:")
            logger.info(f"  - File size: {file_size} bytes")
            logger.info(f"  - Papers included: {len(analysis_result.get('papers', []))}")
            logger.info(f"  - Export file: {filepath}")
            
            return filepath
            
        except Exception as e:
            error_msg = f"JSON export failed: {str(e)}"
            logger.error(error_msg)
            logger.debug(f"JSON export error details: {traceback.format_exc()}")
            raise Exception(error_msg)
    
    def get_analysis_statistics(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the analysis with logging
        """
        logger.info("Generating analysis statistics")
        
        try:
            papers = analysis_result.get('papers', [])
            metadata = analysis_result.get('analysis_metadata', {})
            
            # Basic statistics
            stats = {
                "total_papers": len(papers),
                "sources_used": metadata.get('sources_used', []),
                "analysis_date": metadata.get('analysis_date', ''),
                "query": analysis_result.get('query', ''),
                "has_error": bool(analysis_result.get('error'))
            }
            
            # Source distribution
            source_counts = {}
            years = []
            for paper in papers:
                # Handle both dict and PaperModel instances
                if isinstance(paper, dict):
                    source = paper.get('source', 'Unknown')
                    year = paper.get('year', '')
                elif hasattr(paper, 'source'):
                    source = paper.source
                    year = paper.year
                else:
                    source = 'Unknown'
                    year = ''
                
                source_counts[source] = source_counts.get(source, 0) + 1
                
                if year and str(year).isdigit():
                    years.append(int(year))
            
            stats["source_distribution"] = source_counts
            
            # Year statistics
            if years:
                stats["year_range"] = {
                    "earliest": min(years),
                    "latest": max(years),
                    "span": max(years) - min(years) if len(years) > 1 else 0
                }
            
            # Content statistics
            abstracts = []
            for paper in papers:
                if isinstance(paper, dict):
                    abstract = paper.get('abstract', '')
                elif hasattr(paper, 'abstract'):
                    abstract = paper.abstract
                else:
                    abstract = ''
                
                if abstract:
                    abstracts.append(abstract)
            
            stats["content_stats"] = {
                "papers_with_abstracts": len(abstracts),
                "average_abstract_length": sum(len(abstract) for abstract in abstracts) // len(abstracts) if abstracts else 0
            }
            
            # Analysis completeness
            analysis_steps = [
                'analysis_result', 'detailed_analysis', 'citations', 
                'methodology_comparison', 'statistical_data', 'cross_references'
            ]
            completed_steps = sum(1 for step in analysis_steps if analysis_result.get(step))
            stats["analysis_completeness"] = {
                "completed_steps": completed_steps,
                "total_steps": len(analysis_steps),
                "completion_percentage": (completed_steps / len(analysis_steps)) * 100
            }
            
            # Execution metadata
            exec_metadata = metadata.get('execution_metadata', {})
            if exec_metadata:
                stats["execution_stats"] = exec_metadata
            
            logger.info(f"Generated statistics for {stats['total_papers']} papers")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to generate statistics: {str(e)}")
            return {"error": f"Statistics generation failed: {str(e)}"}

# Keep the original class name for backward compatibility
PaperSearchAgent = ComprehensivePaperSearchAgent

# Usage example and utility functions
def create_search_agent() -> ComprehensivePaperSearchAgent:
    """
    Factory function to create a new search agent instance with logging
    """
    logger.info("Creating new ComprehensivePaperSearchAgent instance")
    try:
        agent = ComprehensivePaperSearchAgent()
        logger.info("Successfully created search agent")
        return agent
    except Exception as e:
        logger.error(f"Failed to create search agent: {str(e)}")
        raise

async def search_and_analyze(query: str, max_papers: int = 30, export_formats: List[str] = None) -> Dict[str, Any]:
    """
    High-level function to search and analyze papers with optional export
    
    Args:
        query: Search query
        max_papers: Maximum number of papers to analyze
        export_formats: List of export formats ('csv', 'json', 'report')
    
    Returns:
        Analysis results dictionary
    """
    logger.info(f"Starting search and analysis for: '{query}'")
    
    try:
        # Create agent and run analysis
        agent = create_search_agent()
        results = await agent.run_full_analysis(query, max_papers)
        
        # Export if requested
        exported_files = {}
        if export_formats:
            logger.info(f"Exporting results in formats: {export_formats}")
            
            if 'csv' in export_formats:
                try:
                    csv_file = agent.export_to_csv(results)
                    exported_files['csv'] = csv_file
                except Exception as e:
                    logger.error(f"CSV export failed: {str(e)}")
            
            if 'json' in export_formats:
                try:
                    json_file = agent.export_json(results)
                    exported_files['json'] = json_file
                except Exception as e:
                    logger.error(f"JSON export failed: {str(e)}")
            
            if 'report' in export_formats:
                try:
                    report_file = agent.export_detailed_report(results)
                    exported_files['report'] = report_file
                except Exception as e:
                    logger.error(f"Report export failed: {str(e)}")
        
        # Add export information to results
        if exported_files:
            results['exported_files'] = exported_files
        
        # Add statistics
        try:
            stats = agent.get_analysis_statistics(results)
            results['statistics'] = stats
        except Exception as e:
            logger.error(f"Statistics generation failed: {str(e)}")
        
        logger.info("Search and analysis completed successfully")
        return results
        
    except Exception as e:
        error_msg = f"Search and analysis failed: {str(e)}"
        logger.error(error_msg)
        logger.debug(f"Search and analysis error details: {traceback.format_exc()}")
        return {
            "error": error_msg,
            "papers": [],
            "analysis_summary": "Analysis failed"
        }

# Configuration validation
def validate_configuration():
    """
    Validate that all required configurations are present
    """
    logger.info("Validating configuration")
    
    required_settings = ['GROQ_API_KEY', 'PRIMARY_MODEL']
    missing_settings = []
    
    for setting in required_settings:
        if not hasattr(settings, setting) or not getattr(settings, setting):
            missing_settings.append(setting)
    
    if missing_settings:
        error_msg = f"Missing required settings: {', '.join(missing_settings)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info("Configuration validation passed")

# Initialize validation on import
try:
    validate_configuration()
    logger.info("Paper search agent module loaded successfully")
except Exception as e:
    logger.error(f"Module initialization failed: {str(e)}")
    # Don't raise here to allow partial functionality