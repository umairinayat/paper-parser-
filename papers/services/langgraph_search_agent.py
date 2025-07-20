import os
import json
import csv
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import contextlib
from pydantic import BaseModel, Field
import arxiv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper

from langgraph.graph import StateGraph, END

from django.conf import settings

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SearchState(BaseModel):
    query: str
    papers: List[Dict] = Field(default_factory=list)
    analysis_result: Optional[Dict] = None
    search_sources: List[str] = Field(default_factory=list)
    current_step: str = "started"
    error: Optional[str] = None

class CustomSearchWrapper:
    """Custom wrapper to handle resource cleanup and deprecation warnings"""
    
    def __init__(self):
        logger.info("Initializing CustomSearchWrapper")
        self.search_tool = DuckDuckGoSearchRun()
        self.arxiv_client = arxiv.Client()
        logger.info("CustomSearchWrapper initialized successfully")
    
    def search_web(self, query: str) -> str:
        """Search web with proper resource cleanup"""
        logger.info(f"Starting web search for query: {query}")
        try:
            with contextlib.suppress(ResourceWarning):
                result = self.search_tool.run(query)
                logger.info(f"Web search completed successfully")
                return result
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return ""
    
    def search_arxiv(self, query: str) -> str:
        """Search ArXiv with proper resource cleanup using newer client"""
        logger.info(f"Starting ArXiv search for query: {query}")
        try:
            with contextlib.suppress(ResourceWarning):
                search = arxiv.Search(query=query, max_results=10)
                results = list(self.arxiv_client.results(search))
                
                if not results:
                    logger.warning("No ArXiv results found")
                    return ""
                
                formatted_results = []
                for result in results:
                    formatted_results.append(f"Title: {result.title}")
                    formatted_results.append(f"Authors: {', '.join(author.name for author in result.authors)}")
                    formatted_results.append(f"Abstract: {result.summary}")
                    formatted_results.append(f"Published: {result.published}")
                    formatted_results.append("-" * 50)
                
                logger.info(f"ArXiv search completed with {len(results)} results")
                return "\n".join(formatted_results)
        except Exception as e:
            logger.error(f"ArXiv search error: {e}")
            return ""

class LangGraphSearchAgent:
    def __init__(self):
        logger.info("Initializing LangGraphSearchAgent")
        self.groq_client = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model_name="llama-3.3-70b-versatile"
        )
        self.search_wrapper = CustomSearchWrapper()
        self.workflow = self._create_workflow()
        logger.info("LangGraphSearchAgent initialized successfully")

    def _create_workflow(self):
        logger.info("Creating LangGraph workflow")
        workflow = StateGraph(SearchState)
        workflow.add_node("search_papers", self._search_papers_node)
        workflow.add_node("analyze_papers", self._analyze_papers_node)
        workflow.add_node("synthesize_results", self._synthesize_results_node)
        workflow.add_node("handle_error", self._handle_error_node)

        workflow.set_entry_point("search_papers")

        workflow.add_conditional_edges("search_papers", self._should_continue, {
            "continue": "analyze_papers",
            "error": "handle_error"
        })

        workflow.add_conditional_edges("analyze_papers", self._should_continue, {
            "continue": "synthesize_results",
            "error": "handle_error"
        })

        workflow.add_conditional_edges("synthesize_results", self._should_continue, {
            "continue": END,
            "error": "handle_error"
        })

        workflow.add_edge("handle_error", END)

        compiled_workflow = workflow.compile()
        logger.info("LangGraph workflow created and compiled successfully")
        return compiled_workflow

    def _should_continue(self, state: SearchState) -> str:
        logger.debug(f"Checking if workflow should continue. Error: {state.error}")
        result = "error" if state.error else "continue"
        logger.debug(f"Workflow decision: {result}")
        return result

    async def _search_papers_node(self, state: SearchState) -> SearchState:
        logger.info(f"Starting search papers node with query: {state.query}")
        try:
            query = state.query
            papers = []
            search_sources = []

            logger.info(f"Searching for papers about: {query}")

            try:
                arxiv_results = self.search_wrapper.search_arxiv(query)
                if arxiv_results:
                    arxiv_papers = self._parse_arxiv_results(arxiv_results)
                    papers.extend(arxiv_papers)
                    search_sources.append("ArXiv")
                    logger.info(f"Found {len(arxiv_papers)} papers from ArXiv")
            except Exception as e:
                logger.error(f"ArXiv error: {e}")

            try:
                web_results = self.search_wrapper.search_web(f"research papers {query}")
                if web_results:
                    web_papers = self._parse_web_results(web_results)
                    papers.extend(web_papers)
                    search_sources.append("Web Search")
                    logger.info(f"Found {len(web_papers)} papers from web search")
            except Exception as e:
                logger.error(f"Web search error: {e}")

            unique_papers = self._deduplicate_papers(papers)
            limit = getattr(settings, 'DEFAULT_SEARCH_LIMIT', 10)
            limited_papers = unique_papers[:limit]

            logger.info(f"Search completed. Found {len(limited_papers)} unique papers")
            return SearchState(
                query=state.query,
                papers=limited_papers,
                search_sources=search_sources,
                current_step="search_completed"
            )
        except Exception as e:
            logger.error(f"Search papers node failed: {e}")
            return SearchState(
                query=state.query,
                error=f"Search failed: {str(e)}",
                current_step="search_failed"
            )

    async def _analyze_papers_node(self, state: SearchState) -> SearchState:
        logger.info(f"Starting analyze papers node with {len(state.papers)} papers")
        try:
            papers = state.papers
            if not papers:
                logger.warning("No papers to analyze")
                return SearchState(
                    query=state.query,
                    papers=state.papers,
                    error="No papers to analyze",
                    current_step="analysis_failed"
                )

            logger.info("Creating analysis prompt")
            analysis_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert research paper analyst. Analyze the provided papers and extract key information.

For each paper, extract:
- Abstract summary (2-3 sentences)
- Main findings (key discoveries)
- Methodology (research approach)
- Research gaps identified
- Future research directions

Also provide:
- Overall synthesis of findings
- Key themes and patterns
- Recommendations for further research

Return the analysis in JSON format with the following structure:
{{
    "papers_analyzed": number,
    "analysis_summary": "overall summary",
    "papers": [
        {{
            "title": "paper title",
            "authors": "authors",
            "year": "year",
            "abstract": "abstract",
            "main_findings": "findings",
            "methodology": "methodology",
            "research_gaps": "gaps",
            "future_research": "future directions"
        }}
    ],
    "themes": ["theme1", "theme2"],
    "research_gaps": ["gap1", "gap2"],
    "future_research": "future research directions"
}}
"""),
                ("human", "Analyze these papers:\n\n{papers}")
            ])
            logger.info("Analysis prompt created successfully")

            logger.info("Creating analysis chain")
            chain = analysis_prompt | self.groq_client | JsonOutputParser()
            papers_text = self._format_papers_for_analysis(papers)
            logger.info(f"Formatted papers text length: {len(papers_text)}")

            try:
                logger.info("Invoking AI analysis chain")
                result = await chain.ainvoke({"papers": papers_text})
                logger.info("AI analysis completed successfully")
            except Exception as e:
                logger.error(f"Analysis chain failed: {e}", exc_info=True)
                logger.info("Using fallback analysis")
                result = self._basic_analysis(papers)

            logger.info("Creating final SearchState with analysis result")
            return SearchState(
                query=state.query,
                papers=state.papers,
                analysis_result=result,
                search_sources=state.search_sources,
                current_step="analysis_completed"
            )
        except Exception as e:
            logger.error(f"Analyze papers node failed: {e}", exc_info=True)
            return SearchState(
                query=state.query,
                papers=state.papers,
                error=f"Analysis failed: {str(e)}",
                current_step="analysis_failed"
            )

    async def _synthesize_results_node(self, state: SearchState) -> SearchState:
        try:
            analysis_result = state.analysis_result

            synthesis_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert research synthesizer. Create a comprehensive summary of the research findings."),
                ("human", "Create a synthesis of these research findings: {analysis}")
            ])

            chain = synthesis_prompt | self.groq_client
            synthesis = await chain.ainvoke({"analysis": json.dumps(analysis_result, indent=2)})

            analysis_result["synthesis"] = synthesis.content
            analysis_result["final_summary"] = {
                "papers_found": len(analysis_result.get("papers", [])),
                "key_themes": analysis_result.get("themes", []),
                "research_gaps": analysis_result.get("research_gaps", []),
                "recommendations": analysis_result.get("future_research", "")
            }

            return SearchState(
                query=state.query,
                papers=state.papers,
                analysis_result=analysis_result,
                search_sources=state.search_sources,
                current_step="synthesis_completed"
            )
        except Exception as e:
            return SearchState(
                query=state.query,
                papers=state.papers,
                error=f"Synthesis failed: {str(e)}",
                current_step="synthesis_failed"
            )

    async def _handle_error_node(self, state: SearchState) -> SearchState:
        error = state.error
        try:
            papers = state.papers
            if papers:
                basic_analysis = self._basic_analysis(papers)
                return SearchState(
                    query=state.query,
                    papers=state.papers,
                    analysis_result=basic_analysis,
                    search_sources=state.search_sources,
                    current_step="error_handled_with_fallback"
                )
        except Exception:
            pass
        return SearchState(
            query=state.query,
            papers=state.papers,
            analysis_result={
                "error": error,
                "papers": [],
                "analysis_summary": f"Analysis failed: {error}"
            },
            search_sources=state.search_sources,
            current_step="error_handled"
        )

    def _parse_arxiv_results(self, results: str) -> List[Dict]:
        papers = []
        try:
            lines = results.split('\n')
            current_paper = {}
            for line in lines:
                if 'Title:' in line:
                    if current_paper:
                        papers.append(current_paper)
                    current_paper = {'title': line.split('Title:')[1].strip(), 'source': 'ArXiv'}
                elif 'Authors:' in line:
                    current_paper['authors'] = line.split('Authors:')[1].strip()
                elif 'Abstract:' in line:
                    current_paper['abstract'] = line.split('Abstract:')[1].strip()
                elif 'Published:' in line:
                    current_paper['year'] = line.split('Published:')[1].strip()[:4]
            if current_paper:
                papers.append(current_paper)
        except Exception as e:
            print(f"ArXiv parse error: {e}")
        return papers

    def _parse_web_results(self, results: str) -> List[Dict]:
        papers = []
        try:
            for line in results.split('\n'):
                if any(k in line.lower() for k in ['paper', 'research', 'study', 'journal']):
                    papers.append({
                        'title': line.strip(),
                        'source': 'Web Search',
                        'abstract': 'Abstract not available'
                    })
        except Exception as e:
            print(f"Web parse error: {e}")
        return papers

    def _deduplicate_papers(self, papers: List[Dict]) -> List[Dict]:
        seen = set()
        unique = []
        for paper in papers:
            title = paper.get('title', '').lower()
            if title and title not in seen:
                seen.add(title)
                unique.append(paper)
        return unique

    def _format_papers_for_analysis(self, papers: List[Dict]) -> str:
        result = []
        for i, paper in enumerate(papers, 1):
            result.append(f"Paper {i}:\nTitle: {paper.get('title', '')}\nAuthors: {paper.get('authors', 'N/A')}\nYear: {paper.get('year', 'N/A')}\nAbstract: {paper.get('abstract', '')}\nSource: {paper.get('source', 'N/A')}\n" + "-"*50)
        return '\n'.join(result)

    def _basic_analysis(self, papers: List[Dict]) -> Dict[str, Any]:
        return {
            "papers_analyzed": len(papers),
            "analysis_summary": "Basic analysis fallback used.",
            "papers": papers,
            "extraction_date": datetime.now().isoformat(),
            "error": "AI analysis failed."
        }

    async def run_full_analysis(self, query: str, max_papers: int = 20) -> Dict[str, Any]:
        logger.info(f"Starting full analysis for query: {query}")
        try:
            logger.info("Creating initial SearchState")
            state = SearchState(query=query)
            logger.info(f"Initial state created: {state}")
            
            logger.info("Invoking LangGraph workflow")
            final_state = await self.workflow.ainvoke(state)
            logger.info(f"Workflow completed. Final state: {final_state}")
            
            # Handle both SearchState and dict return types
            if hasattr(final_state, 'analysis_result'):
                result = final_state.analysis_result or {}
            elif isinstance(final_state, dict):
                result = final_state.get('analysis_result', {})
            else:
                result = {}
                
            logger.info(f"Returning analysis result: {result}")
            return result
        except Exception as e:
            logger.error(f"Workflow failed with error: {e}", exc_info=True)
            return {
                "error": f"Workflow failed: {str(e)}",
                "papers": [],
                "analysis_summary": "Analysis failed"
            }

    def export_to_csv(self, analysis_result: Dict[str, Any], filename: str = None) -> str:
        if not filename:
            filename = f"paper_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(settings.EXPORTS_DIR, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        csv_data = []
        for paper in analysis_result.get('papers', []):
            csv_data.append({
                'Title': paper.get('title', ''),
                'Authors': paper.get('authors', ''),
                'Year': paper.get('year', 'N/A'),
                'Abstract': paper.get('abstract', 'N/A'),
                'Source': paper.get('source', 'N/A'),
                'Main Findings': paper.get('main_findings', 'N/A'),
                'Methodology': paper.get('methodology', 'N/A'),
                'Research Gaps': paper.get('research_gaps', 'N/A'),
                'Future Research': paper.get('future_research', 'N/A')
            })

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
            writer.writeheader()
            writer.writerows(csv_data)

        return filepath
