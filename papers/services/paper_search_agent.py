import os
import json
import csv
import asyncio
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import contextlib
import arxiv
from pydantic import BaseModel, Field

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

# Define enhanced state structure for LangGraph
class SearchState(BaseModel):
    query: str
    papers: List[Dict] = Field(default_factory=list)
    analysis_result: Optional[Dict] = None
    search_sources: List[str] = Field(default_factory=list)
    current_step: str = "started"
    error: Optional[str] = None
    detailed_analysis: Optional[Dict] = None
    citations: List[Dict] = Field(default_factory=list)
    methodology_comparison: Optional[Dict] = None
    statistical_data: List[Dict] = Field(default_factory=list)
    cross_references: List[Dict] = Field(default_factory=list)

class CustomSearchWrapper:
    """Enhanced wrapper to handle resource cleanup and deprecation warnings"""
    
    def __init__(self):
        self.search_tool = DuckDuckGoSearchRun()
        self.arxiv_client = arxiv.Client()
        self.wikipedia_tool = WikipediaAPIWrapper()
    
    def search_web(self, query: str) -> str:
        """Search web with proper resource cleanup"""
        try:
            with contextlib.suppress(ResourceWarning):
                return self.search_tool.run(query)
        except Exception as e:
            print(f"Web search error: {e}")
            return ""
    
    def search_arxiv(self, query: str) -> str:
        """Search ArXiv with proper resource cleanup using newer client"""
        try:
            with contextlib.suppress(ResourceWarning):
                search = arxiv.Search(query=query, max_results=15)
                results = list(self.arxiv_client.results(search))
                
                if not results:
                    return ""
                
                formatted_results = []
                for result in results:
                    formatted_results.append(f"Title: {result.title}")
                    formatted_results.append(f"Authors: {', '.join(author.name for author in result.authors)}")
                    formatted_results.append(f"Abstract: {result.summary}")
                    formatted_results.append(f"Published: {result.published}")
                    formatted_results.append(f"DOI: {result.entry_id}")
                    formatted_results.append(f"Categories: {', '.join(result.categories)}")
                    formatted_results.append("-" * 50)
                
                return "\n".join(formatted_results)
        except Exception as e:
            print(f"ArXiv search error: {e}")
            return ""
    
    def search_wikipedia(self, query: str) -> str:
        """Search Wikipedia with proper resource cleanup"""
        try:
            with contextlib.suppress(ResourceWarning):
                return self.wikipedia_tool.run(query)
        except Exception as e:
            print(f"Wikipedia search error: {e}")
            return ""

class EnhancedPaperSearchAgent:
    """
    Enhanced AI Agent for searching and analyzing research papers with detailed content extraction
    """
    
    def __init__(self):
        self.groq_client = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model_name=settings.PRIMARY_MODEL
        )
        
        # Fallback models
        self.fallback_models = [
            "llama-3.3-70b-versatile",
            "gemma2-9b-it",
            "mistral-saba-24b"
        ]
        
        # Initialize tools with custom wrapper
        self.search_wrapper = CustomSearchWrapper()
        
        # Create the enhanced LangGraph workflow
        self.workflow = self._create_workflow()
    
    def _create_workflow(self):
        """Create enhanced LangGraph workflow for paper search and analysis"""
        
        # Define the state graph
        workflow = StateGraph(SearchState)
        
        # Add enhanced nodes
        workflow.add_node("search_papers", self._search_papers_node)
        workflow.add_node("analyze_papers", self._analyze_papers_node)
        workflow.add_node("extract_detailed_content", self._extract_detailed_content_node)
        workflow.add_node("analyze_citations", self._analyze_citations_node)
        workflow.add_node("compare_methodologies", self._compare_methodologies_node)
        workflow.add_node("extract_statistical_data", self._extract_statistical_data_node)
        workflow.add_node("find_cross_references", self._find_cross_references_node)
        workflow.add_node("assess_impact", self._assess_impact_node)
        workflow.add_node("synthesize_results", self._synthesize_results_node)
        workflow.add_node("handle_error", self._handle_error_node)
        
        # Define enhanced workflow
        workflow.set_entry_point("search_papers")
        
        # Add conditional edges for error handling
        workflow.add_conditional_edges(
            "search_papers",
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
        
        return workflow.compile()
    
    def _should_continue(self, state: SearchState) -> str:
        """Determine if workflow should continue or handle error"""
        if state.error:
            return "error"
        return "continue"
    
    async def _search_papers_node(self, state: SearchState) -> SearchState:
        """Enhanced search for papers using multiple sources"""
        try:
            query = state.query
            papers = []
            search_sources = []
            
            # Search using ArXiv
            try:
                arxiv_results = self.search_wrapper.search_arxiv(query)
                if arxiv_results:
                    arxiv_papers = self._parse_arxiv_results(arxiv_results)
                    papers.extend(arxiv_papers)
                    search_sources.append("ArXiv")
            except Exception as e:
                print(f"ArXiv search error: {e}")
            
            # Search using DuckDuckGo
            try:
                web_results = self.search_wrapper.search_web(f"research papers {query}")
                if web_results:
                    web_papers = self._parse_web_results(web_results)
                    papers.extend(web_papers)
                    search_sources.append("Web Search")
            except Exception as e:
                print(f"Web search error: {e}")
            
            # Search using Wikipedia for context
            try:
                wiki_results = self.search_wrapper.search_wikipedia(query)
                if wiki_results:
                    wiki_papers = self._parse_wikipedia_results(wiki_results)
                    papers.extend(wiki_papers)
                    search_sources.append("Wikipedia")
            except Exception as e:
                print(f"Wikipedia search error: {e}")
            
            # Remove duplicates and limit results
            unique_papers = self._deduplicate_papers(papers)
            
            return SearchState(
                query=state.query,
                papers=unique_papers[:settings.DEFAULT_SEARCH_LIMIT],
                search_sources=search_sources,
                current_step="search_completed"
            )
            
        except Exception as e:
            return SearchState(
                query=state.query,
                error=f"Search failed: {str(e)}",
                current_step="search_failed"
            )
    
    async def _analyze_papers_node(self, state: SearchState) -> SearchState:
        """Enhanced analysis of papers using AI"""
        try:
            papers = state.papers
            if not papers:
                return SearchState(
                    query=state.query,
                    papers=state.papers,
                    error="No papers found to analyze",
                    current_step="analysis_failed"
                )
            
            # Create enhanced analysis prompt
            analysis_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert research paper analyst. Analyze the provided papers and extract comprehensive information.
                
                For each paper, extract:
                - Abstract summary (2-3 sentences)
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
                            "abstract": "abstract",
                            "main_findings": "findings",
                            "methodology": "methodology",
                            "interventions": "interventions",
                            "outcomes": "outcomes",
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
            
            # Run analysis
            chain = analysis_prompt | self.groq_client | JsonOutputParser()
            papers_text = self._format_papers_for_analysis(papers)
            analysis_result = await chain.ainvoke({"papers": papers_text})
            
            return SearchState(
                query=state.query,
                papers=state.papers,
                analysis_result=analysis_result,
                search_sources=state.search_sources,
                current_step="analysis_completed"
            )
            
        except Exception as e:
            return SearchState(
                query=state.query,
                papers=state.papers,
                error=f"Analysis failed: {str(e)}",
                current_step="analysis_failed"
            )
    
    async def _extract_detailed_content_node(self, state: SearchState) -> SearchState:
        """Extract detailed content from papers including specific results and findings"""
        try:
            analysis_result = state.analysis_result
            papers = state.papers
            
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
            detailed_content = await chain.ainvoke({"papers": papers_text})
            
            # Update analysis result with detailed content
            analysis_result["detailed_content"] = detailed_content
            
            return SearchState(
                query=state.query,
                papers=state.papers,
                analysis_result=analysis_result,
                search_sources=state.search_sources,
                detailed_analysis=detailed_content,
                current_step="detailed_extraction_completed"
            )
            
        except Exception as e:
            return SearchState(
                query=state.query,
                papers=state.papers,
                error=f"Detailed extraction failed: {str(e)}",
                current_step="detailed_extraction_failed"
            )
    
    async def _analyze_citations_node(self, state: SearchState) -> SearchState:
        """Analyze citations and references between papers"""
        try:
            papers = state.papers
            
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
            citation_analysis = await chain.ainvoke({"papers": papers_text})
            
            return SearchState(
                query=state.query,
                papers=state.papers,
                analysis_result=state.analysis_result,
                search_sources=state.search_sources,
                detailed_analysis=state.detailed_analysis,
                citations=citation_analysis.get("citations", []),
                current_step="citation_analysis_completed"
            )
            
        except Exception as e:
            return SearchState(
                query=state.query,
                papers=state.papers,
                error=f"Citation analysis failed: {str(e)}",
                current_step="citation_analysis_failed"
            )
    
    async def _compare_methodologies_node(self, state: SearchState) -> SearchState:
        """Compare methodologies across papers"""
        try:
            papers = state.papers
            
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
            methodology_comparison = await chain.ainvoke({"papers": papers_text})
            
            return SearchState(
                query=state.query,
                papers=state.papers,
                analysis_result=state.analysis_result,
                search_sources=state.search_sources,
                detailed_analysis=state.detailed_analysis,
                citations=state.citations,
                methodology_comparison=methodology_comparison,
                current_step="methodology_comparison_completed"
            )
            
        except Exception as e:
            return SearchState(
                query=state.query,
                papers=state.papers,
                error=f"Methodology comparison failed: {str(e)}",
                current_step="methodology_comparison_failed"
            )
    
    async def _extract_statistical_data_node(self, state: SearchState) -> SearchState:
        """Extract statistical data and numerical results from papers"""
        try:
            papers = state.papers
            
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
            statistical_data = await chain.ainvoke({"papers": papers_text})
            
            return SearchState(
                query=state.query,
                papers=state.papers,
                analysis_result=state.analysis_result,
                search_sources=state.search_sources,
                detailed_analysis=state.detailed_analysis,
                citations=state.citations,
                methodology_comparison=state.methodology_comparison,
                statistical_data=statistical_data.get("statistical_data", []),
                current_step="statistical_extraction_completed"
            )
            
        except Exception as e:
            return SearchState(
                query=state.query,
                papers=state.papers,
                error=f"Statistical extraction failed: {str(e)}",
                current_step="statistical_extraction_failed"
            )
    
    async def _find_cross_references_node(self, state: SearchState) -> SearchState:
        """Find cross-references and connections between papers"""
        try:
            papers = state.papers
            
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
            cross_references = await chain.ainvoke({"papers": papers_text})
            
            return SearchState(
                query=state.query,
                papers=state.papers,
                analysis_result=state.analysis_result,
                search_sources=state.search_sources,
                detailed_analysis=state.detailed_analysis,
                citations=state.citations,
                methodology_comparison=state.methodology_comparison,
                statistical_data=state.statistical_data,
                cross_references=cross_references.get("cross_references", []),
                current_step="cross_reference_analysis_completed"
            )
            
        except Exception as e:
            return SearchState(
                query=state.query,
                papers=state.papers,
                error=f"Cross-reference analysis failed: {str(e)}",
                current_step="cross_reference_analysis_failed"
            )
    
    async def _assess_impact_node(self, state: SearchState) -> SearchState:
        """Assess the impact and significance of research findings"""
        try:
            papers = state.papers
            
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
            impact_assessment = await chain.ainvoke({"papers": papers_text})
            
            # Update analysis result with impact assessment
            if state.analysis_result:
                state.analysis_result["impact_assessment"] = impact_assessment
            
            return SearchState(
                query=state.query,
                papers=state.papers,
                analysis_result=state.analysis_result,
                search_sources=state.search_sources,
                detailed_analysis=state.detailed_analysis,
                citations=state.citations,
                methodology_comparison=state.methodology_comparison,
                statistical_data=state.statistical_data,
                cross_references=state.cross_references,
                current_step="impact_assessment_completed"
            )
            
        except Exception as e:
            return SearchState(
                query=state.query,
                papers=state.papers,
                error=f"Impact assessment failed: {str(e)}",
                current_step="impact_assessment_failed"
            )
    
    async def _synthesize_results_node(self, state: SearchState) -> SearchState:
        """Synthesize all analysis results into comprehensive summary"""
        try:
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
            synthesis = await chain.ainvoke({"analysis": json.dumps(state.analysis_result, indent=2)})
            
            # Create final comprehensive result
            final_result = {
                **state.analysis_result,
                "comprehensive_synthesis": synthesis.content,
                "analysis_metadata": {
                    "total_papers": len(state.papers),
                    "sources_used": state.search_sources,
                    "analysis_date": datetime.now().isoformat(),
                    "query": state.query,
                    "detailed_analysis": state.detailed_analysis,
                    "citations": state.citations,
                    "methodology_comparison": state.methodology_comparison,
                    "statistical_data": state.statistical_data,
                    "cross_references": state.cross_references
                }
            }
            
            return SearchState(
                query=state.query,
                papers=state.papers,
                analysis_result=final_result,
                search_sources=state.search_sources,
                detailed_analysis=state.detailed_analysis,
                citations=state.citations,
                methodology_comparison=state.methodology_comparison,
                statistical_data=state.statistical_data,
                cross_references=state.cross_references,
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
        """Handle errors in the workflow"""
        error = state.error
        
        # Try fallback analysis
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
        """Enhanced parse ArXiv results with more metadata"""
        papers = []
        try:
            lines = results.split('\n')
            current_paper = {}
            
            for line in lines:
                if 'Title:' in line:
                    if current_paper:
                        papers.append(current_paper)
                    current_paper = {
                        'title': line.split('Title:')[1].strip(),
                        'source': 'ArXiv'
                    }
                elif 'Authors:' in line and current_paper:
                    current_paper['authors'] = line.split('Authors:')[1].strip()
                elif 'Abstract:' in line and current_paper:
                    current_paper['abstract'] = line.split('Abstract:')[1].strip()
                elif 'Published:' in line and current_paper:
                    current_paper['year'] = line.split('Published:')[1].strip()[:4]
                elif 'DOI:' in line and current_paper:
                    current_paper['doi'] = line.split('DOI:')[1].strip()
                elif 'Categories:' in line and current_paper:
                    current_paper['categories'] = line.split('Categories:')[1].strip()
            
            if current_paper:
                papers.append(current_paper)
                
        except Exception as e:
            print(f"Error parsing ArXiv results: {e}")
        
        return papers
    
    def _parse_web_results(self, results: str) -> List[Dict]:
        """Enhanced parse web search results"""
        papers = []
        try:
            lines = results.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in ['paper', 'research', 'study', 'journal', 'conference']):
                    papers.append({
                        'title': line.strip(),
                        'source': 'Web Search',
                        'abstract': 'Abstract not available from web search',
                        'authors': 'Authors not specified'
                    })
        except Exception as e:
            print(f"Error parsing web results: {e}")
        
        return papers
    
    def _parse_wikipedia_results(self, results: str) -> List[Dict]:
        """Enhanced parse Wikipedia results for research context"""
        papers = []
        try:
            # Extract research-related information from Wikipedia
            if 'research' in results.lower() or 'study' in results.lower():
                papers.append({
                    'title': f"Research context from Wikipedia",
                    'source': 'Wikipedia',
                    'abstract': results[:500] + "..." if len(results) > 500 else results,
                    'authors': 'Wikipedia contributors',
                    'year': datetime.now().year
                })
        except Exception as e:
            print(f"Error parsing Wikipedia results: {e}")
        
        return papers
    
    def _deduplicate_papers(self, papers: List[Dict]) -> List[Dict]:
        """Remove duplicate papers based on title similarity"""
        unique_papers = []
        seen_titles = set()
        
        for paper in papers:
            title = paper.get('title', '').lower().strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_papers.append(paper)
        
        return unique_papers
    
    def _format_papers_for_analysis(self, papers: List[Dict]) -> str:
        """Enhanced format papers data for analysis"""
        formatted_papers = []
        
        for i, paper in enumerate(papers, 1):
            paper_text = f"Paper {i}:\n"
            paper_text += f"Title: {paper.get('title', 'N/A')}\n"
            paper_text += f"Authors: {paper.get('authors', 'N/A')}\n"
            paper_text += f"Year: {paper.get('year', 'N/A')}\n"
            paper_text += f"Abstract: {paper.get('abstract', 'N/A')}\n"
            paper_text += f"Source: {paper.get('source', 'N/A')}\n"
            if paper.get('doi'):
                paper_text += f"DOI: {paper.get('doi', 'N/A')}\n"
            if paper.get('categories'):
                paper_text += f"Categories: {paper.get('categories', 'N/A')}\n"
            paper_text += "-" * 50 + "\n"
            
            formatted_papers.append(paper_text)
        
        return "\n".join(formatted_papers)
    
    def _basic_analysis(self, papers: List[Dict]) -> Dict[str, Any]:
        """Enhanced basic analysis when AI models fail"""
        return {
            "papers_analyzed": len(papers),
            "analysis_summary": "Basic analysis completed",
            "papers": papers,
            "extraction_date": datetime.now().isoformat(),
            "error": "AI analysis failed, showing basic results",
            "detailed_analysis": {},
            "citations": [],
            "methodology_comparison": {},
            "statistical_data": [],
            "cross_references": []
        }
    
    async def run_full_analysis(self, query: str, max_papers: int = 20) -> Dict[str, Any]:
        """Run complete enhanced analysis workflow"""
        state = SearchState(query=query)
        
        try:
            final_state = await self.workflow.ainvoke(state)
            return final_state.analysis_result or {}
        except Exception as e:
            return {
                "error": f"Workflow failed: {str(e)}",
                "papers": [],
                "analysis_summary": "Analysis failed"
            }
    
    def export_to_csv(self, analysis_result: Dict[str, Any], filename: str = None) -> str:
        """
        Enhanced export analysis results to CSV file with detailed information
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_paper_analysis_{timestamp}.csv"
        
        filepath = os.path.join(settings.EXPORTS_DIR, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare enhanced data for CSV export
        csv_data = []
        
        # Add paper details with enhanced information
        for paper in analysis_result.get('papers', []):
            row = {
                'Title': paper.get('title', ''),
                'Authors': paper.get('authors', ''),
                'Year': paper.get('year', ''),
                'Abstract': paper.get('abstract', ''),
                'Source': paper.get('source', ''),
                'DOI': paper.get('doi', ''),
                'Categories': paper.get('categories', ''),
                'Main Findings': paper.get('main_findings', ''),
                'Methodology': paper.get('methodology', ''),
                'Interventions': paper.get('interventions', ''),
                'Outcomes': paper.get('outcomes', ''),
                'Limitations': paper.get('limitations', ''),
                'Contributions': paper.get('contributions', ''),
                'Research Questions': paper.get('research_questions', ''),
                'Hypotheses': paper.get('hypotheses', ''),
                'Statistical Data': paper.get('statistical_data', ''),
                'Sample Size': paper.get('sample_size', ''),
                'Experimental Design': paper.get('experimental_design', ''),
                'Data Collection': paper.get('data_collection', ''),
                'Analysis Techniques': paper.get('analysis_techniques', ''),
                'Research Gaps': paper.get('research_gaps', ''),
                'Future Research': paper.get('future_research', ''),
                'Practical Implications': paper.get('practical_implications', ''),
                'Theoretical Contributions': paper.get('theoretical_contributions', '')
            }
            csv_data.append(row)
        
        # Write to CSV
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            if csv_data:
                fieldnames = csv_data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
        
        return filepath

# Keep the original class name for backward compatibility
PaperSearchAgent = EnhancedPaperSearchAgent 