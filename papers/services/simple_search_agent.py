import os
import json
import csv
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import contextlib
import arxiv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper

from django.conf import settings

class CustomSearchWrapper:
    """Custom wrapper to handle resource cleanup and deprecation warnings"""
    
    def __init__(self):
        self.search_tool = DuckDuckGoSearchRun()
        self.arxiv_client = arxiv.Client()
    
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
                search = arxiv.Search(query=query, max_results=10)
                results = list(self.arxiv_client.results(search))
                
                if not results:
                    return ""
                
                formatted_results = []
                for result in results:
                    formatted_results.append(f"Title: {result.title}")
                    formatted_results.append(f"Authors: {', '.join(author.name for author in result.authors)}")
                    formatted_results.append(f"Abstract: {result.summary}")
                    formatted_results.append(f"Published: {result.published}")
                    formatted_results.append("-" * 50)
                
                return "\n".join(formatted_results)
        except Exception as e:
            print(f"ArXiv search error: {e}")
            return ""

class SimplePaperSearchAgent:
    """
    Simplified AI Agent for searching and analyzing research papers
    """
    
    def __init__(self):
        self.groq_client = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model_name="meta-llama/llama-4-scout-17b-16e-instruct"
        )
        
        # Initialize tools with custom wrapper
        self.search_wrapper = CustomSearchWrapper()
        
        # Create the analysis chain
        self.analysis_chain = self._create_analysis_chain()
    
    def _create_analysis_chain(self):
        """Create the analysis chain"""
        
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
            ("human", "Analyze these papers: {papers}")
        ])
        
        chain = analysis_prompt | self.groq_client | JsonOutputParser()
        return chain
    
    async def search_papers(self, query: str, max_papers: int = 10) -> List[Dict]:
        """
        Search for papers related to the query
        """
        papers = []
        
        try:
            # Search using ArXiv
            try:
                arxiv_results = self.search_wrapper.search_arxiv(query)
                if arxiv_results:
                    papers.extend(self._parse_arxiv_results(arxiv_results))
                    print(f"Found {len(papers)} papers from ArXiv")
            except Exception as e:
                print(f"ArXiv search error: {e}")
            
            # Search using DuckDuckGo
            try:
                web_results = self.search_wrapper.search_web(f"research papers {query}")
                if web_results:
                    web_papers = self._parse_web_results(web_results)
                    papers.extend(web_papers)
                    print(f"Found {len(web_papers)} papers from web search")
            except Exception as e:
                print(f"Web search error: {e}")
            
            # Remove duplicates and limit results
            unique_papers = self._deduplicate_papers(papers)
            return unique_papers[:max_papers]
            
        except Exception as e:
            print(f"Error searching papers: {e}")
            return []
    
    def _parse_arxiv_results(self, results: str) -> List[Dict]:
        """Parse ArXiv results"""
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
            
            if current_paper:
                papers.append(current_paper)
                
        except Exception as e:
            print(f"Error parsing ArXiv results: {e}")
        
        return papers
    
    def _parse_web_results(self, results: str) -> List[Dict]:
        """Parse web search results"""
        papers = []
        try:
            lines = results.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in ['paper', 'research', 'study', 'journal']):
                    papers.append({
                        'title': line.strip(),
                        'source': 'Web Search',
                        'abstract': 'Abstract not available from web search'
                    })
        except Exception as e:
            print(f"Error parsing web results: {e}")
        
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
        """Format papers data for analysis"""
        formatted_papers = []
        
        for i, paper in enumerate(papers, 1):
            paper_text = f"Paper {i}:\n"
            paper_text += f"Title: {str(paper.get('title', 'N/A'))}\n"
            paper_text += f"Authors: {str(paper.get('authors', 'N/A'))}\n"
            paper_text += f"Year: {str(paper.get('year', 'N/A'))}\n"
            paper_text += f"Abstract: {str(paper.get('abstract', 'N/A'))}\n"
            paper_text += f"Source: {str(paper.get('source', 'N/A'))}\n"
            paper_text += "-" * 50 + "\n"
            
            formatted_papers.append(paper_text)
        
        return "\n".join(formatted_papers)
    
    async def analyze_papers(self, papers: List[Dict]) -> Dict[str, Any]:
        """
        Analyze the papers using the AI chain
        """
        try:
            if not papers:
                return {
                    "papers_analyzed": 0,
                    "analysis_summary": "No papers found to analyze",
                    "papers": [],
                    "error": "No papers available for analysis"
                }
            
            # Format papers for analysis
            papers_text = self._format_papers_for_analysis(papers)
            
            # Run the analysis with error handling
            try:
                result = await self.analysis_chain.ainvoke({
                    "papers": papers_text
                })
            except Exception as chain_error:
                print(f"Analysis chain failed: {chain_error}")
                # Fallback to basic analysis
                return self._basic_analysis(papers)
            
            # Add metadata
            result.update({
                "papers_found": len(papers),
                "analysis_date": datetime.now().isoformat(),
                "query": "test query"  # This will be set by the calling function
            })
            
            return result
            
        except Exception as e:
            print(f"Error analyzing papers: {e}")
            return self._basic_analysis(papers)
    
    def _basic_analysis(self, papers: List[Dict]) -> Dict[str, Any]:
        """Basic analysis when AI models fail"""
        return {
            "papers_analyzed": len(papers),
            "analysis_summary": "Basic analysis completed - AI analysis failed",
            "papers": papers,
            "extraction_date": datetime.now().isoformat(),
            "error": "AI analysis failed, showing basic results"
        }
    
    async def run_full_analysis(self, query: str, max_papers: int = 10) -> Dict[str, Any]:
        """
        Run complete analysis: search papers, analyze, and prepare for export
        """
        print(f"🔍 Searching for papers about: {query}")
        
        # Step 1: Search for papers
        papers = await self.search_papers(query, max_papers)
        print(f"📄 Found {len(papers)} papers")
        
        if not papers:
            return {
                "error": "No papers found for the given query",
                "papers": [],
                "analysis_summary": "No papers found to analyze"
            }
        
        # Step 2: Analyze papers
        print("🤖 Analyzing papers with AI...")
        analysis_result = await self.analyze_papers(papers)
        
        # Step 3: Add metadata
        analysis_result.update({
            "query": query,
            "papers_found": len(papers),
            "analysis_date": datetime.now().isoformat(),
            "export_ready": True
        })
        
        print("✅ Analysis completed!")
        return analysis_result
    
    def export_to_csv(self, analysis_result: Dict[str, Any], filename: str = None) -> str:
        """
        Export analysis results to CSV file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"paper_analysis_{timestamp}.csv"
        
        filepath = os.path.join(settings.EXPORTS_DIR, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare data for CSV export
        csv_data = []
        
        # Add paper details
        for paper in analysis_result.get('papers', []):
            row = {
                'Title': str(paper.get('title', '')),
                'Authors': str(paper.get('authors', '')),
                'Year': str(paper.get('year', 'N/A')),
                'Abstract': str(paper.get('abstract', 'N/A')),
                'Source': str(paper.get('source', 'N/A')),
                'Main Findings': str(paper.get('main_findings', 'N/A')),
                'Methodology': str(paper.get('methodology', 'N/A')),
                'Research Gaps': str(paper.get('research_gaps', 'N/A')),
                'Future Research': str(paper.get('future_research', 'N/A'))
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