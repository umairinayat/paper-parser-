"""
Semantic Scholar scraper for retrieving academic papers from Semantic Scholar API.
"""

import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from .base_scraper import BaseScraper
from papers.models import ScrapingTask

logger = logging.getLogger(__name__)


class SemanticScholarScraper(BaseScraper):
    """Scraper for Semantic Scholar papers using the Semantic Scholar API."""
    
    def __init__(self, task: Optional[ScrapingTask] = None):
        super().__init__(task)
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.rate_limit_delay = 5.0  # Very conservative rate limiting to avoid 429 errors
    
    @property
    def source_name(self) -> str:
        return "semantic_scholar"
    
    def search_papers(self, query: str, max_results: int = 100, 
                     filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for papers on Semantic Scholar."""
        url = f"{self.base_url}/paper/search"
        params = {
            'query': query,
            'limit': min(max_results, 100),
            'fields': 'paperId,title,abstract,url,year,venue,publicationDate,publicationTypes,publicationVenue,isOpenAccess,openAccessPdf,fieldsOfStudy,publicationDate,publicationVenue,publicationTypes,isOpenAccess,openAccessPdf,fieldsOfStudy,authors,referenceCount,citationCount,influentialCitationCount'
        }
        
        # Add filters
        if filters:
            if filters.get('year'):
                params['year'] = filters['year']
            if filters.get('venue'):
                params['venue'] = filters['venue']
        
        try:
            response = self._make_request(url, params=params)
            data = response.json()
            return self._parse_search_results(data)
        except Exception as e:
            logger.error(f"Semantic Scholar search failed: {str(e)}")
            raise
    
    def get_paper_details(self, paper_id: str) -> Dict[str, Any]:
        """Get detailed information for a specific Semantic Scholar paper."""
        if not paper_id:
            return {}
        
        url = f"{self.base_url}/paper/{paper_id}"
        params = {
            'fields': 'paperId,title,abstract,url,year,venue,publicationDate,publicationTypes,publicationVenue,isOpenAccess,openAccessPdf,fieldsOfStudy,publicationDate,publicationVenue,publicationTypes,isOpenAccess,openAccessPdf,fieldsOfStudy,authors,referenceCount,citationCount,influentialCitationCount,doi,arxivId,pmid'
        }
        
        try:
            response = self._make_request(url, params=params)
            data = response.json()
            return self._parse_paper_details(data)
        except Exception as e:
            logger.error(f"Failed to get Semantic Scholar paper details for {paper_id}: {str(e)}")
            return {}
    
    def _parse_search_results(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse Semantic Scholar search response."""
        papers = []
        for item in data.get('data', []):
            paper_data = self._parse_paper_item(item)
            if paper_data:
                papers.append(paper_data)
        return papers
    
    def _parse_paper_details(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse detailed paper information."""
        return self._parse_paper_item(data)
    
    def _parse_paper_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a single paper item from Semantic Scholar."""
        try:
            # Extract basic information
            paper_id = item.get('paperId', '')
            title = item.get('title', '')
            abstract = item.get('abstract', '')
            year = item.get('year')
            doi = item.get('doi', '')
            arxiv_id = item.get('arxivId', '')
            pmid = item.get('pmid', '')
            
            # Extract publication information
            venue = item.get('venue', '')
            publication_date = None
            if item.get('publicationDate'):
                try:
                    publication_date = datetime.fromisoformat(item['publicationDate'].replace('Z', '+00:00')).date()
                except ValueError:
                    pass
            
            # Extract authors
            authors = []
            for author in item.get('authors', []):
                if author.get('name'):
                    authors.append({
                        'name': author['name'],
                        'author_id': author.get('authorId', ''),
                        'email': '',
                        'affiliation': '',
                        'orcid_id': '',
                        'h_index': None,
                        'citations_count': 0,
                        'papers_count': 0,
                        'is_corresponding': False
                    })
            
            # Extract metrics
            citation_count = item.get('citationCount', 0)
            reference_count = item.get('referenceCount', 0)
            influential_citation_count = item.get('influentialCitationCount', 0)
            
            # Extract fields of study
            fields_of_study = item.get('fieldsOfStudy', [])
            
            # Extract publication types
            publication_types = item.get('publicationTypes', [])
            
            # Create journal information
            journal_data = None
            if venue:
                journal_data = {
                    'name': venue,
                    'issn': '',
                    'publisher': '',
                    'impact_factor': None,
                    'h_index': None,
                    'quartile': '',
                    'subject_area': ', '.join(fields_of_study) if fields_of_study else ''
                }
            
            # Create paper data
            paper_data = {
                'id': paper_id,
                'title': title,
                'abstract': abstract,
                'doi': doi,
                'arxiv_id': arxiv_id,
                'pmid': pmid,
                'publication_date': publication_date,
                'journal': journal_data,
                'authors': authors,
                'keywords': fields_of_study,
                'subject_areas': fields_of_study,
                'source_url': item.get('url', ''),
                'citation_count': citation_count,
                'download_count': 0,  # Not provided by Semantic Scholar
                'view_count': 0,  # Not provided by Semantic Scholar
                'year': year,
            }
            
            return paper_data
            
        except Exception as e:
            logger.error(f"Failed to parse Semantic Scholar paper item: {str(e)}")
            return {}
    
    def get_papers_by_author(self, author_id: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Get papers by a specific author ID."""
        url = f"{self.base_url}/author/{author_id}/papers"
        params = {
            'limit': min(max_results, 100),
            'fields': 'paperId,title,abstract,url,year,venue,publicationDate,publicationTypes,publicationVenue,isOpenAccess,openAccessPdf,fieldsOfStudy,publicationDate,publicationVenue,publicationTypes,isOpenAccess,openAccessPdf,fieldsOfStudy,authors,referenceCount,citationCount,influentialCitationCount'
        }
        
        try:
            response = self._make_request(url, params=params)
            data = response.json()
            return self._parse_search_results(data)
        except Exception as e:
            logger.error(f"Failed to get papers by author {author_id}: {str(e)}")
            return []
    
    def get_paper_references(self, paper_id: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Get references for a specific paper."""
        url = f"{self.base_url}/paper/{paper_id}/references"
        params = {
            'limit': min(max_results, 100),
            'fields': 'paperId,title,abstract,url,year,venue,publicationDate,publicationTypes,publicationVenue,isOpenAccess,openAccessPdf,fieldsOfStudy,publicationDate,publicationVenue,publicationTypes,isOpenAccess,openAccessPdf,fieldsOfStudy,authors,referenceCount,citationCount,influentialCitationCount'
        }
        
        try:
            response = self._make_request(url, params=params)
            data = response.json()
            papers = []
            for item in data.get('data', []):
                if item.get('citedPaper'):
                    paper_data = self._parse_paper_item(item['citedPaper'])
                    if paper_data:
                        papers.append(paper_data)
            return papers
        except Exception as e:
            logger.error(f"Failed to get references for paper {paper_id}: {str(e)}")
            return []
    
    def get_paper_citations(self, paper_id: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Get citations for a specific paper."""
        url = f"{self.base_url}/paper/{paper_id}/citations"
        params = {
            'limit': min(max_results, 100),
            'fields': 'paperId,title,abstract,url,year,venue,publicationDate,publicationTypes,publicationVenue,isOpenAccess,openAccessPdf,fieldsOfStudy,publicationDate,publicationVenue,publicationTypes,isOpenAccess,openAccessPdf,fieldsOfStudy,authors,referenceCount,citationCount,influentialCitationCount'
        }
        
        try:
            response = self._make_request(url, params=params)
            data = response.json()
            papers = []
            for item in data.get('data', []):
                if item.get('citingPaper'):
                    paper_data = self._parse_paper_item(item['citingPaper'])
                    if paper_data:
                        papers.append(paper_data)
            return papers
        except Exception as e:
            logger.error(f"Failed to get citations for paper {paper_id}: {str(e)}")
            return [] 