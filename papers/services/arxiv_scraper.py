"""
ArXiv scraper for retrieving academic papers from ArXiv API.
"""

import logging
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from datetime import datetime
import re

from .base_scraper import BaseScraper
from papers.models import ScrapingTask

logger = logging.getLogger(__name__)


class ArXivScraper(BaseScraper):
    """Scraper for ArXiv papers using the ArXiv API."""
    
    def __init__(self, task: Optional[ScrapingTask] = None):
        super().__init__(task)
        self.base_url = "http://export.arxiv.org/api/query"
        self.rate_limit_delay = 3.0  # ArXiv recommends 3 seconds between requests
    
    @property
    def source_name(self) -> str:
        return "arxiv"
    
    def search_papers(self, query: str, max_results: int = 100, 
                     filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for papers on ArXiv."""
        # Improve search query for better relevance
        search_query = f'all:"{query}" OR ti:"{query}" OR abs:"{query}"'
        
        params = {
            'search_query': search_query,
            'start': 0,
            'max_results': min(max_results, 100),  # ArXiv API limit
            'sortBy': 'relevance',  # Change to relevance for better results
            'sortOrder': 'descending'
        }
        
        logger.info(f"ArXiv search query: {search_query}")
        
        # Add category filter if provided
        if filters and filters.get('category'):
            category = filters['category']
            if not query.endswith(f"AND cat:{category}"):
                params['search_query'] = f"{query} AND cat:{category}"
        
        try:
            response = self._make_request(self.base_url, params=params)
            return self._parse_search_results(response.text)
        except Exception as e:
            logger.error(f"ArXiv search failed: {str(e)}")
            raise
    
    def get_paper_details(self, arxiv_id: str) -> Dict[str, Any]:
        """Get detailed information for a specific ArXiv paper."""
        if not arxiv_id:
            return {}
        
        # Clean ArXiv ID
        arxiv_id = arxiv_id.replace('arxiv:', '').replace('http://arxiv.org/abs/', '')
        
        params = {
            'id_list': arxiv_id,
            'start': 0,
            'max_results': 1
        }
        
        try:
            response = self._make_request(self.base_url, params=params)
            results = self._parse_search_results(response.text)
            return results[0] if results else {}
        except Exception as e:
            logger.error(f"Failed to get ArXiv paper details for {arxiv_id}: {str(e)}")
            return {}
    
    def _parse_search_results(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parse ArXiv XML response into structured data."""
        try:
            root = ET.fromstring(xml_content)
            
            # Check for errors
            error = root.find('.//{http://www.w3.org/2005/Atom}error')
            if error is not None:
                raise Exception(f"ArXiv API error: {error.text}")
            
            papers = []
            for entry in root.findall('.//{http://www.w3.org/2005/Atom}entry'):
                paper_data = self._parse_entry(entry)
                if paper_data:
                    papers.append(paper_data)
            
            return papers
            
        except ET.ParseError as e:
            logger.error(f"Failed to parse ArXiv XML: {str(e)}")
            return []
    
    def _parse_entry(self, entry) -> Dict[str, Any]:
        """Parse a single ArXiv entry."""
        try:
            # Extract basic information
            title = self._get_text(entry, './/{http://www.w3.org/2005/Atom}title')
            summary = self._get_text(entry, './/{http://www.w3.org/2005/Atom}summary')
            published = self._get_text(entry, './/{http://www.w3.org/2005/Atom}published')
            updated = self._get_text(entry, './/{http://www.w3.org/2005/Atom}updated')
            
            # Extract ArXiv ID from id field
            id_elem = entry.find('.//{http://www.w3.org/2005/Atom}id')
            arxiv_id = None
            if id_elem is not None and id_elem.text:
                # Extract ID from URL like http://arxiv.org/abs/2101.12345
                match = re.search(r'/abs/([^/]+)$', id_elem.text)
                if match:
                    arxiv_id = match.group(1)
            
            # Extract authors
            authors = []
            for author_elem in entry.findall('.//{http://www.w3.org/2005/Atom}author'):
                name = self._get_text(author_elem, './/{http://www.w3.org/2005/Atom}name')
                if name:
                    authors.append({
                        'name': name.strip(),
                        'email': '',
                        'affiliation': '',
                        'orcid_id': '',
                        'h_index': None,
                        'citations_count': 0,
                        'papers_count': 0,
                        'is_corresponding': False
                    })
            
            # Extract categories
            categories = []
            for category_elem in entry.findall('.//{http://arxiv.org/schemas/atom}category'):
                term = category_elem.get('term')
                if term:
                    categories.append(term)
            
            # Extract DOI if available
            doi = None
            for link in entry.findall('.//{http://www.w3.org/2005/Atom}link'):
                title_attr = link.get('title')
                if title_attr == 'doi':
                    doi = link.get('href', '').replace('http://dx.doi.org/', '')
                    break
            
            # Parse dates
            publication_date = None
            if published:
                try:
                    publication_date = datetime.fromisoformat(published.replace('Z', '+00:00')).date()
                except ValueError:
                    pass
            
            # Create paper data
            paper_data = {
                'id': arxiv_id,
                'title': title.strip() if title else '',
                'abstract': summary.strip() if summary else '',
                'arxiv_id': arxiv_id,
                'doi': doi,
                'publication_date': publication_date,
                'authors': authors,
                'keywords': categories,
                'subject_areas': categories,
                'source_url': f"http://arxiv.org/abs/{arxiv_id}" if arxiv_id else '',
                'citation_count': 0,  # ArXiv doesn't provide citation counts
                'download_count': 0,
                'view_count': 0,
            }
            
            return paper_data
            
        except Exception as e:
            logger.error(f"Failed to parse ArXiv entry: {str(e)}")
            return {}
    
    def _get_text(self, element, xpath: str) -> str:
        """Safely extract text from XML element."""
        try:
            found = element.find(xpath)
            return found.text if found is not None else ''
        except Exception:
            return ''
    
    def get_recent_papers(self, category: str = None, max_results: int = 100) -> List[Dict[str, Any]]:
        """Get recent papers from ArXiv, optionally filtered by category."""
        query = "all:recent"
        if category:
            query = f"cat:{category}"
        
        return self.search_papers(query, max_results)
    
    def get_papers_by_author(self, author_name: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Get papers by a specific author."""
        query = f"au:\"{author_name}\""
        return self.search_papers(query, max_results)
    
    def get_papers_by_date_range(self, start_date: str, end_date: str, 
                                category: str = None, max_results: int = 100) -> List[Dict[str, Any]]:
        """Get papers within a date range."""
        query = f"submittedDate:[{start_date}0000 TO {end_date}0000]"
        if category:
            query += f" AND cat:{category}"
        
        return self.search_papers(query, max_results) 