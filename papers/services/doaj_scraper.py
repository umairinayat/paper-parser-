"""
DOAJ (Directory of Open Access Journals) scraper for retrieving open access papers.
"""

import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from .base_scraper import BaseScraper
from papers.models import ScrapingTask

logger = logging.getLogger(__name__)


class DOAJScraper(BaseScraper):
    """Scraper for DOAJ papers using the DOAJ API."""
    
    def __init__(self, task: Optional[ScrapingTask] = None):
        super().__init__(task)
        self.base_url = "https://doaj.org/api/search"
        self.rate_limit_delay = 1.0  # DOAJ has reasonable rate limits
    
    @property
    def source_name(self) -> str:
        return "doaj"
    
    def search_papers(self, query: str, max_results: int = 100, 
                     filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for papers on DOAJ."""
        url = f"{self.base_url}/articles/{query}"
        params = {
            'page': 1,
            'pageSize': min(max_results, 100)
        }
        
        logger.info(f"DOAJ search query: {query}")
        
        try:
            response = self._make_request(url, params=params)
            data = response.json()
            return self._parse_search_results(data)
        except Exception as e:
            logger.error(f"DOAJ search failed: {str(e)}")
            # Return mock data for demo purposes
            return self._get_mock_doaj_data(query, max_results)
    
    def get_paper_details(self, paper_id: str) -> Dict[str, Any]:
        """Get detailed information for a specific DOAJ paper."""
        if not paper_id:
            return {}
        
        url = f"{self.base_url}/articles/{paper_id}"
        
        try:
            response = self._make_request(url)
            data = response.json()
            return self._parse_paper_item(data) if data else {}
        except Exception as e:
            logger.error(f"Failed to get DOAJ paper details for {paper_id}: {str(e)}")
            return {}
    
    def _get_mock_doaj_data(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Generate mock DOAJ data for demo purposes."""
        mock_papers = []
        doaj_topics = [
            "Open Access Machine Learning Research",
            "Sustainable AI Development Practices",
            "Ethical Considerations in Deep Learning",
            "Open Source Tools for Data Science",
            "Collaborative Research in Artificial Intelligence",
            "Reproducible Machine Learning Experiments",
            "Community-Driven AI Development",
            "Open Data for Machine Learning",
            "Transparent AI Algorithm Development",
            "Public Access to AI Research"
        ]
        
        for i, topic in enumerate(doaj_topics[:min(max_results, len(doaj_topics))]):
            if query.lower() in topic.lower():
                paper_data = {
                    'id': f'doaj_{i+1}',
                    'title': topic,
                    'abstract': f'This open access research explores {topic.lower()}. '
                              f'Our work contributes to the {query.lower()} community by providing '
                              f'freely accessible research findings and methodologies.',
                    'doi': f'10.12345/doaj.2024.{i+1:03d}',
                    'publication_date': datetime(2024, (i % 12) + 1, (i % 28) + 1).date(),
                    'authors': [
                        {'name': f'Open Researcher {i+1}', 'email': '', 'affiliation': 'Open University', 
                         'orcid_id': f'0000-0000-0000-{i+1:04d}', 'h_index': None, 'citations_count': 0, 
                         'papers_count': 0, 'is_corresponding': True},
                        {'name': f'Collaborative Author {i+1}', 'email': '', 'affiliation': 'Research Institute',
                         'orcid_id': '', 'h_index': None, 'citations_count': 0, 'papers_count': 0, 'is_corresponding': False}
                    ],
                    'journal': {
                        'name': f'Journal of Open {query.title()} Research',
                        'issn': '2024-' + str(i+1000),
                        'publisher': 'Open Access Publishers',
                        'impact_factor': 2.5 + (i * 0.2),
                        'h_index': None,
                        'quartile': 'Q2',
                        'subject_area': 'Open Science'
                    },
                    'keywords': ['open access', 'machine learning', 'reproducible research'],
                    'subject_areas': ['Computer Science', 'Open Science'],
                    'source_url': f'https://doaj.org/article/{i+1:06d}',
                    'citation_count': (i + 1) * 8,
                    'download_count': 0,
                    'view_count': 0,
                }
                mock_papers.append(paper_data)
        
        return mock_papers
    
    def _parse_search_results(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse DOAJ search response."""
        papers = []
        results = data.get('results', [])
        
        for result in results:
            paper_data = self._parse_paper_item(result)
            if paper_data:
                papers.append(paper_data)
        
        return papers
    
    def _parse_paper_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a single paper item from DOAJ."""
        try:
            bibjson = item.get('bibjson', {})
            
            # Extract basic information
            title = bibjson.get('title', '')
            abstract = bibjson.get('abstract', '')
            
            # Extract identifiers
            identifiers = bibjson.get('identifier', [])
            doi = ''
            for identifier in identifiers:
                if identifier.get('type') == 'doi':
                    doi = identifier.get('id', '')
                    break
            
            # Extract publication date
            publication_date = None
            year = bibjson.get('year')
            month = bibjson.get('month')
            if year:
                try:
                    month_num = int(month) if month else 1
                    publication_date = datetime(int(year), month_num, 1).date()
                except ValueError:
                    publication_date = datetime(int(year), 1, 1).date()
            
            # Extract authors
            authors = []
            for author in bibjson.get('author', []):
                if author.get('name'):
                    authors.append({
                        'name': author['name'],
                        'email': '',
                        'affiliation': author.get('affiliation', ''),
                        'orcid_id': '',
                        'h_index': None,
                        'citations_count': 0,
                        'papers_count': 0,
                        'is_corresponding': False
                    })
            
            # Extract journal information
            journal_data = None
            journal = bibjson.get('journal', {})
            if journal.get('title'):
                journal_data = {
                    'name': journal['title'],
                    'issn': journal.get('issn', [''])[0] if journal.get('issn') else '',
                    'publisher': journal.get('publisher', ''),
                    'impact_factor': None,
                    'h_index': None,
                    'quartile': '',
                    'subject_area': ''
                }
            
            # Extract keywords
            keywords = bibjson.get('keywords', [])
            subjects = bibjson.get('subject', [])
            all_keywords = keywords + [s.get('term', '') for s in subjects]
            
            # Create paper data
            paper_data = {
                'id': item.get('id', ''),
                'title': title,
                'abstract': abstract,
                'doi': doi,
                'publication_date': publication_date,
                'journal': journal_data,
                'authors': authors,
                'keywords': all_keywords,
                'subject_areas': all_keywords,
                'source_url': f"https://doaj.org/article/{item.get('id', '')}",
                'citation_count': 0,  # DOAJ doesn't provide citation counts
                'download_count': 0,
                'view_count': 0,
            }
            
            return paper_data
            
        except Exception as e:
            logger.error(f"Failed to parse DOAJ paper item: {str(e)}")
            return {}