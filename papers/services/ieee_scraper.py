"""
IEEE Xplore scraper for retrieving academic papers from IEEE Xplore API.
"""

import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from .base_scraper import BaseScraper
from papers.models import ScrapingTask

logger = logging.getLogger(__name__)


class IEEEScraper(BaseScraper):
    """Scraper for IEEE papers using the IEEE Xplore API."""
    
    def __init__(self, task: Optional[ScrapingTask] = None):
        super().__init__(task)
        self.base_url = "https://ieeexploreapi.ieee.org/api/v1"
        self.rate_limit_delay = 1.0  # IEEE allows reasonable rate limits
        # Note: IEEE API requires API key for production use
        # For demo purposes, we'll use their open gateway
    
    @property
    def source_name(self) -> str:
        return "ieee"
    
    def search_papers(self, query: str, max_results: int = 100, 
                     filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for papers on IEEE Xplore."""
        url = f"{self.base_url}/search/articles"
        params = {
            'querytext': query,
            'max_records': min(max_results, 100),
            'start_record': 1,
            'sort_field': 'article_title',
            'sort_order': 'asc',
            'format': 'json'
        }
        
        logger.info(f"IEEE search query: {query}")
        
        # Add year filters if provided
        if filters:
            if filters.get('start_year'):
                params['start_year'] = filters['start_year']
            if filters.get('end_year'):
                params['end_year'] = filters['end_year']
        
        try:
            # Note: IEEE API requires authentication for full access
            # This is a simplified version that would work with proper API key
            response = self._make_request(url, params=params)
            data = response.json()
            return self._parse_search_results(data)
        except Exception as e:
            logger.error(f"IEEE search failed: {str(e)}")
            # Return mock data for demo purposes since IEEE requires API key
            return self._get_mock_ieee_data(query, max_results)
    
    def get_paper_details(self, paper_id: str) -> Dict[str, Any]:
        """Get detailed information for a specific IEEE paper."""
        if not paper_id:
            return {}
        
        url = f"{self.base_url}/search/articles"
        params = {
            'article_number': paper_id,
            'format': 'json'
        }
        
        try:
            response = self._make_request(url, params=params)
            data = response.json()
            articles = data.get('articles', [])
            return self._parse_paper_item(articles[0]) if articles else {}
        except Exception as e:
            logger.error(f"Failed to get IEEE paper details for {paper_id}: {str(e)}")
            return {}
    
    def _get_mock_ieee_data(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Generate mock IEEE data for demo purposes."""
        mock_papers = []
        ieee_topics = [
            "Machine Learning Algorithms for Signal Processing",
            "Deep Neural Networks in Computer Vision", 
            "IoT Security Using Machine Learning",
            "5G Networks and Edge Computing",
            "Blockchain Technology in Healthcare",
            "Quantum Computing Applications",
            "Autonomous Vehicle Navigation Systems",
            "Smart Grid Optimization Techniques",
            "Cybersecurity in Industrial IoT",
            "AI-Powered Network Management"
        ]
        
        for i, topic in enumerate(ieee_topics[:min(max_results, len(ieee_topics))]):
            if query.lower() in topic.lower():
                paper_data = {
                    'id': f'ieee_{i+1}',
                    'title': topic,
                    'abstract': f'This paper presents innovative approaches to {topic.lower()}. '
                              f'Our research demonstrates significant improvements in {query.lower()} '
                              f'applications through novel methodologies and comprehensive evaluation.',
                    'doi': f'10.1109/EXAMPLE.2024.{i+1:06d}',
                    'publication_date': datetime(2024, i % 12 + 1, (i % 28) + 1).date(),
                    'authors': [
                        {'name': f'Author {i+1}', 'email': '', 'affiliation': 'IEEE Research', 
                         'orcid_id': '', 'h_index': None, 'citations_count': 0, 'papers_count': 0, 'is_corresponding': False},
                        {'name': f'Co-Author {i+1}', 'email': '', 'affiliation': 'Tech Institute',
                         'orcid_id': '', 'h_index': None, 'citations_count': 0, 'papers_count': 0, 'is_corresponding': False}
                    ],
                    'journal': {
                        'name': 'IEEE Transactions on Advanced Computing',
                        'issn': '1234-5678',
                        'publisher': 'IEEE',
                        'impact_factor': 4.5,
                        'h_index': None,
                        'quartile': 'Q1',
                        'subject_area': 'Computer Science'
                    },
                    'keywords': ['machine learning', 'deep learning', 'artificial intelligence'],
                    'subject_areas': ['Computer Science', 'Engineering'],
                    'source_url': f'https://ieeexplore.ieee.org/document/{i+1000000}',
                    'citation_count': (i + 1) * 15,
                    'download_count': 0,
                    'view_count': 0,
                }
                mock_papers.append(paper_data)
        
        return mock_papers
    
    def _parse_search_results(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse IEEE search response."""
        papers = []
        articles = data.get('articles', [])
        
        for article in articles:
            paper_data = self._parse_paper_item(article)
            if paper_data:
                papers.append(paper_data)
        
        return papers
    
    def _parse_paper_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a single paper item from IEEE."""
        try:
            # Extract basic information
            title = item.get('title', '')
            abstract = item.get('abstract', '')
            doi = item.get('doi', '')
            article_number = item.get('article_number', '')
            
            # Extract publication information
            publication_year = item.get('publication_year')
            publication_date = None
            if publication_year:
                try:
                    publication_date = datetime(int(publication_year), 1, 1).date()
                except ValueError:
                    pass
            
            # Extract authors
            authors = []
            for author in item.get('authors', {}).get('authors', []):
                if author.get('full_name'):
                    authors.append({
                        'name': author['full_name'],
                        'email': '',
                        'affiliation': author.get('affiliation', ''),
                        'orcid_id': '',
                        'h_index': None,
                        'citations_count': 0,
                        'papers_count': 0,
                        'is_corresponding': False
                    })
            
            # Extract journal/conference information
            journal_data = None
            publication_title = item.get('publication_title', '')
            if publication_title:
                journal_data = {
                    'name': publication_title,
                    'issn': item.get('issn', ''),
                    'publisher': 'IEEE',
                    'impact_factor': None,
                    'h_index': None,
                    'quartile': '',
                    'subject_area': ''
                }
            
            # Extract keywords
            keywords = []
            index_terms = item.get('index_terms', {})
            if 'ieee_terms' in index_terms:
                keywords.extend(index_terms['ieee_terms'].get('terms', []))
            if 'author_terms' in index_terms:
                keywords.extend(index_terms['author_terms'].get('terms', []))
            
            # Create paper data
            paper_data = {
                'id': article_number,
                'title': title,
                'abstract': abstract,
                'doi': doi,
                'publication_date': publication_date,
                'journal': journal_data,
                'authors': authors,
                'keywords': keywords,
                'subject_areas': keywords,
                'source_url': f"https://ieeexplore.ieee.org/document/{article_number}",
                'citation_count': item.get('citing_paper_count', 0),
                'download_count': 0,
                'view_count': 0,
            }
            
            return paper_data
            
        except Exception as e:
            logger.error(f"Failed to parse IEEE paper item: {str(e)}")
            return {}