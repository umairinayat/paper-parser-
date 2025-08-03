"""
Crossref scraper for retrieving academic papers from Crossref API.
"""

import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from .base_scraper import BaseScraper
from papers.models import ScrapingTask

logger = logging.getLogger(__name__)


class CrossrefScraper(BaseScraper):
    """Scraper for academic papers using the Crossref API."""
    
    def __init__(self, task: Optional[ScrapingTask] = None):
        super().__init__(task)
        self.base_url = "https://api.crossref.org"
        self.rate_limit_delay = 1.0  # Crossref allows generous rate limits
    
    @property
    def source_name(self) -> str:
        return "crossref"
    
    def search_papers(self, query: str, max_results: int = 100, 
                     filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for papers on Crossref."""
        url = f"{self.base_url}/works"
        params = {
            'query': query,
            'rows': min(max_results, 100),
            'sort': 'score',  # Use score for better relevance
            'order': 'desc'
        }
        
        logger.info(f"Crossref search query: {query}")
        
        # Add filters
        if filters:
            if filters.get('from_year'):
                params['filter'] = f"from-pub-date:{filters['from_year']}"
            if filters.get('to_year'):
                if 'filter' in params:
                    params['filter'] += f",until-pub-date:{filters['to_year']}"
                else:
                    params['filter'] = f"until-pub-date:{filters['to_year']}"
            if filters.get('publisher'):
                filter_val = f"publisher-name:{filters['publisher']}"
                if 'filter' in params:
                    params['filter'] += f",{filter_val}"
                else:
                    params['filter'] = filter_val
        
        try:
            response = self._make_request(url, params=params)
            data = response.json()
            return self._parse_search_results(data)
        except Exception as e:
            logger.error(f"Crossref search failed: {str(e)}")
            raise
    
    def get_paper_details(self, doi: str) -> Dict[str, Any]:
        """Get detailed information for a specific DOI."""
        if not doi:
            return {}
        
        # Clean DOI
        doi = doi.replace('https://doi.org/', '').replace('http://dx.doi.org/', '')
        
        url = f"{self.base_url}/works/{doi}"
        
        try:
            response = self._make_request(url)
            data = response.json()
            return self._parse_paper_item(data.get('message', {}))
        except Exception as e:
            logger.error(f"Failed to get Crossref paper details for {doi}: {str(e)}")
            return {}
    
    def _parse_search_results(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse Crossref search response."""
        papers = []
        message = data.get('message', {})
        items = message.get('items', [])
        
        for item in items:
            paper_data = self._parse_paper_item(item)
            if paper_data:
                papers.append(paper_data)
        
        return papers
    
    def _parse_paper_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a single paper item from Crossref."""
        try:
            # Extract basic information
            doi = item.get('DOI', '')
            title_list = item.get('title', [])
            title = title_list[0] if title_list else ''
            
            # Abstract (if available)
            abstract = item.get('abstract', '')
            
            # Authors
            authors = []
            for author in item.get('author', []):
                given = author.get('given', '')
                family = author.get('family', '')
                name = f"{given} {family}".strip()
                
                if name:
                    # Get ORCID if available
                    orcid_id = ''
                    if 'ORCID' in author:
                        orcid_id = author['ORCID'].replace('http://orcid.org/', '')
                    
                    # Get affiliation if available
                    affiliation = ''
                    if 'affiliation' in author and author['affiliation']:
                        affiliation_list = [aff.get('name', '') for aff in author['affiliation']]
                        affiliation = '; '.join(filter(None, affiliation_list))
                    
                    authors.append({
                        'name': name,
                        'email': '',
                        'affiliation': affiliation,
                        'orcid_id': orcid_id,
                        'h_index': None,
                        'citations_count': 0,
                        'papers_count': 0,
                        'is_corresponding': False
                    })
            
            # Journal information
            journal_data = None
            container_title = item.get('container-title', [])
            if container_title:
                issn_list = item.get('ISSN', [])
                publisher = item.get('publisher', '')
                
                journal_data = {
                    'name': container_title[0],
                    'issn': issn_list[0] if issn_list else '',
                    'publisher': publisher,
                    'impact_factor': None,
                    'h_index': None,
                    'quartile': '',
                    'subject_area': ''
                }
            
            # Publication date
            publication_date = None
            pub_date = item.get('published-print') or item.get('published-online')
            if pub_date and 'date-parts' in pub_date:
                date_parts = pub_date['date-parts'][0]
                if len(date_parts) >= 3:
                    year, month, day = date_parts[0], date_parts[1], date_parts[2]
                elif len(date_parts) >= 2:
                    year, month, day = date_parts[0], date_parts[1], 1
                elif len(date_parts) >= 1:
                    year, month, day = date_parts[0], 1, 1
                else:
                    year = month = day = None
                
                if year:
                    try:
                        publication_date = datetime(year, month, day).date()
                    except ValueError:
                        publication_date = datetime(year, 1, 1).date()
            
            # Subject areas
            subject_areas = item.get('subject', [])
            
            # Citation count (if available)
            citation_count = item.get('is-referenced-by-count', 0)
            
            # Volume, issue, pages
            volume = item.get('volume', '')
            issue = item.get('issue', '')
            page = item.get('page', '')
            
            # Create paper data
            paper_data = {
                'id': doi,
                'title': title,
                'abstract': abstract,
                'doi': doi,
                'publication_date': publication_date,
                'journal': journal_data,
                'volume': volume,
                'issue': issue,
                'pages': page,
                'authors': authors,
                'keywords': subject_areas,
                'subject_areas': subject_areas,
                'source_url': f"https://doi.org/{doi}",
                'citation_count': citation_count,
                'download_count': 0,
                'view_count': 0,
            }
            
            return paper_data
            
        except Exception as e:
            logger.error(f"Failed to parse Crossref paper item: {str(e)}")
            return {}
    
    def get_papers_by_publisher(self, publisher: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Get papers from a specific publisher."""
        return self.search_papers("*", max_results, {"publisher": publisher})
    
    def get_papers_by_year_range(self, from_year: int, to_year: int, 
                                max_results: int = 100) -> List[Dict[str, Any]]:
        """Get papers within a year range."""
        return self.search_papers("*", max_results, {
            "from_year": from_year,
            "to_year": to_year
        })
    
    def get_journal_papers(self, issn: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Get papers from a journal by ISSN."""
        url = f"{self.base_url}/works"
        params = {
            'filter': f'issn:{issn}',
            'rows': min(max_results, 100),
            'sort': 'published',
            'order': 'desc'
        }
        
        try:
            response = self._make_request(url, params=params)
            data = response.json()
            return self._parse_search_results(data)
        except Exception as e:
            logger.error(f"Failed to get journal papers for ISSN {issn}: {str(e)}")
            return []