"""
PubMed scraper for retrieving biomedical papers from PubMed API.
"""

import logging
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from datetime import datetime
import re

from .base_scraper import BaseScraper
from papers.models import ScrapingTask

logger = logging.getLogger(__name__)


class PubMedScraper(BaseScraper):
    """Scraper for PubMed papers using the NCBI E-utilities API."""
    
    def __init__(self, task: Optional[ScrapingTask] = None):
        super().__init__(task)
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.rate_limit_delay = 0.5  # NCBI allows 3 requests per second
    
    @property
    def source_name(self) -> str:
        return "pubmed"
    
    def search_papers(self, query: str, max_results: int = 100, 
                     filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for papers on PubMed."""
        # First, search for PMIDs
        search_url = f"{self.base_url}/esearch.fcgi"
        search_params = {
            'db': 'pubmed',
            'term': query,
            'retmax': min(max_results, 100),
            'retmode': 'xml',
            'sort': 'relevance'
        }
        
        logger.info(f"PubMed search query: {query}")
        
        # Add date filters if provided
        if filters and filters.get('date_from'):
            search_params['mindate'] = filters['date_from']
        if filters and filters.get('date_to'):
            search_params['maxdate'] = filters['date_to']
        
        try:
            response = self._make_request(search_url, params=search_params)
            pmids = self._extract_pmids(response.text)
            
            if not pmids:
                return []
            
            # Fetch detailed information for the PMIDs
            return self._fetch_paper_details(pmids)
            
        except Exception as e:
            logger.error(f"PubMed search failed: {str(e)}")
            raise
    
    def get_paper_details(self, pmid: str) -> Dict[str, Any]:
        """Get detailed information for a specific PubMed paper."""
        if not pmid:
            return {}
        
        details = self._fetch_paper_details([pmid])
        return details[0] if details else {}
    
    def _extract_pmids(self, xml_content: str) -> List[str]:
        """Extract PMIDs from search response."""
        try:
            root = ET.fromstring(xml_content)
            pmids = []
            for id_elem in root.findall('.//Id'):
                if id_elem.text:
                    pmids.append(id_elem.text)
            return pmids
        except ET.ParseError as e:
            logger.error(f"Failed to parse PubMed search XML: {str(e)}")
            return []
    
    def _fetch_paper_details(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """Fetch detailed information for multiple PMIDs."""
        if not pmids:
            return []
        
        # Use efetch to get detailed information
        fetch_url = f"{self.base_url}/efetch.fcgi"
        fetch_params = {
            'db': 'pubmed',
            'id': ','.join(pmids),
            'retmode': 'xml',
            'rettype': 'abstract'
        }
        
        try:
            response = self._make_request(fetch_url, params=fetch_params)
            return self._parse_paper_details(response.text)
        except Exception as e:
            logger.error(f"Failed to fetch PubMed paper details: {str(e)}")
            return []
    
    def _parse_paper_details(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parse detailed paper information from efetch response."""
        papers = []
        try:
            root = ET.fromstring(xml_content)
            
            for article in root.findall('.//PubmedArticle'):
                paper_data = self._parse_article(article)
                if paper_data:
                    papers.append(paper_data)
            
            return papers
            
        except ET.ParseError as e:
            logger.error(f"Failed to parse PubMed details XML: {str(e)}")
            return []
    
    def _parse_article(self, article) -> Dict[str, Any]:
        """Parse a single PubMed article."""
        try:
            # Extract basic information
            medline_citation = article.find('.//MedlineCitation')
            if medline_citation is None:
                return {}
            
            # Get PMID
            pmid_elem = medline_citation.find('.//PMID')
            pmid = pmid_elem.text if pmid_elem is not None else ''
            
            # Get article details
            article_elem = medline_citation.find('.//Article')
            if article_elem is None:
                return {}
            
            # Title
            title_elem = article_elem.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else ''
            
            # Abstract
            abstract_parts = []
            for abstract_elem in article_elem.findall('.//AbstractText'):
                label = abstract_elem.get('Label', '')
                text = abstract_elem.text or ''
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
            abstract = ' '.join(abstract_parts)
            
            # Authors
            authors = []
            for author_elem in article_elem.findall('.//Author'):
                name_parts = []
                
                last_name = author_elem.find('.//LastName')
                if last_name is not None and last_name.text:
                    name_parts.append(last_name.text)
                
                fore_name = author_elem.find('.//ForeName')
                if fore_name is not None and fore_name.text:
                    name_parts.append(fore_name.text)
                elif author_elem.find('.//Initials') is not None:
                    initials = author_elem.find('.//Initials')
                    if initials.text:
                        name_parts.append(initials.text)
                
                if name_parts:
                    full_name = f"{' '.join(name_parts[1:])} {name_parts[0]}" if len(name_parts) > 1 else name_parts[0]
                    
                    # Get affiliation if available
                    affiliation = ''
                    affiliation_elem = author_elem.find('.//Affiliation')
                    if affiliation_elem is not None and affiliation_elem.text:
                        affiliation = affiliation_elem.text
                    
                    authors.append({
                        'name': full_name.strip(),
                        'email': '',
                        'affiliation': affiliation,
                        'orcid_id': '',
                        'h_index': None,
                        'citations_count': 0,
                        'papers_count': 0,
                        'is_corresponding': False
                    })
            
            # Journal information
            journal_elem = article_elem.find('.//Journal')
            journal_data = None
            if journal_elem is not None:
                journal_title = journal_elem.find('.//Title')
                issn_elem = journal_elem.find('.//ISSN')
                
                journal_data = {
                    'name': journal_title.text if journal_title is not None else '',
                    'issn': issn_elem.text if issn_elem is not None else '',
                    'publisher': '',
                    'impact_factor': None,
                    'h_index': None,
                    'quartile': '',
                    'subject_area': ''
                }
            
            # Publication date
            publication_date = None
            pub_date_elem = article_elem.find('.//PubDate')
            if pub_date_elem is not None:
                year_elem = pub_date_elem.find('.//Year')
                month_elem = pub_date_elem.find('.//Month')
                day_elem = pub_date_elem.find('.//Day')
                
                if year_elem is not None and year_elem.text:
                    year = int(year_elem.text)
                    month = 1
                    day = 1
                    
                    if month_elem is not None and month_elem.text:
                        try:
                            month = int(month_elem.text)
                        except ValueError:
                            # Handle month names
                            month_names = {
                                'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
                                'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
                                'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                            }
                            month = month_names.get(month_elem.text, 1)
                    
                    if day_elem is not None and day_elem.text:
                        try:
                            day = int(day_elem.text)
                        except ValueError:
                            day = 1
                    
                    try:
                        publication_date = datetime(year, month, day).date()
                    except ValueError:
                        publication_date = datetime(year, 1, 1).date()
            
            # MeSH terms (keywords)
            mesh_terms = []
            for mesh_elem in medline_citation.findall('.//MeshHeading/DescriptorName'):
                if mesh_elem.text:
                    mesh_terms.append(mesh_elem.text)
            
            # DOI
            doi = ''
            for elocation_elem in article_elem.findall('.//ELocationID'):
                if elocation_elem.get('EIdType') == 'doi' and elocation_elem.text:
                    doi = elocation_elem.text
                    break
            
            # Create paper data
            paper_data = {
                'id': pmid,
                'title': title,
                'abstract': abstract,
                'pmid': pmid,
                'doi': doi,
                'publication_date': publication_date,
                'journal': journal_data,
                'authors': authors,
                'keywords': mesh_terms,
                'subject_areas': mesh_terms,
                'source_url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                'citation_count': 0,  # PubMed doesn't provide citation counts
                'download_count': 0,
                'view_count': 0,
            }
            
            return paper_data
            
        except Exception as e:
            logger.error(f"Failed to parse PubMed article: {str(e)}")
            return {}
    
    def get_papers_by_author(self, author_name: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Get papers by a specific author."""
        query = f'"{author_name}"[Author]'
        return self.search_papers(query, max_results)
    
    def get_papers_by_journal(self, journal_name: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Get papers from a specific journal."""
        query = f'"{journal_name}"[Journal]'
        return self.search_papers(query, max_results)
    
    def get_recent_papers(self, days: int = 30, max_results: int = 100) -> List[Dict[str, Any]]:
        """Get recent papers from the last N days."""
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        query = f"({start_date.strftime('%Y/%m/%d')}[PDAT] : {end_date.strftime('%Y/%m/%d')}[PDAT])"
        return self.search_papers(query, max_results)