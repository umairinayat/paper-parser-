"""
Base scraper class for academic paper sources.
Provides common functionality for rate limiting, error handling, and data processing.
"""

import logging
import time
import hashlib
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import requests
from django.conf import settings
from django.utils import timezone

from papers.models import Paper, Author, Journal, PaperAuthor, ScrapingTask

logger = logging.getLogger(__name__)


class BaseScraper(ABC):
    """Base class for all academic paper scrapers."""
    
    def __init__(self, task: Optional[ScrapingTask] = None):
        self.task = task
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AcademicPaperScraper/1.0 (research-tool)'
        })
        self.rate_limit_delay = 1.0  # Default 1 second between requests
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Implement rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, params: Dict[str, Any] = None, 
                     headers: Dict[str, str] = None) -> requests.Response:
        """Make HTTP request with rate limiting and error handling."""
        self._rate_limit()
        
        try:
            response = self.session.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {str(e)}")
            raise
    
    def _generate_file_hash(self, content: str) -> str:
        """Generate SHA-256 hash for content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _find_or_create_author(self, author_data: Dict[str, Any]) -> Author:
        """Find existing author or create new one."""
        name = author_data.get('name', '').strip()
        if not name:
            return None
            
        # Try to find by name first
        author, created = Author.objects.get_or_create(
            name=name,
            defaults={
                'email': author_data.get('email', ''),
                'affiliation': author_data.get('affiliation', ''),
                'orcid_id': author_data.get('orcid_id', ''),
                'h_index': author_data.get('h_index'),
                'citations_count': author_data.get('citations_count', 0),
                'papers_count': author_data.get('papers_count', 0),
            }
        )
        
        if not created:
            # Update existing author with new information if available
            if author_data.get('h_index') and (not author.h_index or author.h_index < author_data['h_index']):
                author.h_index = author_data['h_index']
            if author_data.get('citations_count', 0) > author.citations_count:
                author.citations_count = author_data['citations_count']
            if author_data.get('papers_count', 0) > author.papers_count:
                author.papers_count = author_data['papers_count']
            author.save()
        
        return author
    
    def _find_or_create_journal(self, journal_data: Dict[str, Any]) -> Optional[Journal]:
        """Find existing journal or create new one."""
        name = journal_data.get('name', '').strip()
        if not name:
            return None
            
        journal, created = Journal.objects.get_or_create(
            name=name,
            defaults={
                'issn': journal_data.get('issn', ''),
                'publisher': journal_data.get('publisher', ''),
                'impact_factor': journal_data.get('impact_factor'),
                'h_index': journal_data.get('h_index'),
                'quartile': journal_data.get('quartile', ''),
                'subject_area': journal_data.get('subject_area', ''),
            }
        )
        
        if not created:
            # Update existing journal with new information if available
            if journal_data.get('impact_factor') and (not journal.impact_factor or journal.impact_factor < journal_data['impact_factor']):
                journal.impact_factor = journal_data['impact_factor']
            if journal_data.get('h_index') and (not journal.h_index or journal.h_index < journal_data['h_index']):
                journal.h_index = journal_data['h_index']
            journal.save()
        
        return journal
    
    def _create_paper(self, paper_data: Dict[str, Any]) -> Paper:
        """Create a new paper from scraped data."""
        # Check for existing paper by DOI, ArXiv ID, or PMID
        existing_paper = None
        
        # Check by DOI (if not empty/null)
        if paper_data.get('doi') and paper_data['doi'].strip():
            existing_paper = Paper.objects.filter(doi=paper_data['doi'].strip()).first()
        
        # Check by ArXiv ID (if not empty/null)
        if not existing_paper and paper_data.get('arxiv_id') and paper_data['arxiv_id'].strip():
            existing_paper = Paper.objects.filter(arxiv_id=paper_data['arxiv_id'].strip()).first()
        
        # Check by PMID (if not empty/null)
        if not existing_paper and paper_data.get('pmid') and paper_data['pmid'].strip():
            existing_paper = Paper.objects.filter(pmid=paper_data['pmid'].strip()).first()
        
        # Check by title if no other identifiers match
        if not existing_paper and paper_data.get('title'):
            title = paper_data['title'].strip()
            if title:
                existing_paper = Paper.objects.filter(title__iexact=title).first()
        
        if existing_paper:
            logger.info(f"Paper already exists: {existing_paper.title}")
            return existing_paper
        
        # Convert authors list to string for the old schema
        authors_str = ''
        if paper_data.get('authors'):
            author_names = [author.get('name', '') for author in paper_data['authors'] if author.get('name')]
            authors_str = ', '.join(author_names)
        
        # Convert keywords and subject_areas to strings safely
        keywords = paper_data.get('keywords', []) or []
        subject_areas = paper_data.get('subject_areas', []) or []
        keywords_str = str(keywords) if keywords else ''
        subject_areas_str = str(subject_areas) if subject_areas else ''
        
        # Create new paper with old schema compatibility
        # Handle empty strings that should be None for unique constraints
        doi_value = paper_data.get('doi', '') or ''
        arxiv_id_value = paper_data.get('arxiv_id', '') or ''
        pmid_value = paper_data.get('pmid', '') or ''
        
        # Convert empty strings to None for unique fields to avoid constraint issues
        doi_final = doi_value.strip() if doi_value and doi_value.strip() else None
        arxiv_id_final = arxiv_id_value.strip() if arxiv_id_value and arxiv_id_value.strip() else None
        pmid_final = pmid_value.strip() if pmid_value and pmid_value.strip() else None
        
        paper = Paper.objects.create(
            title=paper_data.get('title', '') or '',
            abstract=paper_data.get('abstract', '') or '',
            authors=authors_str,  # Use as string for old schema
            doi=doi_final,
            arxiv_id=arxiv_id_final,
            pmid=pmid_final,
            publication_date=paper_data.get('publication_date'),
            volume=paper_data.get('volume', '') or '',
            issue=paper_data.get('issue', '') or '',
            pages=paper_data.get('pages', '') or '',
            keywords=keywords_str,
            subject_areas=subject_areas_str,
            full_text=paper_data.get('full_text', '') or '',
            citation_count=paper_data.get('citation_count', 0) or 0,
            download_count=paper_data.get('download_count', 0) or 0,
            view_count=paper_data.get('view_count', 0) or 0,
            source=self.source_name,
            source_url=paper_data.get('source_url', '') or '',
            scraped_at=timezone.now(),
            user=self.task.user if self.task else None,
        )
        
        # Set journal if available
        if paper_data.get('journal'):
            journal = self._find_or_create_journal(paper_data['journal'])
            if journal:
                paper.journal = journal
                paper.save()
        
        # Set first and corresponding authors
        if paper_data.get('authors'):
            paper.first_author = paper_data['authors'][0].get('name', '') if paper_data['authors'] else ''
            corresponding_author = next((a for a in paper_data['authors'] if a.get('is_corresponding')), None)
            if corresponding_author:
                paper.corresponding_author = corresponding_author.get('name', '')
                paper.save()
        
        return paper
    
    def _update_task_progress(self, papers_found: int = 0, papers_processed: int = 0, 
                             papers_skipped: int = 0, papers_failed: int = 0):
        """Update scraping task progress."""
        if self.task:
            self.task.papers_found += papers_found
            self.task.papers_processed += papers_processed
            self.task.papers_skipped += papers_skipped
            self.task.papers_failed += papers_failed
            self.task.save()
    
    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the name of the source."""
        pass
    
    @abstractmethod
    def search_papers(self, query: str, max_results: int = 100, 
                     filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for papers using the source's API."""
        pass
    
    @abstractmethod
    def get_paper_details(self, paper_id: str) -> Dict[str, Any]:
        """Get detailed information for a specific paper."""
        pass
    
    def scrape_papers(self, query: str, max_results: int = 100, 
                     filters: Dict[str, Any] = None) -> List[Paper]:
        """Main method to scrape papers from the source."""
        if self.task:
            self.task.status = 'running'
            self.task.started_at = timezone.now()
            self.task.save()
        
        try:
            # Search for papers
            papers_data = self.search_papers(query, max_results, filters)
            
            if self.task:
                self.task.papers_found = len(papers_data)
                self.task.save()
            
            papers = []
            for paper_data in papers_data:
                try:
                    # Get detailed information
                    detailed_data = self.get_paper_details(paper_data.get('id', ''))
                    if detailed_data:
                        paper_data.update(detailed_data)
                    
                    # Create paper
                    paper = self._create_paper(paper_data)
                    papers.append(paper)
                    
                    self._update_task_progress(papers_processed=1)
                    
                except Exception as e:
                    logger.error(f"Failed to process paper {paper_data.get('title', 'Unknown')}: {str(e)}")
                    self._update_task_progress(papers_failed=1)
            
            if self.task:
                self.task.status = 'completed'
                self.task.completed_at = timezone.now()
                self.task.save()
            
            return papers
            
        except Exception as e:
            logger.error(f"Scraping failed for {self.source_name}: {str(e)}")
            if self.task:
                self.task.status = 'failed'
                self.task.error_message = str(e)
                self.task.completed_at = timezone.now()
                self.task.save()
            raise 