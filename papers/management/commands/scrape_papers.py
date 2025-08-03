"""
Django management command to scrape papers from academic sources.
"""

import logging
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from django.utils import timezone

from papers.models import ScrapingTask
from papers.services.arxiv_scraper import ArXivScraper
from papers.services.semantic_scholar_scraper import SemanticScholarScraper

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Scrape papers from academic sources (ArXiv, Semantic Scholar, etc.)'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--source',
            type=str,
            choices=['arxiv', 'semantic_scholar', 'all'],
            default='all',
            help='Source to scrape from'
        )
        parser.add_argument(
            '--query',
            type=str,
            default='machine learning',
            help='Search query'
        )
        parser.add_argument(
            '--max-results',
            type=int,
            default=50,
            help='Maximum number of results to retrieve'
        )
        parser.add_argument(
            '--category',
            type=str,
            help='Category filter (for ArXiv)'
        )
        parser.add_argument(
            '--year',
            type=int,
            help='Year filter'
        )
        parser.add_argument(
            '--user-id',
            type=int,
            help='User ID to associate with the scraping task'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be scraped without actually doing it'
        )
    
    def handle(self, *args, **options):
        source = options['source']
        query = options['query']
        max_results = options['max_results']
        category = options.get('category')
        year = options.get('year')
        user_id = options.get('user_id')
        dry_run = options['dry_run']
        
        # Prepare filters
        filters = {}
        if category:
            filters['category'] = category
        if year:
            filters['year'] = year
        
        self.stdout.write(
            self.style.SUCCESS(
                f'Starting paper scraping from {source} with query: "{query}"'
            )
        )
        
        if dry_run:
            self.stdout.write(
                self.style.WARNING('DRY RUN MODE - No papers will be scraped')
            )
        
        try:
            if source == 'arxiv' or source == 'all':
                self._scrape_arxiv(query, max_results, filters, user_id, dry_run)
            
            if source == 'semantic_scholar' or source == 'all':
                self._scrape_semantic_scholar(query, max_results, filters, user_id, dry_run)
            
            self.stdout.write(
                self.style.SUCCESS('Paper scraping completed successfully')
            )
            
        except Exception as e:
            raise CommandError(f'Scraping failed: {str(e)}')
    
    def _scrape_arxiv(self, query, max_results, filters, user_id, dry_run):
        """Scrape papers from ArXiv."""
        self.stdout.write('Scraping from ArXiv...')
        
        if dry_run:
            self.stdout.write(f'Would scrape {max_results} papers from ArXiv')
            return
        
        try:
            scraper = ArXivScraper()
            papers = scraper.scrape_papers(query, max_results, filters)
            
            self.stdout.write(
                self.style.SUCCESS(f'Successfully scraped {len(papers)} papers from ArXiv')
            )
            
            # Log details
            for paper in papers[:5]:  # Show first 5 papers
                self.stdout.write(f'  - {paper.title}')
            
            if len(papers) > 5:
                self.stdout.write(f'  ... and {len(papers) - 5} more papers')
                
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Failed to scrape from ArXiv: {str(e)}')
            )
            raise
    
    def _scrape_semantic_scholar(self, query, max_results, filters, user_id, dry_run):
        """Scrape papers from Semantic Scholar."""
        self.stdout.write('Scraping from Semantic Scholar...')
        
        if dry_run:
            self.stdout.write(f'Would scrape {max_results} papers from Semantic Scholar')
            return
        
        try:
            scraper = SemanticScholarScraper()
            papers = scraper.scrape_papers(query, max_results, filters)
            
            self.stdout.write(
                self.style.SUCCESS(f'Successfully scraped {len(papers)} papers from Semantic Scholar')
            )
            
            # Log details
            for paper in papers[:5]:  # Show first 5 papers
                self.stdout.write(f'  - {paper.title}')
            
            if len(papers) > 5:
                self.stdout.write(f'  ... and {len(papers) - 5} more papers')
                
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Failed to scrape from Semantic Scholar: {str(e)}')
            )
            raise 