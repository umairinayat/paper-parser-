"""
Celery tasks for paper scraping and processing operations.
"""

import logging
from typing import List, Dict, Any
from celery import shared_task
from django.conf import settings
from django.utils import timezone
from django.db import transaction

from papers.models import Paper, ScrapingTask, Author, Journal, PaperAnalysis
from papers.services.arxiv_scraper import ArXivScraper
from papers.services.semantic_scholar_scraper import SemanticScholarScraper

logger = logging.getLogger(__name__)


@shared_task(bind=True, name='papers.scrape_papers')
def scrape_papers(self, source: str, query: str, max_results: int = 100, 
                  filters: Dict[str, Any] = None, user_id: int = None):
    """
    Background task to scrape papers from academic sources.
    
    Args:
        source: Source to scrape from ('arxiv', 'semantic_scholar', 'pubmed')
        query: Search query
        max_results: Maximum number of results to retrieve
        filters: Additional filters for the search
        user_id: ID of the user who initiated the scraping
    """
    task_id = self.request.id
    
    try:
        # Create or get scraping task
        with transaction.atomic():
            scraping_task = ScrapingTask.objects.create(
                task_id=task_id,
                source=source,
                query=query,
                max_results=max_results,
                user_id=user_id,
                status='running'
            )
        
        # Initialize appropriate scraper
        if source == 'arxiv':
            scraper = ArXivScraper(scraping_task)
        elif source == 'semantic_scholar':
            scraper = SemanticScholarScraper(scraping_task)
        else:
            raise ValueError(f"Unsupported source: {source}")
        
        # Perform scraping
        papers = scraper.scrape_papers(query, max_results, filters)
        
        # Update task status
        scraping_task.status = 'completed'
        scraping_task.papers_processed = len(papers)
        scraping_task.completed_at = timezone.now()
        scraping_task.save()
        
        logger.info(f"Successfully scraped {len(papers)} papers from {source}")
        return {
            'status': 'success',
            'papers_count': len(papers),
            'task_id': task_id
        }
        
    except Exception as e:
        logger.error(f"Scraping task failed: {str(e)}")
        
        # Update task status
        if 'scraping_task' in locals():
            scraping_task.status = 'failed'
            scraping_task.error_message = str(e)
            scraping_task.completed_at = timezone.now()
            scraping_task.save()
        
        raise


@shared_task(bind=True, name='papers.scrape_recent_papers')
def scrape_recent_papers(self):
    """
    Periodic task to scrape recent papers from all sources.
    """
    sources = ['arxiv', 'semantic_scholar']
    categories = ['cs.AI', 'cs.LG', 'cs.CL', 'cs.CV', 'cs.NE']  # AI/ML categories
    
    total_papers = 0
    
    for source in sources:
        for category in categories:
            try:
                if source == 'arxiv':
                    scraper = ArXivScraper()
                    papers = scraper.get_recent_papers(category, max_results=20)
                elif source == 'semantic_scholar':
                    scraper = SemanticScholarScraper()
                    papers = scraper.search_papers(f"machine learning {category}", max_results=20)
                
                total_papers += len(papers)
                logger.info(f"Scraped {len(papers)} recent papers from {source} category {category}")
                
            except Exception as e:
                logger.error(f"Failed to scrape recent papers from {source} category {category}: {str(e)}")
    
    logger.info(f"Total recent papers scraped: {total_papers}")
    return {'total_papers': total_papers}


@shared_task(bind=True, name='papers.analyze_paper')
def analyze_paper(self, paper_id: str):
    """
    Background task to analyze a single paper using AI.
    
    Args:
        paper_id: UUID of the paper to analyze
    """
    try:
        paper = Paper.objects.get(id=paper_id)
        
        # Skip if already analyzed
        if paper.has_analysis:
            logger.info(f"Paper {paper_id} already analyzed")
            return {'status': 'already_analyzed'}
        
        # Update paper status
        paper.analysis_status = 'processing'
        paper.processed_at = timezone.now()
        paper.save()
        
        # Import analysis service here to avoid circular imports
        from analysis.services.ai_analysis_service import AIAnalysisService
        
        analysis_service = AIAnalysisService()
        analysis_result = analysis_service.analyze_paper(paper)
        
        # Create analysis record
        PaperAnalysis.objects.create(
            paper=paper,
            analysis_type='comprehensive',
            summary=analysis_result.get('summary', ''),
            key_findings=analysis_result.get('key_findings', []),
            methodology=analysis_result.get('methodology', ''),
            limitations=analysis_result.get('limitations', ''),
            future_work=analysis_result.get('future_work', ''),
            impact_assessment=analysis_result.get('impact_assessment', ''),
            methodology_type=analysis_result.get('methodology_type', ''),
            dataset_info=analysis_result.get('dataset_info', ''),
            evaluation_metrics=analysis_result.get('evaluation_metrics', []),
            model_used=analysis_result.get('model_used', ''),
            processing_time=analysis_result.get('processing_time'),
            confidence_score=analysis_result.get('confidence_score'),
        )
        
        # Update paper status
        paper.analysis_status = 'completed'
        paper.processing_time = analysis_result.get('processing_time')
        paper.save()
        
        logger.info(f"Successfully analyzed paper {paper_id}")
        return {
            'status': 'success',
            'paper_id': paper_id,
            'processing_time': analysis_result.get('processing_time')
        }
        
    except Paper.DoesNotExist:
        logger.error(f"Paper {paper_id} not found")
        raise
    except Exception as e:
        logger.error(f"Analysis failed for paper {paper_id}: {str(e)}")
        
        # Update paper status
        if 'paper' in locals():
            paper.analysis_status = 'failed'
            paper.error_message = str(e)
            paper.save()
        
        raise


@shared_task(bind=True, name='papers.analyze_pending_papers')
def analyze_pending_papers(self, batch_size: int = 10):
    """
    Periodic task to analyze papers that are pending analysis.
    
    Args:
        batch_size: Number of papers to analyze in this batch
    """
    pending_papers = Paper.objects.filter(
        analysis_status='pending'
    ).order_by('created_at')[:batch_size]
    
    analyzed_count = 0
    failed_count = 0
    
    for paper in pending_papers:
        try:
            # Use the analyze_paper task
            result = analyze_paper.delay(str(paper.id))
            analyzed_count += 1
        except Exception as e:
            logger.error(f"Failed to queue analysis for paper {paper.id}: {str(e)}")
            failed_count += 1
    
    logger.info(f"Queued {analyzed_count} papers for analysis, {failed_count} failed")
    return {
        'analyzed_count': analyzed_count,
        'failed_count': failed_count
    }


@shared_task(bind=True, name='papers.update_paper_metrics')
def update_paper_metrics(self):
    """
    Periodic task to update paper metrics (citations, downloads, etc.).
    """
    # This would typically involve calling external APIs to get updated metrics
    # For now, we'll just log the task execution
    logger.info("Updating paper metrics")
    return {'status': 'completed'}


@shared_task(bind=True, name='papers.cleanup_old_tasks')
def cleanup_old_tasks(self, days: int = 7):
    """
    Clean up old scraping tasks and their associated data.
    
    Args:
        days: Number of days to keep tasks
    """
    cutoff_date = timezone.now() - timezone.timedelta(days=days)
    
    # Delete old completed/failed tasks
    deleted_count = ScrapingTask.objects.filter(
        created_at__lt=cutoff_date,
        status__in=['completed', 'failed']
    ).delete()[0]
    
    logger.info(f"Cleaned up {deleted_count} old scraping tasks")
    return {'deleted_count': deleted_count}


@shared_task(bind=True, name='papers.export_papers')
def export_papers(self, user_id: int, format: str = 'csv', filters: Dict[str, Any] = None):
    """
    Background task to export papers in various formats.
    
    Args:
        user_id: ID of the user requesting the export
        format: Export format ('csv', 'json', 'pdf')
        filters: Filters to apply to the papers
    """
    try:
        from papers.services.export_service import ExportService
        
        export_service = ExportService()
        export_file = export_service.export_papers(user_id, format, filters)
        
        logger.info(f"Successfully exported papers for user {user_id} in {format} format")
        return {
            'status': 'success',
            'export_file': export_file,
            'format': format
        }
        
    except Exception as e:
        logger.error(f"Export failed for user {user_id}: {str(e)}")
        raise


@shared_task(bind=True, name='papers.sync_paper_data')
def sync_paper_data(self, paper_id: str = None):
    """
    Sync paper data with external sources to get updated information.
    
    Args:
        paper_id: Specific paper ID to sync, or None for all papers
    """
    if paper_id:
        papers = Paper.objects.filter(id=paper_id)
    else:
        # Sync papers that haven't been updated recently
        cutoff_date = timezone.now() - timezone.timedelta(days=30)
        papers = Paper.objects.filter(
            updated_at__lt=cutoff_date,
            source__in=['arxiv', 'semantic_scholar']
        )
    
    updated_count = 0
    
    for paper in papers:
        try:
            if paper.source == 'arxiv' and paper.arxiv_id:
                scraper = ArXivScraper()
                updated_data = scraper.get_paper_details(paper.arxiv_id)
            elif paper.source == 'semantic_scholar' and paper.id:
                scraper = SemanticScholarScraper()
                updated_data = scraper.get_paper_details(paper.id)
            else:
                continue
            
            # Update paper with new data
            if updated_data:
                paper.citation_count = updated_data.get('citation_count', paper.citation_count)
                paper.download_count = updated_data.get('download_count', paper.download_count)
                paper.view_count = updated_data.get('view_count', paper.view_count)
                paper.save()
                updated_count += 1
                
        except Exception as e:
            logger.error(f"Failed to sync paper {paper.id}: {str(e)}")
    
    logger.info(f"Synced {updated_count} papers")
    return {'updated_count': updated_count} 