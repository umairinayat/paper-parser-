"""
Celery tasks for AI analysis operations.
"""

import logging
from typing import List, Dict, Any
from celery import shared_task
from django.conf import settings
from django.utils import timezone
from django.db import transaction

from papers.models import Paper, PaperAnalysis
from analysis.services.ai_analysis_service import AIAnalysisService

logger = logging.getLogger(__name__)


@shared_task(bind=True, name='analysis.analyze_pending_papers')
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
            # Queue individual analysis task
            result = analyze_single_paper.delay(str(paper.id))
            analyzed_count += 1
            logger.info(f"Queued analysis for paper {paper.id}")
        except Exception as e:
            logger.error(f"Failed to queue analysis for paper {paper.id}: {str(e)}")
            failed_count += 1
    
    logger.info(f"Queued {analyzed_count} papers for analysis, {failed_count} failed")
    return {
        'analyzed_count': analyzed_count,
        'failed_count': failed_count
    }


@shared_task(bind=True, name='analysis.analyze_single_paper')
def analyze_single_paper(self, paper_id: str):
    """
    Analyze a single paper using AI.
    
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
        
        # Perform analysis
        analysis_service = AIAnalysisService()
        analysis_result = analysis_service.analyze_paper(paper)
        
        # Create analysis record
        with transaction.atomic():
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


@shared_task(bind=True, name='analysis.batch_analyze_papers')
def batch_analyze_papers(self, paper_ids: List[str]):
    """
    Analyze multiple papers in batch.
    
    Args:
        paper_ids: List of paper UUIDs to analyze
    """
    results = {
        'successful': [],
        'failed': [],
        'already_analyzed': []
    }
    
    for paper_id in paper_ids:
        try:
            result = analyze_single_paper.delay(paper_id)
            results['successful'].append(paper_id)
        except Exception as e:
            logger.error(f"Failed to queue analysis for paper {paper_id}: {str(e)}")
            results['failed'].append(paper_id)
    
    logger.info(f"Batch analysis queued: {len(results['successful'])} successful, {len(results['failed'])} failed")
    return results


@shared_task(bind=True, name='analysis.regenerate_analysis')
def regenerate_analysis(self, paper_id: str):
    """
    Regenerate analysis for a paper (overwrites existing analysis).
    
    Args:
        paper_id: UUID of the paper to re-analyze
    """
    try:
        paper = Paper.objects.get(id=paper_id)
        
        # Delete existing analysis if any
        if paper.has_analysis:
            paper.analysis.delete()
        
        # Queue new analysis
        result = analyze_single_paper.delay(paper_id)
        
        logger.info(f"Regeneration queued for paper {paper_id}")
        return {
            'status': 'queued',
            'paper_id': paper_id
        }
        
    except Paper.DoesNotExist:
        logger.error(f"Paper {paper_id} not found")
        raise
    except Exception as e:
        logger.error(f"Failed to queue regeneration for paper {paper_id}: {str(e)}")
        raise


@shared_task(bind=True, name='analysis.cleanup_failed_analyses')
def cleanup_failed_analyses(self, days: int = 7):
    """
    Clean up papers with failed analysis status.
    
    Args:
        days: Number of days to look back
    """
    cutoff_date = timezone.now() - timezone.timedelta(days=days)
    
    # Reset failed analyses to pending
    failed_papers = Paper.objects.filter(
        analysis_status='failed',
        updated_at__lt=cutoff_date
    )
    
    updated_count = failed_papers.update(
        analysis_status='pending',
        error_message='',
        processed_at=None,
        processing_time=None
    )
    
    logger.info(f"Reset {updated_count} failed analyses to pending")
    return {'reset_count': updated_count}


@shared_task(bind=True, name='analysis.generate_summary')
def generate_summary(self, paper_id: str):
    """
    Generate a summary for a paper.
    
    Args:
        paper_id: UUID of the paper to summarize
    """
    try:
        paper = Paper.objects.get(id=paper_id)
        
        analysis_service = AIAnalysisService()
        summary = analysis_service.generate_summary(paper)
        
        # Update paper summary
        paper.summary = summary
        paper.save()
        
        logger.info(f"Generated summary for paper {paper_id}")
        return {
            'status': 'success',
            'paper_id': paper_id,
            'summary_length': len(summary)
        }
        
    except Paper.DoesNotExist:
        logger.error(f"Paper {paper_id} not found")
        raise
    except Exception as e:
        logger.error(f"Summary generation failed for paper {paper_id}: {str(e)}")
        raise


@shared_task(bind=True, name='analysis.extract_key_findings')
def extract_key_findings(self, paper_id: str):
    """
    Extract key findings from a paper.
    
    Args:
        paper_id: UUID of the paper to extract findings from
    """
    try:
        paper = Paper.objects.get(id=paper_id)
        
        analysis_service = AIAnalysisService()
        findings = analysis_service.extract_key_findings(paper)
        
        # Update paper analysis if it exists
        if paper.has_analysis:
            paper.analysis.key_findings = findings
            paper.analysis.save()
        else:
            # Create basic analysis record
            PaperAnalysis.objects.create(
                paper=paper,
                analysis_type='key_findings',
                key_findings=findings,
                summary='',
                methodology='',
                limitations='',
                future_work='',
                impact_assessment='',
                methodology_type='',
                dataset_info='',
                evaluation_metrics=[],
                model_used='',
                processing_time=0,
                confidence_score=0.8,
            )
        
        logger.info(f"Extracted {len(findings)} key findings for paper {paper_id}")
        return {
            'status': 'success',
            'paper_id': paper_id,
            'findings_count': len(findings)
        }
        
    except Paper.DoesNotExist:
        logger.error(f"Paper {paper_id} not found")
        raise
    except Exception as e:
        logger.error(f"Key findings extraction failed for paper {paper_id}: {str(e)}")
        raise


@shared_task(bind=True, name='analysis.update_analysis_embeddings')
def update_analysis_embeddings(self, analysis_id: str = None):
    """
    Update vector embeddings for paper analyses.
    
    Args:
        analysis_id: Specific analysis ID to update, or None for all
    """
    try:
        from analysis.services.embedding_service import EmbeddingService
        
        embedding_service = EmbeddingService()
        
        if analysis_id:
            analyses = PaperAnalysis.objects.filter(id=analysis_id)
        else:
            # Update analyses without embeddings
            analyses = PaperAnalysis.objects.filter(
                summary_embedding__isnull=True
            )[:100]  # Limit to 100 at a time
        
        updated_count = 0
        
        for analysis in analyses:
            try:
                # Generate embeddings for summary and key findings
                if analysis.summary:
                    summary_embedding = embedding_service.generate_embedding(analysis.summary)
                    analysis.summary_embedding = summary_embedding
                
                if analysis.key_findings:
                    findings_text = ' '.join(analysis.key_findings)
                    findings_embedding = embedding_service.generate_embedding(findings_text)
                    analysis.key_findings_embedding = findings_embedding
                
                analysis.save()
                updated_count += 1
                
            except Exception as e:
                logger.error(f"Failed to update embeddings for analysis {analysis.id}: {str(e)}")
        
        logger.info(f"Updated embeddings for {updated_count} analyses")
        return {'updated_count': updated_count}
        
    except Exception as e:
        logger.error(f"Embedding update failed: {str(e)}")
        raise 