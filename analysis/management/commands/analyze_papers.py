"""
Django management command to analyze papers using AI.
"""

import logging
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from django.utils import timezone

from papers.models import Paper, PaperAnalysis
from analysis.services.ai_analysis_service import AIAnalysisService

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Analyze papers using AI (OpenAI/Groq)'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--paper-id',
            type=str,
            help='Specific paper ID to analyze'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=10,
            help='Number of papers to analyze in batch'
        )
        parser.add_argument(
            '--status',
            type=str,
            choices=['pending', 'failed', 'all'],
            default='pending',
            help='Status of papers to analyze'
        )
        parser.add_argument(
            '--regenerate',
            action='store_true',
            help='Regenerate analysis for papers that already have analysis'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be analyzed without actually doing it'
        )
    
    def handle(self, *args, **options):
        paper_id = options.get('paper_id')
        batch_size = options['batch_size']
        status = options['status']
        regenerate = options['regenerate']
        dry_run = options['dry_run']
        
        if dry_run:
            self.stdout.write(
                self.style.WARNING('DRY RUN MODE - No papers will be analyzed')
            )
        
        try:
            if paper_id:
                self._analyze_single_paper(paper_id, dry_run)
            else:
                self._analyze_batch_papers(batch_size, status, regenerate, dry_run)
            
            self.stdout.write(
                self.style.SUCCESS('Paper analysis completed successfully')
            )
            
        except Exception as e:
            raise CommandError(f'Analysis failed: {str(e)}')
    
    def _analyze_single_paper(self, paper_id, dry_run):
        """Analyze a single paper."""
        try:
            paper = Paper.objects.get(id=paper_id)
            
            if dry_run:
                self.stdout.write(f'Would analyze paper: {paper.title}')
                return
            
            self.stdout.write(f'Analyzing paper: {paper.title}')
            
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
            
            self.stdout.write(
                self.style.SUCCESS(f'Successfully analyzed paper: {paper.title}')
            )
            
        except Paper.DoesNotExist:
            raise CommandError(f'Paper {paper_id} not found')
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Failed to analyze paper {paper_id}: {str(e)}')
            )
            raise
    
    def _analyze_batch_papers(self, batch_size, status, regenerate, dry_run):
        """Analyze multiple papers in batch."""
        # Build query
        if status == 'all':
            papers_query = Paper.objects.all()
        elif status == 'failed':
            papers_query = Paper.objects.filter(analysis_status='failed')
        else:  # pending
            papers_query = Paper.objects.filter(analysis_status='pending')
        
        if not regenerate:
            # Exclude papers that already have analysis
            papers_query = papers_query.filter(analysis__isnull=True)
        
        papers = papers_query.order_by('created_at')[:batch_size]
        
        if dry_run:
            self.stdout.write(f'Would analyze {papers.count()} papers')
            for paper in papers[:5]:
                self.stdout.write(f'  - {paper.title}')
            if papers.count() > 5:
                self.stdout.write(f'  ... and {papers.count() - 5} more papers')
            return
        
        self.stdout.write(f'Analyzing {papers.count()} papers...')
        
        analyzed_count = 0
        failed_count = 0
        
        for paper in papers:
            try:
                self.stdout.write(f'Analyzing: {paper.title}')
                
                # Delete existing analysis if regenerating
                if regenerate and paper.has_analysis:
                    paper.analysis.delete()
                
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
                
                analyzed_count += 1
                self.stdout.write(
                    self.style.SUCCESS(f'✓ Analyzed: {paper.title}')
                )
                
            except Exception as e:
                failed_count += 1
                self.stdout.write(
                    self.style.ERROR(f'✗ Failed to analyze {paper.title}: {str(e)}')
                )
                
                # Update paper status
                paper.analysis_status = 'failed'
                paper.error_message = str(e)
                paper.save()
        
        self.stdout.write(
            self.style.SUCCESS(
                f'Analysis completed: {analyzed_count} successful, {failed_count} failed'
            )
        ) 