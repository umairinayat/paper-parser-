from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, HttpResponse
from django.contrib import messages
from papers.models import Paper, PaperAnalysis
from .models import AnalysisSession, AnalysisResult, AnalysisTemplate
import json
import csv
from io import StringIO

# Create your views here.

@login_required
def analysis_results(request, paper_id):
    """Display detailed analysis results for a paper"""
    paper = get_object_or_404(Paper, id=paper_id, user=request.user)
    
    # Get analysis if it exists
    analysis = None
    if hasattr(paper, 'analysis'):
        analysis = paper.analysis
    
    if not analysis:
        messages.warning(request, 'Analysis not found for this paper.')
    
    context = {
        'paper': paper,
        'analysis': analysis,
    }
    return render(request, 'analysis/results.html', context)

@login_required
def export_analysis(request, paper_id, format='json'):
    """Export analysis results in various formats"""
    paper = get_object_or_404(Paper, id=paper_id, user=request.user)
    
    # Get analysis if it exists
    analysis = None
    if hasattr(paper, 'analysis'):
        analysis = paper.analysis
    
    if not analysis:
        messages.error(request, 'Analysis not found for this paper.')
        return redirect('papers:detail', paper_id=paper_id)
    
    if format == 'json':
        # Export as JSON
        data = {
            'paper': {
                'title': paper.title,
                'authors': [author.name for author in paper.authors.all()],
                'publication_date': paper.publication_date,
                'doi': paper.doi,
                'journal': paper.journal.name if paper.journal else None,
            },
            'analysis': {
                'summary': analysis.summary,
                'key_findings': analysis.key_findings,
                'methodology': analysis.methodology,
                'limitations': analysis.limitations,
                'future_work': analysis.future_work,
                'impact_assessment': analysis.impact_assessment,
                'methodology_type': analysis.methodology_type,
                'dataset_info': analysis.dataset_info,
                'evaluation_metrics': analysis.evaluation_metrics,
                'model_used': analysis.model_used,
                'processing_time': analysis.processing_time,
                'confidence_score': analysis.confidence_score,
            }
        }
        
        response = HttpResponse(
            json.dumps(data, indent=2, ensure_ascii=False),
            content_type='application/json'
        )
        response['Content-Disposition'] = f'attachment; filename="{paper.title}_analysis.json"'
        return response
    
    elif format == 'csv':
        # Export as CSV
        output = StringIO()
        writer = csv.writer(output)
        
        # Write headers
        writer.writerow([
            'Field', 'Value'
        ])
        
        # Write data
        analysis_fields = [
            ('Summary', analysis.summary),
            ('Key Findings', '; '.join(analysis.key_findings) if analysis.key_findings else ''),
            ('Methodology', analysis.methodology),
            ('Limitations', analysis.limitations),
            ('Future Work', analysis.future_work),
            ('Impact Assessment', analysis.impact_assessment),
            ('Methodology Type', analysis.methodology_type),
            ('Dataset Info', analysis.dataset_info),
            ('Evaluation Metrics', json.dumps(analysis.evaluation_metrics) if analysis.evaluation_metrics else ''),
            ('Model Used', analysis.model_used),
            ('Processing Time', analysis.processing_time),
            ('Confidence Score', analysis.confidence_score),
        ]
        
        for field, value in analysis_fields:
            writer.writerow([field, value])
        
        output.seek(0)
        response = HttpResponse(output.getvalue(), content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="{paper.title}_analysis.csv"'
        return response
    
    else:
        messages.error(request, 'Unsupported export format.')
        return redirect('analysis:results', paper_id=paper_id)

@login_required
def analysis_dashboard(request):
    """Display analysis dashboard with statistics"""
    # Get user's papers with analysis
    papers_with_analysis = Paper.objects.filter(
        user=request.user,
        analysis_status='completed'
    ).select_related('analysis')
    
    # Get analysis sessions
    analysis_sessions = AnalysisSession.objects.filter(user=request.user).order_by('-created_at')[:10]
    
    # Calculate statistics
    total_papers = Paper.objects.filter(user=request.user).count()
    analyzed_papers = papers_with_analysis.count()
    pending_papers = Paper.objects.filter(user=request.user, analysis_status='pending').count()
    failed_papers = Paper.objects.filter(user=request.user, analysis_status='failed').count()
    
    # Get recent analysis results
    recent_analyses = []
    for paper in papers_with_analysis[:5]:
        if hasattr(paper, 'analysis'):
            recent_analyses.append({
                'paper': paper,
                'analysis': paper.analysis
            })
    
    context = {
        'total_papers': total_papers,
        'analyzed_papers': analyzed_papers,
        'pending_papers': pending_papers,
        'failed_papers': failed_papers,
        'analysis_sessions': analysis_sessions,
        'recent_analyses': recent_analyses,
    }
    
    return render(request, 'analysis/dashboard.html', context)

@login_required
def analysis_status_ajax(request, paper_id):
    """Get analysis status via AJAX"""
    paper = get_object_or_404(Paper, id=paper_id, user=request.user)
    
    return JsonResponse({
        'status': paper.analysis_status,
        'processing_time': paper.processing_time,
        'error_message': paper.error_message,
        'has_analysis': hasattr(paper, 'analysis'),
    })
