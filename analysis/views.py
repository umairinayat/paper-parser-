from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, HttpResponse
from django.contrib import messages
from papers.models import Paper
from .models import PaperAnalysis
import json
import csv
from io import StringIO

# Create your views here.

@login_required
def analysis_results(request, paper_id):
    """Display detailed analysis results for a paper"""
    paper = get_object_or_404(Paper, id=paper_id, user=request.user)
    
    try:
        analysis = paper.analysis
    except PaperAnalysis.DoesNotExist:
        analysis = None
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
    
    try:
        analysis = paper.analysis
    except PaperAnalysis.DoesNotExist:
        messages.error(request, 'Analysis not found for this paper.')
        return redirect('paper_detail', paper_id=paper_id)
    
    if format == 'json':
        # Export as JSON
        data = {
            'paper': {
                'title': paper.title,
                'authors': paper.authors,
                'year': paper.year,
                'doi': paper.doi,
                'journal': paper.journal,
            },
            'analysis': {
                'abstract_summary': analysis.abstract_summary,
                'main_findings': analysis.main_findings,
                'key_conclusions': analysis.key_conclusions,
                'study_design': analysis.study_design,
                'study_objectives': analysis.study_objectives,
                'theoretical_framework': analysis.theoretical_framework,
                'research_question': analysis.research_question,
                'hypotheses_tested': analysis.hypotheses_tested,
                'intervention': analysis.intervention,
                'intervention_effects': analysis.intervention_effects,
                'outcome_measured': analysis.outcome_measured,
                'measurement_methods': analysis.measurement_methods,
                'primary_outcomes': analysis.primary_outcomes,
                'secondary_outcomes': analysis.secondary_outcomes,
                'statistical_significance': analysis.statistical_significance,
                'effect_sizes': analysis.effect_sizes,
                'limitations': analysis.limitations,
                'research_gaps': analysis.research_gaps,
                'future_research': analysis.future_research,
                'methodological_constraints': analysis.methodological_constraints,
                'introduction_summary': analysis.introduction_summary,
                'discussion_summary': analysis.discussion_summary,
                'key_arguments': analysis.key_arguments,
                'implications': analysis.implications,
                'related_papers': analysis.related_papers,
                'confidence_scores': analysis.confidence_scores,
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
            ('Abstract Summary', analysis.abstract_summary),
            ('Study Design', analysis.study_design),
            ('Research Question', analysis.research_question),
            ('Intervention', analysis.intervention),
            ('Intervention Effects', analysis.intervention_effects),
            ('Statistical Significance', analysis.statistical_significance),
            ('Introduction Summary', analysis.introduction_summary),
            ('Discussion Summary', analysis.discussion_summary),
        ]
        
        for field_name, value in analysis_fields:
            writer.writerow([field_name, value])
        
        # Write list fields
        list_fields = [
            ('Main Findings', analysis.main_findings),
            ('Key Conclusions', analysis.key_conclusions),
            ('Study Objectives', analysis.study_objectives),
            ('Hypotheses Tested', analysis.hypotheses_tested),
            ('Outcome Measured', analysis.outcome_measured),
            ('Primary Outcomes', analysis.primary_outcomes),
            ('Secondary Outcomes', analysis.secondary_outcomes),
            ('Effect Sizes', analysis.effect_sizes),
            ('Limitations', analysis.limitations),
            ('Research Gaps', analysis.research_gaps),
            ('Future Research', analysis.future_research),
            ('Key Arguments', analysis.key_arguments),
            ('Implications', analysis.implications),
        ]
        
        for field_name, value_list in list_fields:
            if value_list:
                for i, item in enumerate(value_list):
                    writer.writerow([f'{field_name} {i+1}', item])
            else:
                writer.writerow([field_name, ''])
        
        response = HttpResponse(
            output.getvalue(),
            content_type='text/csv'
        )
        response['Content-Disposition'] = f'attachment; filename="{paper.title}_analysis.csv"'
        return response
    
    else:
        messages.error(request, 'Invalid export format.')
        return redirect('analysis_results', paper_id=paper_id)

@login_required
def analysis_dashboard(request):
    """Dashboard showing all user's analyses with statistics"""
    user_papers = request.user.paper_set.all()
    
    # Statistics
    total_papers = user_papers.count()
    completed_analyses = user_papers.filter(status='completed').count()
    pending_analyses = user_papers.filter(status='pending').count()
    processing_analyses = user_papers.filter(status='processing').count()
    
    # Recent analyses
    recent_analyses = []
    for paper in user_papers.filter(status='completed')[:5]:
        try:
            recent_analyses.append(paper.analysis)
        except PaperAnalysis.DoesNotExist:
            pass
    
    context = {
        'total_papers': total_papers,
        'completed_analyses': completed_analyses,
        'pending_analyses': pending_analyses,
        'processing_analyses': processing_analyses,
        'recent_analyses': recent_analyses,
    }
    return render(request, 'analysis/dashboard.html', context)

@login_required
def analysis_status_ajax(request, paper_id):
    """AJAX endpoint for getting analysis status"""
    paper = get_object_or_404(Paper, id=paper_id, user=request.user)
    
    try:
        analysis = paper.analysis
        has_analysis = True
        analysis_complete = analysis.has_complete_analysis
    except PaperAnalysis.DoesNotExist:
        has_analysis = False
        analysis_complete = False
    
    return JsonResponse({
        'status': paper.status,
        'has_analysis': has_analysis,
        'analysis_complete': analysis_complete,
        'processing_time': paper.processing_time,
    })
