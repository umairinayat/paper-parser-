import os
import json
import logging
from datetime import datetime
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.core.paginator import Paginator
from django.db.models import Q
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .models import Paper
from .forms import PaperUploadForm, PaperEditForm, PaperSearchForm
from .services.comprehensive_paper_search_agent import PaperSearchAgent
from analysis.models import PaperAnalysis

# Set up logging
logger = logging.getLogger(__name__)

@login_required
def paper_upload(request):
    """Handle paper upload with file validation"""
    if request.method == 'POST':
        form = PaperUploadForm(request.POST, request.FILES)
        if form.is_valid():
            paper = form.save(commit=False)
            paper.user = request.user
            paper.file_size = paper.file.size
            
            # Extract basic metadata from filename if title is not provided
            if not paper.title:
                paper.title = os.path.splitext(paper.file.name)[0].split('/')[-1]
            
            paper.save()
            messages.success(request, f'Paper "{paper.title}" uploaded successfully! Analysis will begin shortly.')
            return redirect('papers:detail', paper_id=paper.id)
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = PaperUploadForm()
    
    return render(request, 'papers/upload.html', {'form': form})

@login_required
def paper_list(request):
    """Display user's uploaded papers with filtering and pagination"""
    papers = request.user.paper_set.all()
    
    # Search functionality
    search_query = request.GET.get('search', '')
    if search_query:
        papers = papers.filter(
            Q(title__icontains=search_query) |
            Q(authors__icontains=search_query) |
            Q(abstract__icontains=search_query)
        )
    
    # Status filter
    status_filter = request.GET.get('status', '')
    if status_filter:
        papers = papers.filter(status=status_filter)
    
    # Pagination
    paginator = Paginator(papers, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'search_query': search_query,
        'status_filter': status_filter,
        'total_papers': papers.count(),
    }
    return render(request, 'papers/list.html', context)

@login_required
def paper_detail(request, paper_id):
    """Display paper details and analysis results"""
    paper = get_object_or_404(Paper, id=paper_id, user=request.user)
    
    try:
        analysis = paper.analysis
    except PaperAnalysis.DoesNotExist:
        analysis = None
    
    context = {
        'paper': paper,
        'analysis': analysis,
    }
    return render(request, 'papers/detail.html', context)

@login_required
def paper_edit(request, paper_id):
    """Edit paper metadata"""
    paper = get_object_or_404(Paper, id=paper_id, user=request.user)
    
    if request.method == 'POST':
        form = PaperEditForm(request.POST, instance=paper)
        if form.is_valid():
            form.save()
            messages.success(request, 'Paper metadata updated successfully!')
            return redirect('papers:detail', paper_id=paper.id)
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = PaperEditForm(instance=paper)
    
    context = {
        'form': form,
        'paper': paper,
    }
    return render(request, 'papers/edit.html', context)

@login_required
def paper_delete(request, paper_id):
    """Delete a paper and its associated analysis"""
    paper = get_object_or_404(Paper, id=paper_id, user=request.user)
    
    if request.method == 'POST':
        paper_title = paper.title
        paper.delete()
        messages.success(request, f'Paper "{paper_title}" deleted successfully!')
        return redirect('papers:list')
    
    context = {
        'paper': paper,
    }
    return render(request, 'papers/delete_confirm.html', context)

@login_required
@require_http_methods(["POST"])
def start_analysis(request, paper_id):
    """Start analysis for a paper"""
    paper = get_object_or_404(Paper, id=paper_id, user=request.user)
    
    if paper.status == 'pending':
        paper.status = 'processing'
        paper.save()
        
        # TODO: Trigger async analysis task here
        # For now, we'll just update the status
        messages.success(request, f'Analysis started for "{paper.title}"')
    else:
        messages.warning(request, f'Analysis for "{paper.title}" is already in progress or completed.')
    
    return JsonResponse({'status': 'success', 'paper_status': paper.status})

@login_required
def analysis_status(request, paper_id):
    """Get analysis status for AJAX requests"""
    paper = get_object_or_404(Paper, id=paper_id, user=request.user)
    
    return JsonResponse({
        'status': paper.status,
        'processing_time': paper.processing_time,
        'has_analysis': hasattr(paper, 'analysis'),
    })

@login_required
def paper_search(request):
    """Search for papers using AI agent"""
    logger.info("Paper search view called")
    if request.method == 'POST':
        logger.info("Processing POST request for paper search")
        form = PaperSearchForm(request.POST)
        if form.is_valid():
            query = form.cleaned_data['query']
            max_papers = form.cleaned_data['max_papers']
            logger.info(f"Form valid. Query: {query}, Max papers: {max_papers}")
            
            try:
                # Initialize the AI agent
                logger.info("Initializing PaperSearchAgent")
                agent = PaperSearchAgent()
                logger.info("PaperSearchAgent initialized successfully")
                
                # Run the analysis (using asyncio to handle async operation)
                logger.info("Starting async analysis")
                import asyncio
                analysis_result = asyncio.run(agent.run_full_analysis(query, max_papers))
                logger.info(f"Analysis completed. Result: {analysis_result}")
                
                # Store the analysis result in session for export
                request.session['last_analysis'] = analysis_result
                
                context = {
                    'query': query,
                    'analysis_result': analysis_result,
                    'papers_count': len(analysis_result.get('papers', [])),
                    'analysis_summary': analysis_result.get('analysis_summary', ''),
                    'papers': analysis_result.get('papers', []),
                    'detailed_analysis': analysis_result.get('detailed_analysis', {}),
                    'citations': analysis_result.get('citations', []),
                    'methodology_comparison': analysis_result.get('methodology_comparison', {}),
                    'statistical_data': analysis_result.get('statistical_data', []),
                    'cross_references': analysis_result.get('cross_references', []),
                    'comprehensive_synthesis': analysis_result.get('comprehensive_synthesis', ''),
                    'impact_assessment': analysis_result.get('impact_assessment', {})
                }
                
                logger.info("Rendering simple search results template")
                return render(request, 'papers/simple_search_results.html', context)
                
            except Exception as e:
                logger.error(f"Error during analysis: {e}", exc_info=True)
                messages.error(request, f'Error during analysis: {str(e)}')
                return render(request, 'papers/search.html', {'form': form})
        else:
            logger.warning("Form validation failed")
            messages.error(request, 'Please correct the errors below.')
    else:
        logger.info("Rendering search form")
        form = PaperSearchForm()
    
    return render(request, 'papers/search.html', {'form': form})

@login_required
def search_results(request):
    """Display search results"""
    analysis_result = request.session.get('last_analysis')
    
    if not analysis_result:
        messages.error(request, 'No analysis results found.')
        return redirect('papers:search')
    
    context = {
        'analysis_result': analysis_result,
        'papers': analysis_result.get('papers', []),
        'analysis_summary': analysis_result.get('analysis_summary', ''),
        'query': analysis_result.get('query', ''),
        'detailed_analysis': analysis_result.get('detailed_analysis', {}),
        'citations': analysis_result.get('citations', []),
        'methodology_comparison': analysis_result.get('methodology_comparison', {}),
        'statistical_data': analysis_result.get('statistical_data', []),
        'cross_references': analysis_result.get('cross_references', []),
        'comprehensive_synthesis': analysis_result.get('comprehensive_synthesis', ''),
        'impact_assessment': analysis_result.get('impact_assessment', {})
    }
    
    return render(request, 'papers/enhanced_search_results.html', context)

@login_required
def export_analysis_csv(request):
    """Export analysis results to CSV"""
    analysis_result = request.session.get('last_analysis')
    
    if not analysis_result:
        messages.error(request, 'No analysis results to export.')
        return redirect('papers:search')
    
    try:
        agent = PaperSearchAgent()
        query_name = analysis_result.get('query', 'research').replace(' ', '_').replace('/', '_').replace('\\', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{query_name}_{timestamp}.csv"
        filepath = agent.export_to_csv(analysis_result, filename)
        
        # Read the file and serve as download
        with open(filepath, 'rb') as f:
            response = HttpResponse(f.read(), content_type='text/csv')
            response['Content-Disposition'] = f'attachment; filename="{filename}"'
            return response
            
    except Exception as e:
        messages.error(request, f'Error exporting CSV: {str(e)}')
        return redirect('papers:search_results')
