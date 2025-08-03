import os
import logging
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Q
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_http_methods
from django.utils import timezone
import csv
import json

from .models import Paper, PaperAnalysis, SearchQuery
from .forms import PaperUploadForm, PaperEditForm, PaperSearchForm
from .services.unified_search_service import UnifiedSearchService, SearchFilter

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
            if paper.file:
                paper.file_size = paper.file.size
            
            # Extract basic metadata from filename if title is not provided
            if not paper.title and paper.file:
                paper.title = os.path.splitext(paper.file.name)[0].split('/')[-1]
            
            paper.save()
            
            # Handle author assignment if we have stored authors
            if hasattr(paper, '_author_list'):
                paper.authors.set(paper._author_list)
            
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
    papers = Paper.objects.filter(user=request.user)
    
    # Search functionality
    search_query = request.GET.get('search', '')
    if search_query:
        papers = papers.filter(
            Q(title__icontains=search_query) |
            Q(abstract__icontains=search_query) |
            Q(first_author__icontains=search_query)
        )
    
    # Status filter
    status_filter = request.GET.get('status', '')
    if status_filter:
        papers = papers.filter(analysis_status=status_filter)
    
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
    
    # Get analysis if it exists
    analysis = None
    if hasattr(paper, 'analysis'):
        analysis = paper.analysis
    
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
            paper = form.save(commit=False)
            paper.save()
            
            # Handle author assignment if we have stored authors
            if hasattr(paper, '_author_list'):
                paper.authors.set(paper._author_list)
            
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
    """Delete a paper"""
    paper = get_object_or_404(Paper, id=paper_id, user=request.user)
    
    if request.method == 'POST':
        title = paper.title
        paper.delete()
        messages.success(request, f'Paper "{title}" deleted successfully!')
        return redirect('papers:list')
    
    return render(request, 'papers/delete_confirm.html', {'paper': paper})

@login_required
@require_http_methods(["POST"])
def start_analysis(request, paper_id):
    """Start AI analysis for a paper"""
    paper = get_object_or_404(Paper, id=paper_id, user=request.user)
    
    if paper.analysis_status == 'pending':
        from papers.tasks import analyze_paper
        analyze_paper.delay(str(paper.id))
        messages.success(request, 'Analysis started! Check back in a few minutes.')
    else:
        messages.warning(request, 'Analysis is already in progress or completed.')
    
    return redirect('papers:detail', paper_id=paper.id)

@login_required
def analysis_status(request, paper_id):
    """Get analysis status via AJAX"""
    paper = get_object_or_404(Paper, id=paper_id, user=request.user)
    
    return JsonResponse({
        'status': paper.analysis_status,
        'processing_time': paper.processing_time,
        'error_message': paper.error_message,
    })

@login_required
def paper_search(request):
    """Handle comprehensive paper search using unified search service"""
    if request.method == 'POST':
        form = PaperSearchForm(request.POST)
        if form.is_valid():
            query = form.cleaned_data['query']
            max_papers = form.cleaned_data['max_papers']
            sources = form.cleaned_data['sources']
            year_from = form.cleaned_data.get('year_from')
            year_to = form.cleaned_data.get('year_to')
            min_citations = form.cleaned_data.get('min_citations')
            subject_areas = form.cleaned_data.get('subject_areas', [])
            open_access_only = form.cleaned_data.get('open_access_only', False)
            
            try:
                # Create search filter
                search_filter = SearchFilter(
                    sources=sources,
                    year_from=year_from,
                    year_to=year_to,
                    min_citations=min_citations,
                    subject_areas=subject_areas,
                    open_access_only=open_access_only
                )
                
                logger.info(f"Starting search: query='{query}', sources={sources}, max_results={max_papers}")
                
                # Perform unified search
                search_service = UnifiedSearchService()
                results = search_service.search_papers(
                    query=query,
                    max_results=max_papers,
                    search_filter=search_filter,
                    user=request.user
                )
                
                logger.info(f"Search completed: found={results['total_found']}, saved={results['total_returned']}")
                
                messages.success(
                    request, 
                    f'Search completed! Found {results["total_returned"]} papers from '
                    f'{results["total_found"]} total results across {len(sources)} sources.'
                )
                
                # Store search results in session for display
                request.session['last_search_id'] = str(results['search_query'].id)
                
                return redirect('papers:search_results')
                
            except Exception as e:
                logger.error(f"Search failed: {str(e)}")
                messages.error(request, f'Search failed: {str(e)}')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = PaperSearchForm()
    
    return render(request, 'papers/search.html', {'form': form})

@login_required
def search_results(request):
    """Display search results with enhanced filtering and analysis"""
    # Get the latest search or specific search from session/parameter
    search_query = None
    papers = []
    
    # Check for specific search_id in GET parameters first
    search_id = request.GET.get('search_id') or request.session.get('last_search_id')
    if search_id:
        try:
            search_query = SearchQuery.objects.get(id=search_id, user=request.user)
            # Get papers from this search with their ranking scores
            search_results = search_query.searchresult_set.select_related('paper').order_by('rank')
            papers = [sr.paper for sr in search_results]
        except SearchQuery.DoesNotExist:
            pass
    
    # Get recent search queries for this user
    search_queries = SearchQuery.objects.filter(user=request.user).order_by('-created_at')[:10]
    
    # Get all analyzed papers as fallback
    if not papers:
        papers = Paper.objects.filter(
            user=request.user,
            analysis_status='completed'
        ).order_by('-created_at')[:20]
    
    # Apply filters if provided
    search_filter = request.GET.get('search', '')
    source_filter = request.GET.get('source', '')
    year_filter = request.GET.get('year', '')
    citation_filter = request.GET.get('min_citations', '')
    
    if search_filter:
        papers = [p for p in papers if search_filter.lower() in p.title.lower()]
    
    if source_filter:
        papers = [p for p in papers if p.source == source_filter]
    
    if year_filter:
        try:
            year = int(year_filter)
            papers = [p for p in papers if p.publication_date and p.publication_date.year == year]
        except ValueError:
            pass
    
    if citation_filter:
        try:
            min_cit = int(citation_filter)
            papers = [p for p in papers if p.citation_count >= min_cit]
        except ValueError:
            pass
    
    # Pagination
    paginator = Paginator(papers, 20)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # Get source statistics
    source_stats = {}
    if search_query and hasattr(search_query, 'filters') and 'results_by_source' in search_query.filters:
        source_stats = search_query.filters.get('results_by_source', {})
    
    context = {
        'search_query': search_query,
        'page_obj': page_obj,
        'search_queries': search_queries,
        'source_stats': source_stats,
        'filters': {
            'search': search_filter,
            'source': source_filter,
            'year': year_filter,
            'min_citations': citation_filter,
        },
        'total_papers': len(papers),
    }
    return render(request, 'papers/search_results_simple.html', context)

@login_required
def export_analysis_csv(request):
    """Export analysis results as CSV"""
    papers = Paper.objects.filter(
        user=request.user,
        analysis_status='completed'
    ).select_related('analysis')
    
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="paper_analysis.csv"'
    
    writer = csv.writer(response)
    writer.writerow([
        'Title', 'Authors', 'Publication Date', 'Journal', 'DOI',
        'ArXiv ID', 'PubMed ID', 'Source', 'Source URL', 'Paper URL',
        'Citation Count', 'Keywords', 'Subject Areas', 'Abstract',
        'Summary', 'Key Findings', 'Methodology', 'Limitations',
        'Analysis Date', 'Created Date'
    ])
    
    for paper in papers:
        analysis = getattr(paper, 'analysis', None)
        
        # Generate paper URL based on identifiers
        paper_url = ''
        if paper.doi:
            paper_url = f"https://doi.org/{paper.doi}"
        elif paper.arxiv_id:
            paper_url = f"https://arxiv.org/abs/{paper.arxiv_id}"
        elif paper.pmid:
            paper_url = f"https://pubmed.ncbi.nlm.nih.gov/{paper.pmid}/"
        elif paper.source_url:
            paper_url = paper.source_url
        
        writer.writerow([
            paper.title,
            paper.authors,  # Already a string in our schema
            paper.publication_date,
            paper.journal.name if paper.journal else '',
            paper.doi or '',
            paper.arxiv_id or '',
            paper.pmid or '',
            paper.get_source_display(),
            paper.source_url or '',
            paper_url,
            paper.citation_count,
            str(paper.keywords) if paper.keywords else '',
            str(paper.subject_areas) if paper.subject_areas else '',
            paper.abstract,
            analysis.summary if analysis else '',
            '; '.join(analysis.key_findings) if analysis and analysis.key_findings else '',
            analysis.methodology if analysis else '',
            analysis.limitations if analysis else '',
            analysis.created_at if analysis else '',
            paper.created_at,
        ])
    
    return response


@login_required
def export_analysis_json(request):
    """Export analysis results as JSON"""
    papers = Paper.objects.filter(
        user=request.user,
        analysis_status='completed'
    ).select_related('analysis')
    
    data = {
        'search_metadata': {
            'exported_at': timezone.now().isoformat(),
            'total_papers': papers.count(),
            'user': request.user.username
        },
        'papers': []
    }
    
    for paper in papers:
        analysis = getattr(paper, 'analysis', None)
        paper_data = {
            'id': str(paper.id),
            'title': paper.title,
            'authors': paper.authors,
            'publication_date': paper.publication_date.isoformat() if paper.publication_date else None,
            'journal': paper.journal.name if paper.journal else None,
            'doi': paper.doi or None,
            'arxiv_id': paper.arxiv_id or None,
            'pmid': paper.pmid or None,
            'source': paper.source,
            'source_url': paper.source_url,
            'citation_count': paper.citation_count,
            'keywords': paper.keywords,
            'subject_areas': paper.subject_areas,
            'abstract': paper.abstract,
        }
        
        if analysis:
            paper_data['analysis'] = {
                'summary': analysis.summary,
                'key_findings': analysis.key_findings,
                'methodology': analysis.methodology,
                'limitations': analysis.limitations,
                'future_work': analysis.future_work,
                'impact_assessment': analysis.impact_assessment,
                'methodology_type': analysis.methodology_type,
                'dataset_info': analysis.dataset_info,
                'evaluation_metrics': analysis.evaluation_metrics,
                'confidence_score': analysis.confidence_score,
                'analysis_date': analysis.created_at.isoformat(),
            }
        
        data['papers'].append(paper_data)
    
    response = HttpResponse(
        json.dumps(data, indent=2, ensure_ascii=False),
        content_type='application/json'
    )
    response['Content-Disposition'] = 'attachment; filename="paper_analysis.json"'
    
    return response


@login_required
def similar_papers(request, paper_id):
    """Find papers similar to the given paper"""
    paper = get_object_or_404(Paper, id=paper_id, user=request.user)
    
    try:
        search_service = UnifiedSearchService()
        similar_papers = search_service.get_similar_papers(paper_id, max_results=20)
        
        context = {
            'paper': paper,
            'similar_papers': similar_papers,
        }
        return render(request, 'papers/similar_papers.html', context)
        
    except Exception as e:
        logger.error(f"Failed to find similar papers: {str(e)}")
        messages.error(request, f'Failed to find similar papers: {str(e)}')
        return redirect('papers:detail', paper_id=paper_id)


@login_required
def bulk_analysis(request):
    """Start bulk analysis for multiple papers"""
    if request.method == 'POST':
        paper_ids = request.POST.getlist('paper_ids')
        if paper_ids:
            # Queue analysis tasks for selected papers
            from papers.tasks import analyze_paper
            for paper_id in paper_ids:
                try:
                    paper = Paper.objects.get(id=paper_id, user=request.user)
                    if paper.analysis_status == 'pending':
                        analyze_paper.delay(str(paper.id))
                except Paper.DoesNotExist:
                    continue
            
            messages.success(request, f'Analysis started for {len(paper_ids)} papers!')
        else:
            messages.warning(request, 'No papers selected for analysis.')
    
    return redirect('papers:search_results')


@login_required 
def search_suggestions(request):
    """Provide search suggestions based on user's previous searches and popular topics"""
    recent_queries = SearchQuery.objects.filter(user=request.user).order_by('-created_at')[:5]
    
    # Popular search terms (could be enhanced with analytics)
    popular_topics = [
        'machine learning',
        'artificial intelligence', 
        'deep learning',
        'natural language processing',
        'computer vision',
        'reinforcement learning',
        'neural networks',
        'data science',
        'bioinformatics',
        'climate change'
    ]
    
    context = {
        'recent_queries': recent_queries,
        'popular_topics': popular_topics,
    }
    
    return JsonResponse(context)
