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

from .models import Paper, PaperAnalysis, SearchQuery, QASession, PaperQuestion, QuestionTemplate, PaperTag, ResearchNote
from .forms import PaperUploadForm, PaperEditForm, PaperSearchForm
from .forms_simple import SimpleUploadForm
from .services.unified_search_service import UnifiedSearchService, SearchFilter
from .services.rag_service import RAGService
from .services.comparative_analysis_service import ComparativeAnalysisService

# Set up logging
logger = logging.getLogger(__name__)

@login_required
def paper_upload(request):
    """Handle simplified paper upload - just file required"""
    if request.method == 'POST':
        form = SimpleUploadForm(request.POST, request.FILES)
        if form.is_valid():
            paper = form.save(commit=False)
            paper.user = request.user
            if paper.file:
                paper.file_size = paper.file.size
            
            paper.save()
            
            messages.success(request, f'Paper "{paper.title}" uploaded successfully! You can now ask questions about it.')
            return redirect('papers:qa', paper_id=paper.id)  # Redirect directly to Q&A
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = SimpleUploadForm()
    
    return render(request, 'papers/upload_enhanced.html', {'form': form})

@login_required
def paper_list(request):
    """Display user's uploaded papers with filtering and pagination"""
    papers = Paper.objects.filter(user=request.user)
    
    # Get statistics for all user papers
    all_papers = Paper.objects.filter(user=request.user)
    completed_count = all_papers.filter(analysis_status='completed').count()
    processing_count = all_papers.filter(analysis_status='processing').count()
    pending_count = all_papers.filter(analysis_status='pending').count()
    
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
    
    # Pagination - increased to 12 for better grid layout
    paginator = Paginator(papers, 12)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'search_query': search_query,
        'status_filter': status_filter,
        'total_papers': papers.count(),
        'completed_count': completed_count,
        'processing_count': processing_count,
        'pending_count': pending_count,
    }
    return render(request, 'papers/list_enhanced.html', context)

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


# RAG Q&A Views
@login_required
def paper_qa(request, paper_id):
    """Main Q&A interface for a paper"""
    paper = get_object_or_404(Paper, id=paper_id, user=request.user)
    
    # Get or create Q&A session
    qa_session, created = QASession.objects.get_or_create(
        paper=paper,
        user=request.user,
        defaults={'session_name': f"Q&A for {paper.title[:50]}"}
    )
    
    # Get previous questions in this session
    questions = qa_session.questions.all()
    
    # Get RAG service for paper processing status
    rag_service = RAGService()
    is_processed = bool(rag_service.load_vectorstore(paper.id))
    
    # Get question templates organized by category
    templates_by_category = {}
    templates = QuestionTemplate.objects.filter(is_active=True)
    for template in templates:
        if template.category not in templates_by_category:
            templates_by_category[template.category] = []
        templates_by_category[template.category].append(template)
    
    # Get paper tags
    available_tags = PaperTag.objects.filter(created_by=request.user)
    paper_tags = paper.tags.all()
    
    # Get research notes for this paper
    notes = ResearchNote.objects.filter(paper=paper, user=request.user)
    
    context = {
        'paper': paper,
        'qa_session': qa_session,
        'questions': questions,
        'is_processed': is_processed,
        'templates_by_category': templates_by_category,
        'available_tags': available_tags,
        'paper_tags': paper_tags,
        'notes': notes,
    }
    
    return render(request, 'papers/qa_interface_v2.html', context)


@login_required
@require_http_methods(["POST"])
def ask_question(request, paper_id):
    """Handle question submission and return answer"""
    paper = get_object_or_404(Paper, id=paper_id, user=request.user)
    
    question = request.POST.get('question', '').strip()
    if not question:
        return JsonResponse({'error': 'Question cannot be empty'}, status=400)
    
    try:
        # Get or create Q&A session
        qa_session, created = QASession.objects.get_or_create(
            paper=paper,
            user=request.user,
            defaults={'session_name': f"Q&A for {paper.title[:50]}"}
        )
        
        # Initialize RAG service
        rag_service = RAGService()
        
        # Get answer using RAG
        start_time = timezone.now()
        result = rag_service.answer_question(paper, question)
        processing_time = (timezone.now() - start_time).total_seconds()
        
        # Save question and answer
        paper_question = PaperQuestion.objects.create(
            qa_session=qa_session,
            question=question,
            answer=result['answer'],
            sources=result['sources'],
            model_used=result['model_used'],
            processing_time=processing_time
        )
        
        # Generate follow-up questions
        followup_questions = rag_service.generate_followup_questions(
            paper, question, result['answer']
        )
        
        return JsonResponse({
            'success': True,
            'question_id': str(paper_question.id),
            'answer': result['answer'],
            'sources': result['sources'],
            'processing_time': processing_time,
            'timestamp': paper_question.created_at.isoformat(),
            'followup_questions': followup_questions
        })
        
    except Exception as e:
        logger.error(f"Error processing question for paper {paper_id}: {str(e)}")
        return JsonResponse({
            'error': f'Error processing question: {str(e)}'
        }, status=500)


@login_required
def process_paper(request, paper_id):
    """Process paper for RAG (extract text and create embeddings)"""
    paper = get_object_or_404(Paper, id=paper_id, user=request.user)
    
    try:
        rag_service = RAGService()
        result = rag_service.process_paper(paper)
        
        messages.success(request, result)
        return JsonResponse({'success': True, 'message': result})
        
    except Exception as e:
        error_msg = f'Error processing paper: {str(e)}'
        logger.error(f"Error processing paper {paper_id}: {str(e)}")
        return JsonResponse({'error': error_msg}, status=500)


@login_required
def get_question_suggestions(request, paper_id):
    """Get suggested questions for a paper"""
    paper = get_object_or_404(Paper, id=paper_id, user=request.user)
    
    try:
        rag_service = RAGService()
        suggestions = rag_service.suggest_questions(paper)
        
        return JsonResponse({
            'success': True,
            'suggestions': suggestions
        })
        
    except Exception as e:
        logger.error(f"Error getting question suggestions for paper {paper_id}: {str(e)}")
        return JsonResponse({
            'error': f'Error getting suggestions: {str(e)}'
        }, status=500)


@login_required
def get_paper_summary(request, paper_id):
    """Get AI-generated summary of the paper"""
    paper = get_object_or_404(Paper, id=paper_id, user=request.user)
    
    try:
        rag_service = RAGService()
        summary_result = rag_service.get_paper_summary(paper)
        
        return JsonResponse({
            'success': True,
            'summary': summary_result['summary'],
            'generated_at': summary_result['generated_at'],
            'model_used': summary_result['model_used']
        })
        
    except Exception as e:
        logger.error(f"Error generating summary for paper {paper_id}: {str(e)}")
        return JsonResponse({
            'error': f'Error generating summary: {str(e)}'
        }, status=500)


@login_required
def qa_session_history(request, paper_id):
    """Get Q&A session history for a paper"""
    paper = get_object_or_404(Paper, id=paper_id, user=request.user)
    
    try:
        qa_session = QASession.objects.get(paper=paper, user=request.user)
        questions = qa_session.questions.all().values(
            'id', 'question', 'answer', 'created_at', 'processing_time'
        )
        
        return JsonResponse({
            'success': True,
            'session_id': str(qa_session.id),
            'questions': list(questions)
        })
        
    except QASession.DoesNotExist:
        return JsonResponse({
            'success': True,
            'session_id': None,
            'questions': []
        })
    except Exception as e:
        logger.error(f"Error getting Q&A history for paper {paper_id}: {str(e)}")
        return JsonResponse({
            'error': f'Error getting history: {str(e)}'
        }, status=500)


# Tag Management Views
@login_required
@require_http_methods(["POST"])
def create_tag(request):
    """Create a new paper tag"""
    try:
        name = request.POST.get('name', '').strip()
        color = request.POST.get('color', '#007bff')
        description = request.POST.get('description', '').strip()
        
        if not name:
            return JsonResponse({'error': 'Tag name is required'}, status=400)
        
        tag, created = PaperTag.objects.get_or_create(
            name=name,
            created_by=request.user,
            defaults={'color': color, 'description': description}
        )
        
        if not created:
            return JsonResponse({'error': 'Tag already exists'}, status=400)
        
        return JsonResponse({
            'success': True,
            'tag': {
                'id': str(tag.id),
                'name': tag.name,
                'color': tag.color,
                'description': tag.description
            }
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


# Export Views
@login_required
def export_qa_session(request, paper_id):
    """Export Q&A session as PDF"""
    paper = get_object_or_404(Paper, id=paper_id, user=request.user)
    
    try:
        qa_session = QASession.objects.get(paper=paper, user=request.user)
        questions = qa_session.questions.all()
        
        if not questions:
            return JsonResponse({'error': 'No Q&A session found'}, status=404)
        
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from io import BytesIO
        
        # Create PDF buffer
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=18)
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.darkblue,
            spaceAfter=30,
        )
        
        question_style = ParagraphStyle(
            'Question',
            parent=styles['Normal'],
            fontSize=12,
            textColor=colors.blue,
            fontName='Helvetica-Bold',
            spaceBefore=12,
            spaceAfter=6,
        )
        
        answer_style = ParagraphStyle(
            'Answer',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.black,
            spaceAfter=12,
            leftIndent=20,
        )
        
        # Build story
        story = []
        
        # Title
        story.append(Paragraph(f"Q&A Session: {paper.title}", title_style))
        story.append(Spacer(1, 12))
        
        # Paper info
        paper_info = f"<b>Authors:</b> {paper.authors or 'Not specified'}<br/>"
        paper_info += f"<b>Generated:</b> {timezone.now().strftime('%B %d, %Y at %I:%M %p')}<br/>"
        paper_info += f"<b>Total Questions:</b> {questions.count()}"
        story.append(Paragraph(paper_info, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Questions and answers
        for i, q in enumerate(questions, 1):
            story.append(Paragraph(f"Q{i}: {q.question}", question_style))
            story.append(Paragraph(q.answer, answer_style))
            
            if q.sources:
                sources_text = "<b>Sources:</b><br/>"
                for source in q.sources:
                    sources_text += f"• Chunk {source.get('chunk_id', 'N/A')}: {source.get('content_preview', '')[:100]}...<br/>"
                story.append(Paragraph(sources_text, styles['Normal']))
            
            story.append(Spacer(1, 12))
        
        # Build PDF
        doc.build(story)
        
        # Get PDF data
        buffer.seek(0)
        pdf_data = buffer.getvalue()
        buffer.close()
        
        # Return PDF response
        response = HttpResponse(pdf_data, content_type='application/pdf')
        filename = f"qa_session_{paper.title[:30]}.pdf".replace(' ', '_')
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        
        return response
        
    except QASession.DoesNotExist:
        return JsonResponse({'error': 'No Q&A session found'}, status=404)
    except Exception as e:
        logger.error(f"Error exporting Q&A session for paper {paper_id}: {str(e)}")
        return JsonResponse({'error': f'Export failed: {str(e)}'}, status=500)


# Comparative Analysis Views
@login_required
def comparative_analysis(request):
    """Main comparative analysis interface"""
    user_papers = Paper.objects.filter(user=request.user).order_by('-created_at')
    
    context = {
        'papers': user_papers,
        'total_papers': user_papers.count()
    }
    
    return render(request, 'papers/comparative_analysis.html', context)


@login_required
@require_http_methods(["POST"])
def compare_papers(request):
    """Compare selected papers"""
    try:
        paper_ids = request.POST.getlist('paper_ids')
        comparison_aspects = request.POST.getlist('aspects')
        
        if len(paper_ids) < 2:
            return JsonResponse({'error': 'Please select at least 2 papers to compare'}, status=400)
        
        # Get papers
        papers = Paper.objects.filter(id__in=paper_ids, user=request.user)
        
        if papers.count() != len(paper_ids):
            return JsonResponse({'error': 'Some selected papers were not found'}, status=404)
        
        # Perform comparison
        comparison_service = ComparativeAnalysisService()
        result = comparison_service.compare_papers(
            list(papers), 
            comparison_aspects if comparison_aspects else None
        )
        
        return JsonResponse({
            'success': True,
            'comparison': result
        })
        
    except Exception as e:
        logger.error(f"Error in comparative analysis: {str(e)}")
        return JsonResponse({'error': f'Comparison failed: {str(e)}'}, status=500)


@login_required
@require_http_methods(["POST"])
def generate_literature_review(request):
    """Generate literature review from selected papers"""
    try:
        paper_ids = request.POST.getlist('paper_ids')
        topic = request.POST.get('topic', '').strip()
        
        if not paper_ids:
            return JsonResponse({'error': 'Please select papers for literature review'}, status=400)
        
        # Get papers
        papers = Paper.objects.filter(id__in=paper_ids, user=request.user)
        
        if papers.count() == 0:
            return JsonResponse({'error': 'No valid papers selected'}, status=404)
        
        # Generate literature review
        comparison_service = ComparativeAnalysisService()
        result = comparison_service.generate_literature_review(
            list(papers), 
            topic if topic else None
        )
        
        return JsonResponse({
            'success': True,
            'literature_review': result
        })
        
    except Exception as e:
        logger.error(f"Error generating literature review: {str(e)}")
        return JsonResponse({'error': f'Literature review generation failed: {str(e)}'}, status=500)


@login_required
@require_http_methods(["POST"])
def add_paper_tag(request, paper_id):
    """Add tag to paper"""
    paper = get_object_or_404(Paper, id=paper_id, user=request.user)
    
    try:
        tag_id = request.POST.get('tag_id')
        tag = get_object_or_404(PaperTag, id=tag_id, created_by=request.user)
        
        paper.tags.add(tag)
        
        return JsonResponse({
            'success': True,
            'message': f'Tag "{tag.name}" added to paper'
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


# Export Views
@login_required
def export_qa_session(request, paper_id):
    """Export Q&A session as PDF"""
    paper = get_object_or_404(Paper, id=paper_id, user=request.user)
    
    try:
        qa_session = QASession.objects.get(paper=paper, user=request.user)
        questions = qa_session.questions.all()
        
        if not questions:
            return JsonResponse({'error': 'No Q&A session found'}, status=404)
        
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from io import BytesIO
        
        # Create PDF buffer
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=18)
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.darkblue,
            spaceAfter=30,
        )
        
        question_style = ParagraphStyle(
            'Question',
            parent=styles['Normal'],
            fontSize=12,
            textColor=colors.blue,
            fontName='Helvetica-Bold',
            spaceBefore=12,
            spaceAfter=6,
        )
        
        answer_style = ParagraphStyle(
            'Answer',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.black,
            spaceAfter=12,
            leftIndent=20,
        )
        
        # Build story
        story = []
        
        # Title
        story.append(Paragraph(f"Q&A Session: {paper.title}", title_style))
        story.append(Spacer(1, 12))
        
        # Paper info
        paper_info = f"<b>Authors:</b> {paper.authors or 'Not specified'}<br/>"
        paper_info += f"<b>Generated:</b> {timezone.now().strftime('%B %d, %Y at %I:%M %p')}<br/>"
        paper_info += f"<b>Total Questions:</b> {questions.count()}"
        story.append(Paragraph(paper_info, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Questions and answers
        for i, q in enumerate(questions, 1):
            story.append(Paragraph(f"Q{i}: {q.question}", question_style))
            story.append(Paragraph(q.answer, answer_style))
            
            if q.sources:
                sources_text = "<b>Sources:</b><br/>"
                for source in q.sources:
                    sources_text += f"• Chunk {source.get('chunk_id', 'N/A')}: {source.get('content_preview', '')[:100]}...<br/>"
                story.append(Paragraph(sources_text, styles['Normal']))
            
            story.append(Spacer(1, 12))
        
        # Build PDF
        doc.build(story)
        
        # Get PDF data
        buffer.seek(0)
        pdf_data = buffer.getvalue()
        buffer.close()
        
        # Return PDF response
        response = HttpResponse(pdf_data, content_type='application/pdf')
        filename = f"qa_session_{paper.title[:30]}.pdf".replace(' ', '_')
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        
        return response
        
    except QASession.DoesNotExist:
        return JsonResponse({'error': 'No Q&A session found'}, status=404)
    except Exception as e:
        logger.error(f"Error exporting Q&A session for paper {paper_id}: {str(e)}")
        return JsonResponse({'error': f'Export failed: {str(e)}'}, status=500)


# Comparative Analysis Views
@login_required
def comparative_analysis(request):
    """Main comparative analysis interface"""
    user_papers = Paper.objects.filter(user=request.user).order_by('-created_at')
    
    context = {
        'papers': user_papers,
        'total_papers': user_papers.count()
    }
    
    return render(request, 'papers/comparative_analysis.html', context)


@login_required
@require_http_methods(["POST"])
def compare_papers(request):
    """Compare selected papers"""
    try:
        paper_ids = request.POST.getlist('paper_ids')
        comparison_aspects = request.POST.getlist('aspects')
        
        if len(paper_ids) < 2:
            return JsonResponse({'error': 'Please select at least 2 papers to compare'}, status=400)
        
        # Get papers
        papers = Paper.objects.filter(id__in=paper_ids, user=request.user)
        
        if papers.count() != len(paper_ids):
            return JsonResponse({'error': 'Some selected papers were not found'}, status=404)
        
        # Perform comparison
        comparison_service = ComparativeAnalysisService()
        result = comparison_service.compare_papers(
            list(papers), 
            comparison_aspects if comparison_aspects else None
        )
        
        return JsonResponse({
            'success': True,
            'comparison': result
        })
        
    except Exception as e:
        logger.error(f"Error in comparative analysis: {str(e)}")
        return JsonResponse({'error': f'Comparison failed: {str(e)}'}, status=500)


@login_required
@require_http_methods(["POST"])
def generate_literature_review(request):
    """Generate literature review from selected papers"""
    try:
        paper_ids = request.POST.getlist('paper_ids')
        topic = request.POST.get('topic', '').strip()
        
        if not paper_ids:
            return JsonResponse({'error': 'Please select papers for literature review'}, status=400)
        
        # Get papers
        papers = Paper.objects.filter(id__in=paper_ids, user=request.user)
        
        if papers.count() == 0:
            return JsonResponse({'error': 'No valid papers selected'}, status=404)
        
        # Generate literature review
        comparison_service = ComparativeAnalysisService()
        result = comparison_service.generate_literature_review(
            list(papers), 
            topic if topic else None
        )
        
        return JsonResponse({
            'success': True,
            'literature_review': result
        })
        
    except Exception as e:
        logger.error(f"Error generating literature review: {str(e)}")
        return JsonResponse({'error': f'Literature review generation failed: {str(e)}'}, status=500)


@login_required
@require_http_methods(["POST"])
def remove_paper_tag(request, paper_id):
    """Remove tag from paper"""
    paper = get_object_or_404(Paper, id=paper_id, user=request.user)
    
    try:
        tag_id = request.POST.get('tag_id')
        tag = get_object_or_404(PaperTag, id=tag_id, created_by=request.user)
        
        paper.tags.remove(tag)
        
        return JsonResponse({
            'success': True,
            'message': f'Tag "{tag.name}" removed from paper'
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


# Export Views
@login_required
def export_qa_session(request, paper_id):
    """Export Q&A session as PDF"""
    paper = get_object_or_404(Paper, id=paper_id, user=request.user)
    
    try:
        qa_session = QASession.objects.get(paper=paper, user=request.user)
        questions = qa_session.questions.all()
        
        if not questions:
            return JsonResponse({'error': 'No Q&A session found'}, status=404)
        
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from io import BytesIO
        
        # Create PDF buffer
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=18)
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.darkblue,
            spaceAfter=30,
        )
        
        question_style = ParagraphStyle(
            'Question',
            parent=styles['Normal'],
            fontSize=12,
            textColor=colors.blue,
            fontName='Helvetica-Bold',
            spaceBefore=12,
            spaceAfter=6,
        )
        
        answer_style = ParagraphStyle(
            'Answer',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.black,
            spaceAfter=12,
            leftIndent=20,
        )
        
        # Build story
        story = []
        
        # Title
        story.append(Paragraph(f"Q&A Session: {paper.title}", title_style))
        story.append(Spacer(1, 12))
        
        # Paper info
        paper_info = f"<b>Authors:</b> {paper.authors or 'Not specified'}<br/>"
        paper_info += f"<b>Generated:</b> {timezone.now().strftime('%B %d, %Y at %I:%M %p')}<br/>"
        paper_info += f"<b>Total Questions:</b> {questions.count()}"
        story.append(Paragraph(paper_info, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Questions and answers
        for i, q in enumerate(questions, 1):
            story.append(Paragraph(f"Q{i}: {q.question}", question_style))
            story.append(Paragraph(q.answer, answer_style))
            
            if q.sources:
                sources_text = "<b>Sources:</b><br/>"
                for source in q.sources:
                    sources_text += f"• Chunk {source.get('chunk_id', 'N/A')}: {source.get('content_preview', '')[:100]}...<br/>"
                story.append(Paragraph(sources_text, styles['Normal']))
            
            story.append(Spacer(1, 12))
        
        # Build PDF
        doc.build(story)
        
        # Get PDF data
        buffer.seek(0)
        pdf_data = buffer.getvalue()
        buffer.close()
        
        # Return PDF response
        response = HttpResponse(pdf_data, content_type='application/pdf')
        filename = f"qa_session_{paper.title[:30]}.pdf".replace(' ', '_')
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        
        return response
        
    except QASession.DoesNotExist:
        return JsonResponse({'error': 'No Q&A session found'}, status=404)
    except Exception as e:
        logger.error(f"Error exporting Q&A session for paper {paper_id}: {str(e)}")
        return JsonResponse({'error': f'Export failed: {str(e)}'}, status=500)


# Comparative Analysis Views
@login_required
def comparative_analysis(request):
    """Main comparative analysis interface"""
    user_papers = Paper.objects.filter(user=request.user).order_by('-created_at')
    
    context = {
        'papers': user_papers,
        'total_papers': user_papers.count()
    }
    
    return render(request, 'papers/comparative_analysis.html', context)


@login_required
@require_http_methods(["POST"])
def compare_papers(request):
    """Compare selected papers"""
    try:
        paper_ids = request.POST.getlist('paper_ids')
        comparison_aspects = request.POST.getlist('aspects')
        
        if len(paper_ids) < 2:
            return JsonResponse({'error': 'Please select at least 2 papers to compare'}, status=400)
        
        # Get papers
        papers = Paper.objects.filter(id__in=paper_ids, user=request.user)
        
        if papers.count() != len(paper_ids):
            return JsonResponse({'error': 'Some selected papers were not found'}, status=404)
        
        # Perform comparison
        comparison_service = ComparativeAnalysisService()
        result = comparison_service.compare_papers(
            list(papers), 
            comparison_aspects if comparison_aspects else None
        )
        
        return JsonResponse({
            'success': True,
            'comparison': result
        })
        
    except Exception as e:
        logger.error(f"Error in comparative analysis: {str(e)}")
        return JsonResponse({'error': f'Comparison failed: {str(e)}'}, status=500)


@login_required
@require_http_methods(["POST"])
def generate_literature_review(request):
    """Generate literature review from selected papers"""
    try:
        paper_ids = request.POST.getlist('paper_ids')
        topic = request.POST.get('topic', '').strip()
        
        if not paper_ids:
            return JsonResponse({'error': 'Please select papers for literature review'}, status=400)
        
        # Get papers
        papers = Paper.objects.filter(id__in=paper_ids, user=request.user)
        
        if papers.count() == 0:
            return JsonResponse({'error': 'No valid papers selected'}, status=404)
        
        # Generate literature review
        comparison_service = ComparativeAnalysisService()
        result = comparison_service.generate_literature_review(
            list(papers), 
            topic if topic else None
        )
        
        return JsonResponse({
            'success': True,
            'literature_review': result
        })
        
    except Exception as e:
        logger.error(f"Error generating literature review: {str(e)}")
        return JsonResponse({'error': f'Literature review generation failed: {str(e)}'}, status=500)


# Research Notes Views
@login_required
@require_http_methods(["POST"])
def create_note(request, paper_id):
    """Create a research note for a paper"""
    paper = get_object_or_404(Paper, id=paper_id, user=request.user)
    
    try:
        title = request.POST.get('title', '').strip()
        content = request.POST.get('content', '').strip()
        
        if not title or not content:
            return JsonResponse({'error': 'Title and content are required'}, status=400)
        
        note = ResearchNote.objects.create(
            paper=paper,
            user=request.user,
            title=title,
            content=content
        )
        
        return JsonResponse({
            'success': True,
            'note': {
                'id': str(note.id),
                'title': note.title,
                'content': note.content,
                'created_at': note.created_at.isoformat()
            }
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


# Export Views
@login_required
def export_qa_session(request, paper_id):
    """Export Q&A session as PDF"""
    paper = get_object_or_404(Paper, id=paper_id, user=request.user)
    
    try:
        qa_session = QASession.objects.get(paper=paper, user=request.user)
        questions = qa_session.questions.all()
        
        if not questions:
            return JsonResponse({'error': 'No Q&A session found'}, status=404)
        
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from io import BytesIO
        
        # Create PDF buffer
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=18)
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.darkblue,
            spaceAfter=30,
        )
        
        question_style = ParagraphStyle(
            'Question',
            parent=styles['Normal'],
            fontSize=12,
            textColor=colors.blue,
            fontName='Helvetica-Bold',
            spaceBefore=12,
            spaceAfter=6,
        )
        
        answer_style = ParagraphStyle(
            'Answer',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.black,
            spaceAfter=12,
            leftIndent=20,
        )
        
        # Build story
        story = []
        
        # Title
        story.append(Paragraph(f"Q&A Session: {paper.title}", title_style))
        story.append(Spacer(1, 12))
        
        # Paper info
        paper_info = f"<b>Authors:</b> {paper.authors or 'Not specified'}<br/>"
        paper_info += f"<b>Generated:</b> {timezone.now().strftime('%B %d, %Y at %I:%M %p')}<br/>"
        paper_info += f"<b>Total Questions:</b> {questions.count()}"
        story.append(Paragraph(paper_info, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Questions and answers
        for i, q in enumerate(questions, 1):
            story.append(Paragraph(f"Q{i}: {q.question}", question_style))
            story.append(Paragraph(q.answer, answer_style))
            
            if q.sources:
                sources_text = "<b>Sources:</b><br/>"
                for source in q.sources:
                    sources_text += f"• Chunk {source.get('chunk_id', 'N/A')}: {source.get('content_preview', '')[:100]}...<br/>"
                story.append(Paragraph(sources_text, styles['Normal']))
            
            story.append(Spacer(1, 12))
        
        # Build PDF
        doc.build(story)
        
        # Get PDF data
        buffer.seek(0)
        pdf_data = buffer.getvalue()
        buffer.close()
        
        # Return PDF response
        response = HttpResponse(pdf_data, content_type='application/pdf')
        filename = f"qa_session_{paper.title[:30]}.pdf".replace(' ', '_')
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        
        return response
        
    except QASession.DoesNotExist:
        return JsonResponse({'error': 'No Q&A session found'}, status=404)
    except Exception as e:
        logger.error(f"Error exporting Q&A session for paper {paper_id}: {str(e)}")
        return JsonResponse({'error': f'Export failed: {str(e)}'}, status=500)


# Comparative Analysis Views
@login_required
def comparative_analysis(request):
    """Main comparative analysis interface"""
    user_papers = Paper.objects.filter(user=request.user).order_by('-created_at')
    
    context = {
        'papers': user_papers,
        'total_papers': user_papers.count()
    }
    
    return render(request, 'papers/comparative_analysis.html', context)


@login_required
@require_http_methods(["POST"])
def compare_papers(request):
    """Compare selected papers"""
    try:
        paper_ids = request.POST.getlist('paper_ids')
        comparison_aspects = request.POST.getlist('aspects')
        
        if len(paper_ids) < 2:
            return JsonResponse({'error': 'Please select at least 2 papers to compare'}, status=400)
        
        # Get papers
        papers = Paper.objects.filter(id__in=paper_ids, user=request.user)
        
        if papers.count() != len(paper_ids):
            return JsonResponse({'error': 'Some selected papers were not found'}, status=404)
        
        # Perform comparison
        comparison_service = ComparativeAnalysisService()
        result = comparison_service.compare_papers(
            list(papers), 
            comparison_aspects if comparison_aspects else None
        )
        
        return JsonResponse({
            'success': True,
            'comparison': result
        })
        
    except Exception as e:
        logger.error(f"Error in comparative analysis: {str(e)}")
        return JsonResponse({'error': f'Comparison failed: {str(e)}'}, status=500)


@login_required
@require_http_methods(["POST"])
def generate_literature_review(request):
    """Generate literature review from selected papers"""
    try:
        paper_ids = request.POST.getlist('paper_ids')
        topic = request.POST.get('topic', '').strip()
        
        if not paper_ids:
            return JsonResponse({'error': 'Please select papers for literature review'}, status=400)
        
        # Get papers
        papers = Paper.objects.filter(id__in=paper_ids, user=request.user)
        
        if papers.count() == 0:
            return JsonResponse({'error': 'No valid papers selected'}, status=404)
        
        # Generate literature review
        comparison_service = ComparativeAnalysisService()
        result = comparison_service.generate_literature_review(
            list(papers), 
            topic if topic else None
        )
        
        return JsonResponse({
            'success': True,
            'literature_review': result
        })
        
    except Exception as e:
        logger.error(f"Error generating literature review: {str(e)}")
        return JsonResponse({'error': f'Literature review generation failed: {str(e)}'}, status=500)


@login_required
@require_http_methods(["POST"])
def update_note(request, paper_id, note_id):
    """Update a research note"""
    paper = get_object_or_404(Paper, id=paper_id, user=request.user)
    note = get_object_or_404(ResearchNote, id=note_id, paper=paper, user=request.user)
    
    try:
        title = request.POST.get('title', '').strip()
        content = request.POST.get('content', '').strip()
        
        if title:
            note.title = title
        if content:
            note.content = content
        
        note.save()
        
        return JsonResponse({
            'success': True,
            'message': 'Note updated successfully'
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


# Export Views
@login_required
def export_qa_session(request, paper_id):
    """Export Q&A session as PDF"""
    paper = get_object_or_404(Paper, id=paper_id, user=request.user)
    
    try:
        qa_session = QASession.objects.get(paper=paper, user=request.user)
        questions = qa_session.questions.all()
        
        if not questions:
            return JsonResponse({'error': 'No Q&A session found'}, status=404)
        
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from io import BytesIO
        
        # Create PDF buffer
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=18)
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.darkblue,
            spaceAfter=30,
        )
        
        question_style = ParagraphStyle(
            'Question',
            parent=styles['Normal'],
            fontSize=12,
            textColor=colors.blue,
            fontName='Helvetica-Bold',
            spaceBefore=12,
            spaceAfter=6,
        )
        
        answer_style = ParagraphStyle(
            'Answer',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.black,
            spaceAfter=12,
            leftIndent=20,
        )
        
        # Build story
        story = []
        
        # Title
        story.append(Paragraph(f"Q&A Session: {paper.title}", title_style))
        story.append(Spacer(1, 12))
        
        # Paper info
        paper_info = f"<b>Authors:</b> {paper.authors or 'Not specified'}<br/>"
        paper_info += f"<b>Generated:</b> {timezone.now().strftime('%B %d, %Y at %I:%M %p')}<br/>"
        paper_info += f"<b>Total Questions:</b> {questions.count()}"
        story.append(Paragraph(paper_info, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Questions and answers
        for i, q in enumerate(questions, 1):
            story.append(Paragraph(f"Q{i}: {q.question}", question_style))
            story.append(Paragraph(q.answer, answer_style))
            
            if q.sources:
                sources_text = "<b>Sources:</b><br/>"
                for source in q.sources:
                    sources_text += f"• Chunk {source.get('chunk_id', 'N/A')}: {source.get('content_preview', '')[:100]}...<br/>"
                story.append(Paragraph(sources_text, styles['Normal']))
            
            story.append(Spacer(1, 12))
        
        # Build PDF
        doc.build(story)
        
        # Get PDF data
        buffer.seek(0)
        pdf_data = buffer.getvalue()
        buffer.close()
        
        # Return PDF response
        response = HttpResponse(pdf_data, content_type='application/pdf')
        filename = f"qa_session_{paper.title[:30]}.pdf".replace(' ', '_')
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        
        return response
        
    except QASession.DoesNotExist:
        return JsonResponse({'error': 'No Q&A session found'}, status=404)
    except Exception as e:
        logger.error(f"Error exporting Q&A session for paper {paper_id}: {str(e)}")
        return JsonResponse({'error': f'Export failed: {str(e)}'}, status=500)


# Comparative Analysis Views
@login_required
def comparative_analysis(request):
    """Main comparative analysis interface"""
    user_papers = Paper.objects.filter(user=request.user).order_by('-created_at')
    
    context = {
        'papers': user_papers,
        'total_papers': user_papers.count()
    }
    
    return render(request, 'papers/comparative_analysis.html', context)


@login_required
@require_http_methods(["POST"])
def compare_papers(request):
    """Compare selected papers"""
    try:
        paper_ids = request.POST.getlist('paper_ids')
        comparison_aspects = request.POST.getlist('aspects')
        
        if len(paper_ids) < 2:
            return JsonResponse({'error': 'Please select at least 2 papers to compare'}, status=400)
        
        # Get papers
        papers = Paper.objects.filter(id__in=paper_ids, user=request.user)
        
        if papers.count() != len(paper_ids):
            return JsonResponse({'error': 'Some selected papers were not found'}, status=404)
        
        # Perform comparison
        comparison_service = ComparativeAnalysisService()
        result = comparison_service.compare_papers(
            list(papers), 
            comparison_aspects if comparison_aspects else None
        )
        
        return JsonResponse({
            'success': True,
            'comparison': result
        })
        
    except Exception as e:
        logger.error(f"Error in comparative analysis: {str(e)}")
        return JsonResponse({'error': f'Comparison failed: {str(e)}'}, status=500)


@login_required
@require_http_methods(["POST"])
def generate_literature_review(request):
    """Generate literature review from selected papers"""
    try:
        paper_ids = request.POST.getlist('paper_ids')
        topic = request.POST.get('topic', '').strip()
        
        if not paper_ids:
            return JsonResponse({'error': 'Please select papers for literature review'}, status=400)
        
        # Get papers
        papers = Paper.objects.filter(id__in=paper_ids, user=request.user)
        
        if papers.count() == 0:
            return JsonResponse({'error': 'No valid papers selected'}, status=404)
        
        # Generate literature review
        comparison_service = ComparativeAnalysisService()
        result = comparison_service.generate_literature_review(
            list(papers), 
            topic if topic else None
        )
        
        return JsonResponse({
            'success': True,
            'literature_review': result
        })
        
    except Exception as e:
        logger.error(f"Error generating literature review: {str(e)}")
        return JsonResponse({'error': f'Literature review generation failed: {str(e)}'}, status=500)


@login_required
@require_http_methods(["POST"])
def delete_note(request, paper_id, note_id):
    """Delete a research note"""
    paper = get_object_or_404(Paper, id=paper_id, user=request.user)
    note = get_object_or_404(ResearchNote, id=note_id, paper=paper, user=request.user)
    
    try:
        note.delete()
        return JsonResponse({
            'success': True,
            'message': 'Note deleted successfully'
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


# Export Views
@login_required
def export_qa_session(request, paper_id):
    """Export Q&A session as PDF"""
    paper = get_object_or_404(Paper, id=paper_id, user=request.user)
    
    try:
        qa_session = QASession.objects.get(paper=paper, user=request.user)
        questions = qa_session.questions.all()
        
        if not questions:
            return JsonResponse({'error': 'No Q&A session found'}, status=404)
        
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from io import BytesIO
        
        # Create PDF buffer
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=18)
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.darkblue,
            spaceAfter=30,
        )
        
        question_style = ParagraphStyle(
            'Question',
            parent=styles['Normal'],
            fontSize=12,
            textColor=colors.blue,
            fontName='Helvetica-Bold',
            spaceBefore=12,
            spaceAfter=6,
        )
        
        answer_style = ParagraphStyle(
            'Answer',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.black,
            spaceAfter=12,
            leftIndent=20,
        )
        
        # Build story
        story = []
        
        # Title
        story.append(Paragraph(f"Q&A Session: {paper.title}", title_style))
        story.append(Spacer(1, 12))
        
        # Paper info
        paper_info = f"<b>Authors:</b> {paper.authors or 'Not specified'}<br/>"
        paper_info += f"<b>Generated:</b> {timezone.now().strftime('%B %d, %Y at %I:%M %p')}<br/>"
        paper_info += f"<b>Total Questions:</b> {questions.count()}"
        story.append(Paragraph(paper_info, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Questions and answers
        for i, q in enumerate(questions, 1):
            story.append(Paragraph(f"Q{i}: {q.question}", question_style))
            story.append(Paragraph(q.answer, answer_style))
            
            if q.sources:
                sources_text = "<b>Sources:</b><br/>"
                for source in q.sources:
                    sources_text += f"• Chunk {source.get('chunk_id', 'N/A')}: {source.get('content_preview', '')[:100]}...<br/>"
                story.append(Paragraph(sources_text, styles['Normal']))
            
            story.append(Spacer(1, 12))
        
        # Build PDF
        doc.build(story)
        
        # Get PDF data
        buffer.seek(0)
        pdf_data = buffer.getvalue()
        buffer.close()
        
        # Return PDF response
        response = HttpResponse(pdf_data, content_type='application/pdf')
        filename = f"qa_session_{paper.title[:30]}.pdf".replace(' ', '_')
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        
        return response
        
    except QASession.DoesNotExist:
        return JsonResponse({'error': 'No Q&A session found'}, status=404)
    except Exception as e:
        logger.error(f"Error exporting Q&A session for paper {paper_id}: {str(e)}")
        return JsonResponse({'error': f'Export failed: {str(e)}'}, status=500)


# Comparative Analysis Views
@login_required
def comparative_analysis(request):
    """Main comparative analysis interface"""
    user_papers = Paper.objects.filter(user=request.user).order_by('-created_at')
    
    context = {
        'papers': user_papers,
        'total_papers': user_papers.count()
    }
    
    return render(request, 'papers/comparative_analysis.html', context)


@login_required
@require_http_methods(["POST"])
def compare_papers(request):
    """Compare selected papers"""
    try:
        paper_ids = request.POST.getlist('paper_ids')
        comparison_aspects = request.POST.getlist('aspects')
        
        if len(paper_ids) < 2:
            return JsonResponse({'error': 'Please select at least 2 papers to compare'}, status=400)
        
        # Get papers
        papers = Paper.objects.filter(id__in=paper_ids, user=request.user)
        
        if papers.count() != len(paper_ids):
            return JsonResponse({'error': 'Some selected papers were not found'}, status=404)
        
        # Perform comparison
        comparison_service = ComparativeAnalysisService()
        result = comparison_service.compare_papers(
            list(papers), 
            comparison_aspects if comparison_aspects else None
        )
        
        return JsonResponse({
            'success': True,
            'comparison': result
        })
        
    except Exception as e:
        logger.error(f"Error in comparative analysis: {str(e)}")
        return JsonResponse({'error': f'Comparison failed: {str(e)}'}, status=500)


@login_required
@require_http_methods(["POST"])
def generate_literature_review(request):
    """Generate literature review from selected papers"""
    try:
        paper_ids = request.POST.getlist('paper_ids')
        topic = request.POST.get('topic', '').strip()
        
        if not paper_ids:
            return JsonResponse({'error': 'Please select papers for literature review'}, status=400)
        
        # Get papers
        papers = Paper.objects.filter(id__in=paper_ids, user=request.user)
        
        if papers.count() == 0:
            return JsonResponse({'error': 'No valid papers selected'}, status=404)
        
        # Generate literature review
        comparison_service = ComparativeAnalysisService()
        result = comparison_service.generate_literature_review(
            list(papers), 
            topic if topic else None
        )
        
        return JsonResponse({
            'success': True,
            'literature_review': result
        })
        
    except Exception as e:
        logger.error(f"Error generating literature review: {str(e)}")
        return JsonResponse({'error': f'Literature review generation failed: {str(e)}'}, status=500)


# Question Templates Views
@login_required
def get_question_templates(request):
    """Get question templates organized by category"""
    try:
        category = request.GET.get('category')
        templates = QuestionTemplate.objects.filter(is_active=True)
        
        if category:
            templates = templates.filter(category=category)
        
        templates_data = {}
        for template in templates:
            if template.category not in templates_data:
                templates_data[template.category] = []
            templates_data[template.category].append({
                'id': str(template.id),
                'question_text': template.question_text,
                'description': template.description,
                'usage_count': template.usage_count
            })
        
        return JsonResponse({
            'success': True,
            'templates': templates_data
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


# Export Views
@login_required
def export_qa_session(request, paper_id):
    """Export Q&A session as PDF"""
    paper = get_object_or_404(Paper, id=paper_id, user=request.user)
    
    try:
        qa_session = QASession.objects.get(paper=paper, user=request.user)
        questions = qa_session.questions.all()
        
        if not questions:
            return JsonResponse({'error': 'No Q&A session found'}, status=404)
        
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from io import BytesIO
        
        # Create PDF buffer
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=18)
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.darkblue,
            spaceAfter=30,
        )
        
        question_style = ParagraphStyle(
            'Question',
            parent=styles['Normal'],
            fontSize=12,
            textColor=colors.blue,
            fontName='Helvetica-Bold',
            spaceBefore=12,
            spaceAfter=6,
        )
        
        answer_style = ParagraphStyle(
            'Answer',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.black,
            spaceAfter=12,
            leftIndent=20,
        )
        
        # Build story
        story = []
        
        # Title
        story.append(Paragraph(f"Q&A Session: {paper.title}", title_style))
        story.append(Spacer(1, 12))
        
        # Paper info
        paper_info = f"<b>Authors:</b> {paper.authors or 'Not specified'}<br/>"
        paper_info += f"<b>Generated:</b> {timezone.now().strftime('%B %d, %Y at %I:%M %p')}<br/>"
        paper_info += f"<b>Total Questions:</b> {questions.count()}"
        story.append(Paragraph(paper_info, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Questions and answers
        for i, q in enumerate(questions, 1):
            story.append(Paragraph(f"Q{i}: {q.question}", question_style))
            story.append(Paragraph(q.answer, answer_style))
            
            if q.sources:
                sources_text = "<b>Sources:</b><br/>"
                for source in q.sources:
                    sources_text += f"• Chunk {source.get('chunk_id', 'N/A')}: {source.get('content_preview', '')[:100]}...<br/>"
                story.append(Paragraph(sources_text, styles['Normal']))
            
            story.append(Spacer(1, 12))
        
        # Build PDF
        doc.build(story)
        
        # Get PDF data
        buffer.seek(0)
        pdf_data = buffer.getvalue()
        buffer.close()
        
        # Return PDF response
        response = HttpResponse(pdf_data, content_type='application/pdf')
        filename = f"qa_session_{paper.title[:30]}.pdf".replace(' ', '_')
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        
        return response
        
    except QASession.DoesNotExist:
        return JsonResponse({'error': 'No Q&A session found'}, status=404)
    except Exception as e:
        logger.error(f"Error exporting Q&A session for paper {paper_id}: {str(e)}")
        return JsonResponse({'error': f'Export failed: {str(e)}'}, status=500)


# Comparative Analysis Views
@login_required
def comparative_analysis(request):
    """Main comparative analysis interface"""
    user_papers = Paper.objects.filter(user=request.user).order_by('-created_at')
    
    context = {
        'papers': user_papers,
        'total_papers': user_papers.count()
    }
    
    return render(request, 'papers/comparative_analysis.html', context)


@login_required
@require_http_methods(["POST"])
def compare_papers(request):
    """Compare selected papers"""
    try:
        paper_ids = request.POST.getlist('paper_ids')
        comparison_aspects = request.POST.getlist('aspects')
        
        if len(paper_ids) < 2:
            return JsonResponse({'error': 'Please select at least 2 papers to compare'}, status=400)
        
        # Get papers
        papers = Paper.objects.filter(id__in=paper_ids, user=request.user)
        
        if papers.count() != len(paper_ids):
            return JsonResponse({'error': 'Some selected papers were not found'}, status=404)
        
        # Perform comparison
        comparison_service = ComparativeAnalysisService()
        result = comparison_service.compare_papers(
            list(papers), 
            comparison_aspects if comparison_aspects else None
        )
        
        return JsonResponse({
            'success': True,
            'comparison': result
        })
        
    except Exception as e:
        logger.error(f"Error in comparative analysis: {str(e)}")
        return JsonResponse({'error': f'Comparison failed: {str(e)}'}, status=500)


@login_required
@require_http_methods(["POST"])
def generate_literature_review(request):
    """Generate literature review from selected papers"""
    try:
        paper_ids = request.POST.getlist('paper_ids')
        topic = request.POST.get('topic', '').strip()
        
        if not paper_ids:
            return JsonResponse({'error': 'Please select papers for literature review'}, status=400)
        
        # Get papers
        papers = Paper.objects.filter(id__in=paper_ids, user=request.user)
        
        if papers.count() == 0:
            return JsonResponse({'error': 'No valid papers selected'}, status=404)
        
        # Generate literature review
        comparison_service = ComparativeAnalysisService()
        result = comparison_service.generate_literature_review(
            list(papers), 
            topic if topic else None
        )
        
        return JsonResponse({
            'success': True,
            'literature_review': result
        })
        
    except Exception as e:
        logger.error(f"Error generating literature review: {str(e)}")
        return JsonResponse({'error': f'Literature review generation failed: {str(e)}'}, status=500)


@login_required
@require_http_methods(["POST"])
def use_question_template(request, template_id):
    """Track usage of a question template and return the question"""
    try:
        template = get_object_or_404(QuestionTemplate, id=template_id, is_active=True)
        template.usage_count += 1
        template.save()
        
        return JsonResponse({
            'success': True,
            'question_text': template.question_text
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


# Export Views
@login_required
def export_qa_session(request, paper_id):
    """Export Q&A session as PDF"""
    paper = get_object_or_404(Paper, id=paper_id, user=request.user)
    
    try:
        qa_session = QASession.objects.get(paper=paper, user=request.user)
        questions = qa_session.questions.all()
        
        if not questions:
            return JsonResponse({'error': 'No Q&A session found'}, status=404)
        
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from io import BytesIO
        
        # Create PDF buffer
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=18)
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.darkblue,
            spaceAfter=30,
        )
        
        question_style = ParagraphStyle(
            'Question',
            parent=styles['Normal'],
            fontSize=12,
            textColor=colors.blue,
            fontName='Helvetica-Bold',
            spaceBefore=12,
            spaceAfter=6,
        )
        
        answer_style = ParagraphStyle(
            'Answer',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.black,
            spaceAfter=12,
            leftIndent=20,
        )
        
        # Build story
        story = []
        
        # Title
        story.append(Paragraph(f"Q&A Session: {paper.title}", title_style))
        story.append(Spacer(1, 12))
        
        # Paper info
        paper_info = f"<b>Authors:</b> {paper.authors or 'Not specified'}<br/>"
        paper_info += f"<b>Generated:</b> {timezone.now().strftime('%B %d, %Y at %I:%M %p')}<br/>"
        paper_info += f"<b>Total Questions:</b> {questions.count()}"
        story.append(Paragraph(paper_info, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Questions and answers
        for i, q in enumerate(questions, 1):
            story.append(Paragraph(f"Q{i}: {q.question}", question_style))
            story.append(Paragraph(q.answer, answer_style))
            
            if q.sources:
                sources_text = "<b>Sources:</b><br/>"
                for source in q.sources:
                    sources_text += f"• Chunk {source.get('chunk_id', 'N/A')}: {source.get('content_preview', '')[:100]}...<br/>"
                story.append(Paragraph(sources_text, styles['Normal']))
            
            story.append(Spacer(1, 12))
        
        # Build PDF
        doc.build(story)
        
        # Get PDF data
        buffer.seek(0)
        pdf_data = buffer.getvalue()
        buffer.close()
        
        # Return PDF response
        response = HttpResponse(pdf_data, content_type='application/pdf')
        filename = f"qa_session_{paper.title[:30]}.pdf".replace(' ', '_')
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        
        return response
        
    except QASession.DoesNotExist:
        return JsonResponse({'error': 'No Q&A session found'}, status=404)
    except Exception as e:
        logger.error(f"Error exporting Q&A session for paper {paper_id}: {str(e)}")
        return JsonResponse({'error': f'Export failed: {str(e)}'}, status=500)


# Comparative Analysis Views
@login_required
def comparative_analysis(request):
    """Main comparative analysis interface"""
    user_papers = Paper.objects.filter(user=request.user).order_by('-created_at')
    
    context = {
        'papers': user_papers,
        'total_papers': user_papers.count()
    }
    
    return render(request, 'papers/comparative_analysis.html', context)


@login_required
@require_http_methods(["POST"])
def compare_papers(request):
    """Compare selected papers"""
    try:
        paper_ids = request.POST.getlist('paper_ids')
        comparison_aspects = request.POST.getlist('aspects')
        
        if len(paper_ids) < 2:
            return JsonResponse({'error': 'Please select at least 2 papers to compare'}, status=400)
        
        # Get papers
        papers = Paper.objects.filter(id__in=paper_ids, user=request.user)
        
        if papers.count() != len(paper_ids):
            return JsonResponse({'error': 'Some selected papers were not found'}, status=404)
        
        # Perform comparison
        comparison_service = ComparativeAnalysisService()
        result = comparison_service.compare_papers(
            list(papers), 
            comparison_aspects if comparison_aspects else None
        )
        
        return JsonResponse({
            'success': True,
            'comparison': result
        })
        
    except Exception as e:
        logger.error(f"Error in comparative analysis: {str(e)}")
        return JsonResponse({'error': f'Comparison failed: {str(e)}'}, status=500)


@login_required
@require_http_methods(["POST"])
def generate_literature_review(request):
    """Generate literature review from selected papers"""
    try:
        paper_ids = request.POST.getlist('paper_ids')
        topic = request.POST.get('topic', '').strip()
        
        if not paper_ids:
            return JsonResponse({'error': 'Please select papers for literature review'}, status=400)
        
        # Get papers
        papers = Paper.objects.filter(id__in=paper_ids, user=request.user)
        
        if papers.count() == 0:
            return JsonResponse({'error': 'No valid papers selected'}, status=404)
        
        # Generate literature review
        comparison_service = ComparativeAnalysisService()
        result = comparison_service.generate_literature_review(
            list(papers), 
            topic if topic else None
        )
        
        return JsonResponse({
            'success': True,
            'literature_review': result
        })
        
    except Exception as e:
        logger.error(f"Error generating literature review: {str(e)}")
        return JsonResponse({'error': f'Literature review generation failed: {str(e)}'}, status=500)
