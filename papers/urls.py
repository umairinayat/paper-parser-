from django.urls import path
from . import views

app_name = 'papers'

urlpatterns = [
    path('upload/', views.paper_upload, name='upload'),
    path('list/', views.paper_list, name='list'),
    path('<uuid:paper_id>/', views.paper_detail, name='detail'),
    path('<uuid:paper_id>/edit/', views.paper_edit, name='edit'),
    path('<uuid:paper_id>/delete/', views.paper_delete, name='delete'),
    path('<uuid:paper_id>/start-analysis/', views.start_analysis, name='start_analysis'),
    path('<uuid:paper_id>/status/', views.analysis_status, name='analysis_status'),
    path('<uuid:paper_id>/similar/', views.similar_papers, name='similar_papers'),
    
    # RAG Q&A URLs
    path('<uuid:paper_id>/qa/', views.paper_qa, name='qa'),
    path('<uuid:paper_id>/qa/ask/', views.ask_question, name='ask_question'),
    path('<uuid:paper_id>/qa/process/', views.process_paper, name='process_paper'),
    path('<uuid:paper_id>/qa/suggestions/', views.get_question_suggestions, name='question_suggestions'),
    path('<uuid:paper_id>/qa/summary/', views.get_paper_summary, name='paper_summary'),
    path('<uuid:paper_id>/qa/history/', views.qa_session_history, name='qa_history'),
    
    # Tag Management URLs
    path('tags/create/', views.create_tag, name='create_tag'),
    path('<uuid:paper_id>/tags/add/', views.add_paper_tag, name='add_paper_tag'),
    path('<uuid:paper_id>/tags/remove/', views.remove_paper_tag, name='remove_paper_tag'),
    
    # Research Notes URLs
    path('<uuid:paper_id>/notes/create/', views.create_note, name='create_note'),
    path('<uuid:paper_id>/notes/<uuid:note_id>/update/', views.update_note, name='update_note'),
    path('<uuid:paper_id>/notes/<uuid:note_id>/delete/', views.delete_note, name='delete_note'),
    
    # Question Templates URLs
    path('templates/', views.get_question_templates, name='question_templates'),
    path('templates/<uuid:template_id>/use/', views.use_question_template, name='use_template'),
    
    # Export URLs
    path('<uuid:paper_id>/export/qa/', views.export_qa_session, name='export_qa'),
    
    # Comparative Analysis URLs
    path('compare/', views.comparative_analysis, name='comparative_analysis'),
    path('compare/analyze/', views.compare_papers, name='compare_papers'),
    path('literature-review/', views.generate_literature_review, name='generate_literature_review'),
    
    # Enhanced search and analysis URLs
    path('search/', views.paper_search, name='search'),
    path('search/results/', views.search_results, name='search_results'),
    path('search/suggestions/', views.search_suggestions, name='search_suggestions'),
    path('search/export/csv/', views.export_analysis_csv, name='export_csv'),
    path('search/export/json/', views.export_analysis_json, name='export_json'),
    path('bulk-analysis/', views.bulk_analysis, name='bulk_analysis'),
] 