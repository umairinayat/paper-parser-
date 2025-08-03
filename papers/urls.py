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
    
    # Enhanced search and analysis URLs
    path('search/', views.paper_search, name='search'),
    path('search/results/', views.search_results, name='search_results'),
    path('search/suggestions/', views.search_suggestions, name='search_suggestions'),
    path('search/export/csv/', views.export_analysis_csv, name='export_csv'),
    path('search/export/json/', views.export_analysis_json, name='export_json'),
    path('bulk-analysis/', views.bulk_analysis, name='bulk_analysis'),
] 