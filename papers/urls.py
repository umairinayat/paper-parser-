from django.urls import path
from . import views

app_name = 'papers'

urlpatterns = [
    path('upload/', views.paper_upload, name='upload'),
    path('list/', views.paper_list, name='list'),
    path('<int:pk>/', views.paper_detail, name='detail'),
    path('<int:pk>/edit/', views.paper_edit, name='edit'),
    path('<int:pk>/delete/', views.paper_delete, name='delete'),
    path('<int:pk>/start-analysis/', views.start_analysis, name='start_analysis'),
    path('<int:pk>/status/', views.analysis_status, name='analysis_status'),
    
    # New search and analysis URLs
    path('search/', views.paper_search, name='search'),
    path('search/results/', views.search_results, name='search_results'),
    path('search/export/csv/', views.export_analysis_csv, name='export_csv'),
] 