from django.urls import path
from . import views

app_name = 'analysis'

urlpatterns = [
    path('<int:paper_id>/results/', views.analysis_results, name='results'),
    path('<int:paper_id>/export/<str:format>/', views.export_analysis, name='export'),
    path('dashboard/', views.analysis_dashboard, name='dashboard'),
    path('<int:paper_id>/status/', views.analysis_status_ajax, name='status_ajax'),
] 