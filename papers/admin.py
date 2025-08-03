from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.utils.safestring import mark_safe
from django.db.models import Count, Avg, Sum
from django.utils import timezone
from datetime import timedelta

from .models import (
    Paper, Author, Journal, PaperAnalysis, ScrapingTask, 
    SearchQuery, SearchResult, PaperAuthor
)


@admin.register(Author)
class AuthorAdmin(admin.ModelAdmin):
    list_display = ['name', 'email', 'affiliation', 'h_index', 'citations_count', 'papers_count', 'created_at']
    list_filter = ['h_index', 'citations_count', 'papers_count', 'created_at']
    search_fields = ['name', 'email', 'affiliation', 'orcid_id']
    readonly_fields = ['created_at', 'updated_at']
    ordering = ['-citations_count', '-h_index']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'email', 'affiliation', 'orcid_id')
        }),
        ('Metrics', {
            'fields': ('h_index', 'citations_count', 'papers_count')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def get_queryset(self, request):
        return super().get_queryset(request).annotate(
            papers_count=Count('paperauthor')
        )


@admin.register(Journal)
class JournalAdmin(admin.ModelAdmin):
    list_display = ['name', 'publisher', 'impact_factor', 'h_index', 'quartile', 'subject_area', 'created_at']
    list_filter = ['impact_factor', 'h_index', 'quartile', 'publisher', 'created_at']
    search_fields = ['name', 'publisher', 'subject_area', 'issn']
    readonly_fields = ['created_at', 'updated_at']
    ordering = ['-impact_factor', '-h_index']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'publisher', 'issn', 'subject_area')
        }),
        ('Metrics', {
            'fields': ('impact_factor', 'h_index', 'quartile')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )


class PaperAuthorInline(admin.TabularInline):
    model = PaperAuthor
    extra = 1
    autocomplete_fields = ['author']


@admin.register(Paper)
class PaperAdmin(admin.ModelAdmin):
    list_display = [
        'title', 'first_author', 'journal', 'publication_date', 'source', 
        'analysis_status', 'citation_count', 'created_at'
    ]
    list_filter = [
        'source', 'analysis_status', 'publication_date', 'created_at', 
        'journal', 'subject_areas'
    ]
    search_fields = ['title', 'abstract', 'doi', 'arxiv_id', 'pmid', 'first_author']
    readonly_fields = [
        'id', 'created_at', 'updated_at', 'scraped_at', 'processed_at', 
        'processing_time', 'error_message'
    ]
    autocomplete_fields = ['journal']
    ordering = ['-created_at']
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('title', 'abstract', 'doi', 'arxiv_id', 'pmid')
        }),
        ('Publication', {
            'fields': ('publication_date', 'journal', 'volume', 'issue', 'pages')
        }),
        ('Authors', {
            'fields': ('first_author', 'corresponding_author')
        }),
        ('Content', {
            'fields': ('keywords', 'subject_areas', 'full_text', 'summary')
        }),
        ('Metrics', {
            'fields': ('citation_count', 'download_count', 'view_count')
        }),
        ('Source & Processing', {
            'fields': ('source', 'source_url', 'analysis_status', 'user')
        }),
        ('File', {
            'fields': ('file', 'file_size', 'file_hash')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at', 'scraped_at', 'processed_at'),
            'classes': ('collapse',)
        }),
        ('Processing', {
            'fields': ('processing_time', 'error_message'),
            'classes': ('collapse',)
        }),
    )
    
    inlines = [PaperAuthorInline]
    
    actions = ['mark_as_analyzed', 'mark_as_pending', 'queue_for_analysis']
    
    def mark_as_analyzed(self, request, queryset):
        updated = queryset.update(analysis_status='completed')
        self.message_user(request, f'{updated} papers marked as analyzed.')
    mark_as_analyzed.short_description = "Mark selected papers as analyzed"
    
    def mark_as_pending(self, request, queryset):
        updated = queryset.update(analysis_status='pending')
        self.message_user(request, f'{updated} papers marked as pending.')
    mark_as_pending.short_description = "Mark selected papers as pending"
    
    def queue_for_analysis(self, request, queryset):
        from papers.tasks import analyze_paper
        count = 0
        for paper in queryset:
            if paper.analysis_status == 'pending':
                analyze_paper.delay(str(paper.id))
                count += 1
        self.message_user(request, f'{count} papers queued for analysis.')
    queue_for_analysis.short_description = "Queue selected papers for analysis"


@admin.register(PaperAnalysis)
class PaperAnalysisAdmin(admin.ModelAdmin):
    list_display = [
        'paper_title', 'analysis_type', 'model_used', 'processing_time', 
        'confidence_score', 'created_at'
    ]
    list_filter = ['analysis_type', 'model_used', 'confidence_score', 'created_at']
    search_fields = ['paper__title', 'summary', 'methodology']
    readonly_fields = [
        'paper', 'created_at', 'updated_at', 'processing_time', 'confidence_score'
    ]
    ordering = ['-created_at']
    
    fieldsets = (
        ('Paper', {
            'fields': ('paper',)
        }),
        ('Analysis Type', {
            'fields': ('analysis_type', 'model_used')
        }),
        ('Content', {
            'fields': ('summary', 'key_findings', 'methodology', 'limitations', 'future_work', 'impact_assessment')
        }),
        ('Technical Details', {
            'fields': ('methodology_type', 'dataset_info', 'evaluation_metrics')
        }),
        ('Metadata', {
            'fields': ('processing_time', 'confidence_score', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
        ('Embeddings', {
            'fields': ('summary_embedding', 'key_findings_embedding'),
            'classes': ('collapse',)
        }),
    )
    
    def paper_title(self, obj):
        return obj.paper.title if obj.paper else 'N/A'
    paper_title.short_description = 'Paper Title'
    
    def has_add_permission(self, request):
        return False  # Analyses should be created by the system, not manually


@admin.register(ScrapingTask)
class ScrapingTaskAdmin(admin.ModelAdmin):
    list_display = [
        'id', 'source', 'query', 'status', 'papers_found', 'papers_processed', 
        'papers_skipped', 'papers_failed', 'created_at', 'completed_at'
    ]
    list_filter = ['source', 'status', 'created_at', 'completed_at']
    search_fields = ['query', 'task_id', 'error_message']
    readonly_fields = [
        'id', 'task_id', 'created_at', 'started_at', 'completed_at', 
        'papers_found', 'papers_processed', 'papers_skipped', 'papers_failed',
        'error_message', 'retry_count'
    ]
    ordering = ['-created_at']
    
    fieldsets = (
        ('Task Information', {
            'fields': ('id', 'task_id', 'source', 'query', 'category', 'max_results')
        }),
        ('Status', {
            'fields': ('status', 'retry_count', 'max_retries')
        }),
        ('Results', {
            'fields': ('papers_found', 'papers_processed', 'papers_skipped', 'papers_failed')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'started_at', 'completed_at')
        }),
        ('Error Information', {
            'fields': ('error_message',),
            'classes': ('collapse',)
        }),
        ('User', {
            'fields': ('user',)
        }),
    )
    
    actions = ['retry_failed_tasks']
    
    def retry_failed_tasks(self, request, queryset):
        from papers.tasks import scrape_papers
        count = 0
        for task in queryset.filter(status='failed'):
            if task.retry_count < task.max_retries:
                # Reset task status and retry
                task.status = 'pending'
                task.retry_count += 1
                task.error_message = ''
                task.save()
                
                # Queue the scraping task
                scrape_papers.delay(
                    source=task.source,
                    query=task.query,
                    max_results=task.max_results,
                    user_id=task.user.id if task.user else None
                )
                count += 1
        self.message_user(request, f'{count} failed tasks queued for retry.')
    retry_failed_tasks.short_description = "Retry selected failed tasks"


@admin.register(SearchQuery)
class SearchQueryAdmin(admin.ModelAdmin):
    list_display = ['query', 'user', 'results_count', 'created_at']
    list_filter = ['created_at']
    search_fields = ['query', 'user__username']
    readonly_fields = ['created_at']
    ordering = ['-created_at']
    
    fieldsets = (
        ('Query Information', {
            'fields': ('query', 'user', 'filters')
        }),
        ('Results', {
            'fields': ('results_count',)
        }),
        ('Timestamps', {
            'fields': ('created_at',)
        }),
    )


@admin.register(SearchResult)
class SearchResultAdmin(admin.ModelAdmin):
    list_display = ['search_query', 'paper_title', 'relevance_score', 'rank']
    list_filter = ['rank', 'relevance_score']
    search_fields = ['search_query__query', 'paper__title']
    ordering = ['search_query', 'rank']
    
    def paper_title(self, obj):
        return obj.paper.title if obj.paper else 'N/A'
    paper_title.short_description = 'Paper Title'


# Custom admin site configuration
admin.site.site_header = "Academic Paper Analyzer Admin"
admin.site.site_title = "Paper Analyzer Admin"
admin.site.index_title = "Welcome to Academic Paper Analyzer Administration"
