from django.contrib import admin
from .models import AnalysisSession, AnalysisSessionPaper, AnalysisResult, AnalysisTemplate


class AnalysisSessionPaperInline(admin.TabularInline):
    model = AnalysisSessionPaper
    extra = 0
    readonly_fields = ['created_at']


@admin.register(AnalysisSession)
class AnalysisSessionAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'query', 'status', 'papers_count', 'created_at', 'completed_at']
    list_filter = ['status', 'created_at', 'completed_at']
    search_fields = ['query', 'user__username']
    readonly_fields = ['id', 'created_at', 'completed_at']
    ordering = ['-created_at']
    
    fieldsets = (
        ('Session Information', {
            'fields': ('id', 'user', 'query', 'status')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'completed_at')
        }),
    )
    
    inlines = [AnalysisSessionPaperInline]
    
    def papers_count(self, obj):
        return obj.papers.count()
    papers_count.short_description = 'Papers'


@admin.register(AnalysisResult)
class AnalysisResultAdmin(admin.ModelAdmin):
    list_display = ['paper_title', 'analysis_type', 'model_used', 'confidence_score', 'processing_time', 'created_at']
    list_filter = ['analysis_type', 'model_used', 'confidence_score', 'created_at']
    search_fields = ['paper__title', 'content']
    readonly_fields = ['created_at', 'updated_at']
    ordering = ['-created_at']
    
    fieldsets = (
        ('Paper', {
            'fields': ('paper',)
        }),
        ('Analysis', {
            'fields': ('analysis_type', 'content', 'confidence_score', 'processing_time', 'model_used')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def paper_title(self, obj):
        return obj.paper.title if obj.paper else 'N/A'
    paper_title.short_description = 'Paper Title'


@admin.register(AnalysisTemplate)
class AnalysisTemplateAdmin(admin.ModelAdmin):
    list_display = ['name', 'analysis_type', 'is_active', 'created_at']
    list_filter = ['analysis_type', 'is_active', 'created_at']
    search_fields = ['name', 'description', 'prompt_template']
    readonly_fields = ['created_at', 'updated_at']
    ordering = ['name']
    
    fieldsets = (
        ('Template Information', {
            'fields': ('name', 'description', 'analysis_type', 'is_active')
        }),
        ('Content', {
            'fields': ('prompt_template', 'output_format')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
