from django.contrib import admin
from .models import PaperAnalysis

@admin.register(PaperAnalysis)
class PaperAnalysisAdmin(admin.ModelAdmin):
    list_display = ('paper', 'has_complete_analysis', 'created_at')
    list_filter = ('created_at', 'updated_at')
    search_fields = ('paper__title', 'abstract_summary', 'study_design')
    readonly_fields = ('created_at', 'updated_at')
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Paper Information', {
            'fields': ('paper',)
        }),
        ('Summary Analysis', {
            'fields': ('abstract_summary', 'main_findings', 'key_conclusions')
        }),
        ('Methodology', {
            'fields': ('study_design', 'study_objectives', 'theoretical_framework', 'research_question', 'hypotheses_tested')
        }),
        ('Intervention & Outcomes', {
            'fields': ('intervention', 'intervention_effects', 'outcome_measured', 'measurement_methods')
        }),
        ('Findings', {
            'fields': ('primary_outcomes', 'secondary_outcomes', 'statistical_significance', 'effect_sizes')
        }),
        ('Critical Analysis', {
            'fields': ('limitations', 'research_gaps', 'future_research', 'methodological_constraints')
        }),
        ('Discussion', {
            'fields': ('introduction_summary', 'discussion_summary', 'key_arguments', 'implications')
        }),
        ('Related Papers', {
            'fields': ('related_papers', 'confidence_scores')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('paper')
