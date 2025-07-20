from django.db import models
from papers.models import Paper

# Create your models here.

class PaperAnalysis(models.Model):
    paper = models.OneToOneField(Paper, on_delete=models.CASCADE, related_name='analysis')
    
    # Summary Analysis
    abstract_summary = models.TextField()
    main_findings = models.JSONField(default=list)
    key_conclusions = models.JSONField(default=list)
    
    # Methodology Analysis
    study_design = models.TextField()
    study_objectives = models.JSONField(default=list)
    theoretical_framework = models.TextField()
    research_question = models.TextField()
    hypotheses_tested = models.JSONField(default=list)
    
    # Intervention Analysis
    intervention = models.TextField()
    intervention_effects = models.TextField()
    outcome_measured = models.JSONField(default=list)
    measurement_methods = models.JSONField(default=list)
    
    # Findings Analysis
    primary_outcomes = models.JSONField(default=list)
    secondary_outcomes = models.JSONField(default=list)
    statistical_significance = models.TextField()
    effect_sizes = models.JSONField(default=list)
    
    # Critical Analysis
    limitations = models.JSONField(default=list)
    research_gaps = models.JSONField(default=list)
    future_research = models.JSONField(default=list)
    methodological_constraints = models.JSONField(default=list)
    
    # Discussion Analysis
    introduction_summary = models.TextField()
    discussion_summary = models.TextField()
    key_arguments = models.JSONField(default=list)
    implications = models.JSONField(default=list)
    
    # Related Papers
    related_papers = models.JSONField(default=list)
    
    # Metadata
    confidence_scores = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'Paper Analysis'
        verbose_name_plural = 'Paper Analyses'
    
    def __str__(self):
        return f"Analysis for {self.paper.title}"
    
    @property
    def has_complete_analysis(self):
        """Check if all analysis fields have been populated"""
        required_fields = [
            'abstract_summary', 'study_design', 'intervention',
            'introduction_summary', 'discussion_summary'
        ]
        return all(getattr(self, field) for field in required_fields)
