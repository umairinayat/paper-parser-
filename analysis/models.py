from django.db import models
from django.conf import settings
from papers.models import Paper

# Create your models here.

class AnalysisSession(models.Model):
    """Model for tracking analysis sessions and their results."""
    SESSION_STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    id = models.UUIDField(primary_key=True, default=models.UUIDField().default, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True, blank=True)
    papers = models.ManyToManyField(Paper, through='AnalysisSessionPaper')
    query = models.CharField(max_length=500)
    status = models.CharField(max_length=20, choices=SESSION_STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Analysis Session'
        verbose_name_plural = 'Analysis Sessions'
    
    def __str__(self):
        return f"Analysis Session {self.id} - {self.status}"


class AnalysisSessionPaper(models.Model):
    """Through model for AnalysisSession-Paper relationship."""
    session = models.ForeignKey(AnalysisSession, on_delete=models.CASCADE)
    paper = models.ForeignKey(Paper, on_delete=models.CASCADE)
    analysis_status = models.CharField(max_length=20, choices=AnalysisSession.SESSION_STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['session', 'paper']
    
    def __str__(self):
        return f"{self.session.id} - {self.paper.title}"


class AnalysisResult(models.Model):
    """Model for storing detailed analysis results from AI."""
    ANALYSIS_TYPE_CHOICES = [
        ('summary', 'Summary'),
        ('methodology', 'Methodology'),
        ('findings', 'Findings'),
        ('limitations', 'Limitations'),
        ('implications', 'Implications'),
        ('comprehensive', 'Comprehensive'),
    ]
    
    paper = models.ForeignKey(Paper, on_delete=models.CASCADE, related_name='analysis_results')
    analysis_type = models.CharField(max_length=20, choices=ANALYSIS_TYPE_CHOICES)
    
    # Analysis content
    content = models.JSONField(default=dict)
    confidence_score = models.FloatField(null=True, blank=True)
    processing_time = models.FloatField(null=True, blank=True)
    model_used = models.CharField(max_length=100, blank=True)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Analysis Result'
        verbose_name_plural = 'Analysis Results'
        unique_together = ['paper', 'analysis_type']
    
    def __str__(self):
        return f"{self.analysis_type} analysis for {self.paper.title}"


class AnalysisTemplate(models.Model):
    """Model for storing analysis templates and prompts."""
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    analysis_type = models.CharField(max_length=20, choices=AnalysisResult.ANALYSIS_TYPE_CHOICES)
    
    # Template content
    prompt_template = models.TextField()
    output_format = models.JSONField(default=dict)
    
    # Metadata
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['name']
        verbose_name = 'Analysis Template'
        verbose_name_plural = 'Analysis Templates'
    
    def __str__(self):
        return f"{self.name} ({self.analysis_type})"
