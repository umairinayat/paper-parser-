from django.db import models
from django.conf import settings

# Create your models here.


class Paper(models.Model):
    ANALYSIS_STATUS = (
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    )
    
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    title = models.CharField(max_length=500)
    authors = models.JSONField(default=list)
    year = models.IntegerField(null=True, blank=True)
    file = models.FileField(upload_to='papers/')
    file_size = models.BigIntegerField()
    status = models.CharField(max_length=20, choices=ANALYSIS_STATUS, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    processing_time = models.FloatField(null=True, blank=True)
    
    # Additional metadata
    abstract = models.TextField(blank=True)
    keywords = models.JSONField(default=list)
    doi = models.CharField(max_length=100, blank=True)
    journal = models.CharField(max_length=200, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Paper'
        verbose_name_plural = 'Papers'
    
    def __str__(self):
        return self.title
    
    @property
    def file_extension(self):
        return self.file.name.split('.')[-1].lower() if self.file else ''
    
    @property
    def is_processed(self):
        return self.status == 'completed'

