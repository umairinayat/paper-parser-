from django.db import models
from django.conf import settings
from django.core.validators import MinValueValidator, MaxValueValidator
import uuid
from datetime import datetime


class Author(models.Model):
    """Model for storing author information with affiliations and metrics."""
    name = models.CharField(max_length=200, db_index=True)
    email = models.EmailField(blank=True, null=True)
    affiliation = models.CharField(max_length=500, blank=True)
    orcid_id = models.CharField(max_length=50, blank=True, unique=True, null=True)
    h_index = models.IntegerField(null=True, blank=True)
    citations_count = models.IntegerField(default=0)
    papers_count = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['name']
        verbose_name = 'Author'
        verbose_name_plural = 'Authors'
    
    def __str__(self):
        return self.name


class Journal(models.Model):
    """Model for storing journal metadata with impact factors."""
    name = models.CharField(max_length=300, db_index=True)
    issn = models.CharField(max_length=20, blank=True, unique=True, null=True)
    publisher = models.CharField(max_length=200, blank=True)
    impact_factor = models.FloatField(null=True, blank=True)
    h_index = models.IntegerField(null=True, blank=True)
    quartile = models.CharField(max_length=10, blank=True)  # Q1, Q2, Q3, Q4
    subject_area = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['name']
        verbose_name = 'Journal'
        verbose_name_plural = 'Journals'
    
    def __str__(self):
        return self.name


class Paper(models.Model):
    """Enhanced model for storing academic papers with comprehensive metadata."""
    SOURCE_CHOICES = [
        ('arxiv', 'ArXiv'),
        ('semantic_scholar', 'Semantic Scholar'),
        ('pubmed', 'PubMed'),
        ('crossref', 'Crossref'),
        ('ieee', 'IEEE Xplore'),
        ('doaj', 'DOAJ'),
        ('springer', 'Springer'),
        ('elsevier', 'Elsevier'),
        ('manual', 'Manual Upload'),
    ]
    
    ANALYSIS_STATUS = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    # Core identifiers
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=500, db_index=True)
    abstract = models.TextField(blank=True)
    doi = models.CharField(max_length=100, blank=True, null=True, db_index=True)
    arxiv_id = models.CharField(max_length=50, blank=True, null=True, db_index=True)
    pmid = models.CharField(max_length=50, blank=True, null=True, db_index=True)
    
    # Publication metadata
    publication_date = models.DateField(null=True, blank=True)
    journal = models.ForeignKey(Journal, on_delete=models.SET_NULL, null=True, blank=True)
    volume = models.CharField(max_length=50, blank=True)
    issue = models.CharField(max_length=50, blank=True)
    pages = models.CharField(max_length=50, blank=True)
    
    # Authors and relationships
    authors = models.TextField(blank=True)  # Temporarily use TextField for compatibility
    first_author = models.CharField(max_length=200, blank=True)
    corresponding_author = models.CharField(max_length=200, blank=True)
    
    # Content and analysis
    keywords = models.JSONField(default=list, blank=True)
    subject_areas = models.JSONField(default=list, blank=True)
    full_text = models.TextField(blank=True)
    summary = models.TextField(blank=True)
    
    # Metrics
    citation_count = models.IntegerField(default=0)
    download_count = models.IntegerField(default=0)
    view_count = models.IntegerField(default=0)
    
    # Source and processing
    source = models.CharField(max_length=20, choices=SOURCE_CHOICES, default='manual')
    source_url = models.URLField(blank=True)
    analysis_status = models.CharField(max_length=20, choices=ANALYSIS_STATUS, default='pending')
    
    # File handling
    file = models.FileField(upload_to='papers/', blank=True, null=True)
    file_size = models.BigIntegerField(null=True, blank=True)
    file_hash = models.CharField(max_length=64, blank=True)  # SHA-256 hash
    
    # User association
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    scraped_at = models.DateTimeField(null=True, blank=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    
    # Processing metadata
    processing_time = models.FloatField(null=True, blank=True)
    error_message = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Paper'
        verbose_name_plural = 'Papers'
        indexes = [
            models.Index(fields=['title']),
            models.Index(fields=['doi']),
            models.Index(fields=['arxiv_id']),
            models.Index(fields=['publication_date']),
            models.Index(fields=['source']),
            models.Index(fields=['analysis_status']),
        ]
    
    def __str__(self):
        return self.title
    
    @property
    def file_extension(self):
        return self.file.name.split('.')[-1].lower() if self.file else ''
    
    @property
    def is_processed(self):
        return self.analysis_status == 'completed'
    
    @property
    def has_analysis(self):
        return hasattr(self, 'analysis') and self.analysis is not None
    
    def get_author_names(self):
        """Return list of author names as strings."""
        return [author.name for author in self.authors.all()]
    
    def get_citation_style(self):
        """Generate citation in APA format."""
        if not self.authors.exists():
            return self.title
        
        author_names = self.get_author_names()
        if len(author_names) == 1:
            authors_str = author_names[0]
        elif len(author_names) == 2:
            authors_str = f"{author_names[0]} & {author_names[1]}"
        else:
            authors_str = f"{', '.join(author_names[:-1])}, & {author_names[-1]}"
        
        year = self.publication_date.year if self.publication_date else ''
        title = self.title
        journal = self.journal.name if self.journal else ''
        
        return f"{authors_str} ({year}). {title}. {journal}."


class PaperAuthor(models.Model):
    """Through model for Paper-Author relationship with order and roles."""
    paper = models.ForeignKey(Paper, on_delete=models.CASCADE)
    author = models.ForeignKey(Author, on_delete=models.CASCADE)
    order = models.PositiveIntegerField(default=0)
    is_corresponding = models.BooleanField(default=False)
    is_first_author = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['order']
        unique_together = ['paper', 'author', 'order']
    
    def __str__(self):
        return f"{self.author.name} - {self.paper.title}"


class PaperAnalysis(models.Model):
    """Model for storing AI-generated analysis of papers."""
    ANALYSIS_TYPE_CHOICES = [
        ('summary', 'Summary'),
        ('key_findings', 'Key Findings'),
        ('methodology', 'Methodology'),
        ('limitations', 'Limitations'),
        ('future_work', 'Future Work'),
        ('impact', 'Impact Assessment'),
        ('comprehensive', 'Comprehensive Analysis'),
    ]
    
    paper = models.OneToOneField(Paper, on_delete=models.CASCADE, related_name='analysis')
    analysis_type = models.CharField(max_length=20, choices=ANALYSIS_TYPE_CHOICES, default='comprehensive')
    
    # Analysis content
    summary = models.TextField(blank=True)
    key_findings = models.JSONField(default=list)
    methodology = models.TextField(blank=True)
    limitations = models.TextField(blank=True)
    future_work = models.TextField(blank=True)
    impact_assessment = models.TextField(blank=True)
    
    # Technical details
    methodology_type = models.CharField(max_length=100, blank=True)  # e.g., "Machine Learning", "Survey"
    dataset_info = models.TextField(blank=True)
    evaluation_metrics = models.JSONField(default=list)
    
    # AI processing metadata
    model_used = models.CharField(max_length=100, blank=True)
    processing_time = models.FloatField(null=True, blank=True)
    confidence_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        null=True, blank=True
    )
    
    # Vector embeddings for semantic search
    summary_embedding = models.JSONField(null=True, blank=True)
    key_findings_embedding = models.JSONField(null=True, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'Paper Analysis'
        verbose_name_plural = 'Paper Analyses'
    
    def __str__(self):
        return f"Analysis of {self.paper.title}"


class ScrapingTask(models.Model):
    """Model for tracking scraping tasks and their status."""
    TASK_STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled'),
    ]
    
    SOURCE_CHOICES = [
        ('arxiv', 'ArXiv'),
        ('semantic_scholar', 'Semantic Scholar'),
        ('pubmed', 'PubMed'),
        ('crossref', 'Crossref'),
        ('ieee', 'IEEE Xplore'),
        ('doaj', 'DOAJ'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    task_id = models.CharField(max_length=255, unique=True)  # Celery task ID
    source = models.CharField(max_length=20, choices=SOURCE_CHOICES)
    query = models.CharField(max_length=500, blank=True)
    category = models.CharField(max_length=100, blank=True)
    max_results = models.IntegerField(default=100)
    
    # Status tracking
    status = models.CharField(max_length=20, choices=TASK_STATUS_CHOICES, default='pending')
    papers_found = models.IntegerField(default=0)
    papers_processed = models.IntegerField(default=0)
    papers_skipped = models.IntegerField(default=0)
    papers_failed = models.IntegerField(default=0)
    
    # Error handling
    error_message = models.TextField(blank=True)
    retry_count = models.IntegerField(default=0)
    max_retries = models.IntegerField(default=3)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    # User association
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Scraping Task'
        verbose_name_plural = 'Scraping Tasks'
    
    def __str__(self):
        return f"{self.source} scraping task - {self.status}"


class SearchQuery(models.Model):
    """Model for storing search queries and their results."""
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True, blank=True)
    query = models.CharField(max_length=500)
    filters = models.JSONField(default=dict)
    results_count = models.IntegerField(default=0)
    papers = models.ManyToManyField(Paper, through='SearchResult')
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Search Query'
        verbose_name_plural = 'Search Queries'
    
    def __str__(self):
        return f"Search: {self.query}"


class SearchResult(models.Model):
    """Through model for SearchQuery-Paper relationship with relevance scores."""
    search_query = models.ForeignKey(SearchQuery, on_delete=models.CASCADE)
    paper = models.ForeignKey(Paper, on_delete=models.CASCADE)
    relevance_score = models.FloatField(null=True, blank=True)
    rank = models.PositiveIntegerField()
    
    class Meta:
        ordering = ['rank']
        unique_together = ['search_query', 'paper']
    
    def __str__(self):
        return f"{self.paper.title} (rank: {self.rank})"

