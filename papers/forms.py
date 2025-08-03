from django import forms
from django.core.exceptions import ValidationError
from django.conf import settings
import os

from .models import Paper, Author, Journal

class PaperUploadForm(forms.ModelForm):
    # Custom field for author names as text input
    author_names = forms.CharField(
        max_length=500,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter author names separated by commas'
        }),
        help_text='Enter author names separated by commas'
    )
    
    # Custom field for publication year
    publication_year = forms.IntegerField(
        required=False,
        min_value=1900,
        max_value=2100,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'YYYY'
        }),
        help_text='Publication year'
    )
    
    class Meta:
        model = Paper
        fields = ['title', 'abstract', 'doi', 'arxiv_id', 'pmid', 'journal', 'keywords', 'file']
        widgets = {
            'title': forms.TextInput(attrs={'class': 'form-control'}),
            'abstract': forms.Textarea(attrs={'class': 'form-control', 'rows': 4}),
            'doi': forms.URLInput(attrs={'class': 'form-control'}),
            'arxiv_id': forms.TextInput(attrs={'class': 'form-control'}),
            'pmid': forms.TextInput(attrs={'class': 'form-control'}),
            'journal': forms.Select(attrs={'class': 'form-select'}),
            'keywords': forms.TextInput(attrs={'class': 'form-control'}),
            'file': forms.FileInput(attrs={'class': 'form-control'}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Make journal field optional
        self.fields['journal'].required = False
        self.fields['journal'].queryset = Journal.objects.all()

    def clean_file(self):
        file = self.cleaned_data.get('file')
        if file:
            # Check file size
            if file.size > settings.MAX_UPLOAD_SIZE:
                raise ValidationError(f'File size must be under {settings.MAX_UPLOAD_SIZE // (1024*1024)}MB.')
            
            # Check file extension
            ext = os.path.splitext(file.name)[1].lower()
            if ext not in settings.ALLOWED_EXTENSIONS:
                raise ValidationError(f'File type {ext} is not supported. Please upload a PDF, DOCX, or TXT file.')
        
        return file
    
    def clean_keywords(self):
        keywords = self.cleaned_data.get('keywords')
        if keywords:
            # Convert comma-separated keywords to list
            return [kw.strip() for kw in keywords.split(',') if kw.strip()]
        return []
    
    def save(self, commit=True):
        paper = super().save(commit=False)
        
        # Handle publication year
        publication_year = self.cleaned_data.get('publication_year')
        if publication_year:
            from datetime import date
            paper.publication_date = date(publication_year, 1, 1)
        
        # Handle author names
        author_names = self.cleaned_data.get('author_names')
        if author_names:
            # Create or get authors
            author_list = []
            for name in author_names.split(','):
                name = name.strip()
                if name:
                    author, created = Author.objects.get_or_create(name=name)
                    author_list.append(author)
            
            if commit:
                paper.save()
                paper.authors.set(author_list)
            else:
                # Store authors for later assignment
                paper._author_list = author_list
        
        return paper


class PaperEditForm(forms.ModelForm):
    # Custom field for author names as text input
    author_names = forms.CharField(
        max_length=500,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter author names separated by commas'
        }),
        help_text='Enter author names separated by commas'
    )
    
    # Custom field for publication year
    publication_year = forms.IntegerField(
        required=False,
        min_value=1900,
        max_value=2100,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'YYYY'
        }),
        help_text='Publication year'
    )
    
    class Meta:
        model = Paper
        fields = ['title', 'abstract', 'doi', 'arxiv_id', 'pmid', 'journal', 'keywords']
        widgets = {
            'title': forms.TextInput(attrs={'class': 'form-control'}),
            'abstract': forms.Textarea(attrs={'class': 'form-control', 'rows': 4}),
            'doi': forms.URLInput(attrs={'class': 'form-control'}),
            'arxiv_id': forms.TextInput(attrs={'class': 'form-control'}),
            'pmid': forms.TextInput(attrs={'class': 'form-control'}),
            'journal': forms.Select(attrs={'class': 'form-select'}),
            'keywords': forms.TextInput(attrs={'class': 'form-control'}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Make journal field optional
        self.fields['journal'].required = False
        self.fields['journal'].queryset = Journal.objects.all()
        
        # Set initial values
        if self.instance.pk:
            # Set author names
            author_names = ', '.join([author.name for author in self.instance.authors.all()])
            self.fields['author_names'].initial = author_names
            
            # Set publication year
            if self.instance.publication_date:
                self.fields['publication_year'].initial = self.instance.publication_date.year
            
            # Set keywords
            if self.instance.keywords:
                self.fields['keywords'].initial = ', '.join(self.instance.keywords)

    def clean_keywords(self):
        keywords = self.cleaned_data.get('keywords')
        if keywords:
            # Convert comma-separated keywords to list
            return [kw.strip() for kw in keywords.split(',') if kw.strip()]
        return []
    
    def save(self, commit=True):
        paper = super().save(commit=False)
        
        # Handle publication year
        publication_year = self.cleaned_data.get('publication_year')
        if publication_year:
            from datetime import date
            paper.publication_date = date(publication_year, 1, 1)
        
        # Handle author names
        author_names = self.cleaned_data.get('author_names')
        if author_names:
            # Create or get authors
            author_list = []
            for name in author_names.split(','):
                name = name.strip()
                if name:
                    author, created = Author.objects.get_or_create(name=name)
                    author_list.append(author)
            
            if commit:
                paper.save()
                paper.authors.set(author_list)
            else:
                # Store authors for later assignment
                paper._author_list = author_list
        
        return paper


class PaperSearchForm(forms.Form):
    query = forms.CharField(
        max_length=500,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 4,
            'placeholder': 'Enter your research question or topic...'
        }),
        help_text='Describe what you are looking for in natural language'
    )
    
    max_papers = forms.IntegerField(
        min_value=5,
        max_value=100,
        initial=20,
        widget=forms.Select(choices=[
            (10, '10 papers'),
            (20, '20 papers'),
            (30, '30 papers'),
            (50, '50 papers'),
            (100, '100 papers')
        ], attrs={'class': 'form-select'}),
        help_text='Number of papers to analyze'
    )
    
    sources = forms.MultipleChoiceField(
        choices=[
            ('arxiv', 'ArXiv (Preprints & Physics/CS)'),
            ('semantic_scholar', 'Semantic Scholar (Multi-disciplinary)'),
            ('pubmed', 'PubMed (Biomedical)'),
            ('crossref', 'Crossref (Journal Articles)'),
            ('ieee', 'IEEE Xplore (Engineering & Technology)'),
            ('doaj', 'DOAJ (Open Access Journals)'),
        ],
        initial=['arxiv', 'crossref', 'pubmed', 'ieee', 'doaj', 'semantic_scholar'],
        widget=forms.CheckboxSelectMultiple(attrs={'class': 'form-check-input'}),
        help_text='Select academic sources to search',
        required=False
    )
    
    # Advanced filters
    year_from = forms.IntegerField(
        min_value=1900,
        max_value=2024,
        required=False,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'YYYY'
        }),
        help_text='From year (optional)'
    )
    
    year_to = forms.IntegerField(
        min_value=1900,
        max_value=2024,
        required=False,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'YYYY'
        }),
        help_text='To year (optional)'
    )
    
    min_citations = forms.IntegerField(
        min_value=0,
        required=False,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': '0'
        }),
        help_text='Minimum citation count (optional)'
    )
    
    subject_areas = forms.CharField(
        max_length=200,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'machine learning, computer vision, etc.'
        }),
        help_text='Subject areas separated by commas (optional)'
    )
    
    open_access_only = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        help_text='Only show open access papers'
    )
    
    def clean_sources(self):
        sources = self.cleaned_data.get('sources')
        if not sources:
            # Default to working sources if none selected
            return ['arxiv', 'crossref', 'pubmed', 'ieee', 'doaj', 'semantic_scholar']  # Add all sources
        return sources
    
    def clean_subject_areas(self):
        subject_areas = self.cleaned_data.get('subject_areas')
        if subject_areas:
            return [area.strip() for area in subject_areas.split(',') if area.strip()]
        return []
    
    def clean(self):
        cleaned_data = super().clean()
        year_from = cleaned_data.get('year_from')
        year_to = cleaned_data.get('year_to')
        
        if year_from and year_to and year_from > year_to:
            raise forms.ValidationError('From year must be less than or equal to To year.')
        
        return cleaned_data
    
    def clean_query(self):
        query = self.cleaned_data.get('query')
        if len(query.strip()) < 10:
            raise ValidationError('Please provide a more detailed search query (at least 10 characters).')
        return query.strip()


class ScrapingTaskForm(forms.Form):
    query = forms.CharField(
        max_length=500,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter search query...'
        }),
        help_text='Search query for papers'
    )
    
    source = forms.ChoiceField(
        choices=[
            ('arxiv', 'ArXiv'),
            ('semantic_scholar', 'Semantic Scholar'),
        ],
        widget=forms.Select(attrs={'class': 'form-select'}),
        help_text='Academic source to scrape'
    )
    
    max_results = forms.IntegerField(
        min_value=1,
        max_value=100,
        initial=20,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        help_text='Maximum number of papers to retrieve'
    )
    
    category = forms.CharField(
        max_length=100,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g., cs.AI, cs.LG (for ArXiv)'
        }),
        help_text='Category filter (optional)'
    ) 