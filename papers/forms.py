from django import forms
from django.core.exceptions import ValidationError
from django.conf import settings
import os

from .models import Paper

class PaperUploadForm(forms.ModelForm):
    class Meta:
        model = Paper
        fields = ['title', 'authors', 'year', 'journal', 'doi', 'file']
        widgets = {
            'title': forms.TextInput(attrs={'class': 'form-control'}),
            'authors': forms.TextInput(attrs={'class': 'form-control'}),
            'year': forms.NumberInput(attrs={'class': 'form-control'}),
            'journal': forms.TextInput(attrs={'class': 'form-control'}),
            'doi': forms.URLInput(attrs={'class': 'form-control'}),
            'file': forms.FileInput(attrs={'class': 'form-control'}),
        }

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

class PaperEditForm(forms.ModelForm):
    class Meta:
        model = Paper
        fields = ['title', 'authors', 'year', 'journal', 'doi']
        widgets = {
            'title': forms.TextInput(attrs={'class': 'form-control'}),
            'authors': forms.TextInput(attrs={'class': 'form-control'}),
            'year': forms.NumberInput(attrs={'class': 'form-control'}),
            'journal': forms.TextInput(attrs={'class': 'form-control'}),
            'doi': forms.URLInput(attrs={'class': 'form-control'}),
        }

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
        max_value=50,
        initial=20,
        widget=forms.Select(choices=[
            (10, '10 papers'),
            (20, '20 papers'),
            (30, '30 papers'),
            (50, '50 papers')
        ], attrs={'class': 'form-select'}),
        help_text='Number of papers to analyze'
    )
    
    def clean_query(self):
        query = self.cleaned_data.get('query')
        if len(query.strip()) < 10:
            raise ValidationError('Please provide a more detailed search query (at least 10 characters).')
        return query.strip() 