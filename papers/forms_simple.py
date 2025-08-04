from django import forms
from django.core.exceptions import ValidationError
from django.conf import settings
import os

from .models import Paper

class SimpleUploadForm(forms.ModelForm):
    """Simplified paper upload form - only requires file upload"""
    
    class Meta:
        model = Paper
        fields = ['file']
        widgets = {
            'file': forms.FileInput(attrs={
                'class': 'form-control form-control-lg',
                'accept': '.pdf,.docx,.txt'
            }),
        }

    def clean_file(self):
        file = self.cleaned_data.get('file')
        if not file:
            raise ValidationError('Please select a file to upload.')
        
        # Check file size
        if file.size > settings.MAX_UPLOAD_SIZE:
            raise ValidationError(f'File size must be under {settings.MAX_UPLOAD_SIZE // (1024*1024)}MB.')
        
        # Check file extension
        ext = os.path.splitext(file.name)[1].lower()
        if ext not in settings.ALLOWED_EXTENSIONS:
            raise ValidationError(f'File type {ext} is not supported. Please upload a PDF, DOCX, or TXT file.')
        
        return file
    
    def save(self, commit=True):
        paper = super().save(commit=False)
        
        # Auto-generate title from filename if not provided
        if not paper.title and paper.file:
            filename = os.path.splitext(paper.file.name)[0]
            # Clean up filename for title
            paper.title = filename.replace('_', ' ').replace('-', ' ').title()
        
        # Set default values
        paper.source = 'manual'
        paper.analysis_status = 'pending'
        
        if commit:
            paper.save()
        
        return paper