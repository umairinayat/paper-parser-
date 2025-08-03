"""
AI analysis service for comprehensive paper analysis using OpenAI/Groq APIs.
"""

import logging
import time
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

import openai
from groq import Groq
from django.conf import settings
from django.utils import timezone

from papers.models import Paper, PaperAnalysis

logger = logging.getLogger(__name__)


class AIAnalysisService:
    """Service for AI-powered paper analysis."""
    
    def __init__(self):
        self.groq_client = None
        self.openai_client = None
        
        # Initialize Groq client
        if hasattr(settings, 'GROQ_API_KEY') and settings.GROQ_API_KEY:
            self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
        
        # Initialize OpenAI client
        if hasattr(settings, 'OPENAI_API_KEY') and settings.OPENAI_API_KEY:
            self.openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
    
    def analyze_paper(self, paper: Paper) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a paper using AI.
        
        Args:
            paper: Paper object to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        start_time = time.time()
        
        try:
            # Prepare paper content for analysis
            paper_content = self._prepare_paper_content(paper)
            
            # Perform analysis using available AI services
            if self.groq_client:
                analysis_result = self._analyze_with_groq(paper_content)
            elif self.openai_client:
                analysis_result = self._analyze_with_openai(paper_content)
            else:
                raise Exception("No AI service available")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Add metadata
            analysis_result.update({
                'processing_time': processing_time,
                'model_used': analysis_result.get('model_used', 'unknown'),
                'confidence_score': analysis_result.get('confidence_score', 0.8),
            })
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Analysis failed for paper {paper.id}: {str(e)}")
            raise
    
    def _prepare_paper_content(self, paper: Paper) -> str:
        """Prepare paper content for AI analysis."""
        content_parts = []
        
        # Add title
        if paper.title:
            content_parts.append(f"Title: {paper.title}")
        
        # Add abstract
        if paper.abstract:
            content_parts.append(f"Abstract: {paper.abstract}")
        
        # Add authors (handle as string for now)
        if paper.authors:
            content_parts.append(f"Authors: {paper.authors}")
        
        # Add keywords
        if paper.keywords:
            # Handle keywords as string or list
            if isinstance(paper.keywords, str):
                keywords_str = paper.keywords
            else:
                keywords_str = ', '.join(paper.keywords)
            content_parts.append(f"Keywords: {keywords_str}")
        
        # Add subject areas
        if paper.subject_areas:
            # Handle subject_areas as string or list
            if isinstance(paper.subject_areas, str):
                subject_areas_str = paper.subject_areas
            else:
                subject_areas_str = ', '.join(paper.subject_areas)
            content_parts.append(f"Subject Areas: {subject_areas_str}")
        
        # Add publication info
        if paper.journal:
            content_parts.append(f"Journal: {paper.journal.name}")
        
        if paper.publication_date:
            content_parts.append(f"Publication Date: {paper.publication_date}")
        
        # Add metrics
        content_parts.append(f"Citation Count: {paper.citation_count}")
        
        # Add full text if available
        if paper.full_text:
            content_parts.append(f"Full Text: {paper.full_text[:5000]}...")  # Limit to first 5000 chars
        
        return "\n\n".join(content_parts)
    
    def _analyze_with_groq(self, paper_content: str) -> Dict[str, Any]:
        """Analyze paper using Groq API."""
        try:
            prompt = self._create_analysis_prompt(paper_content)
            
            response = self.groq_client.chat.completions.create(
                model=settings.PRIMARY_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert research analyst specializing in academic paper analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=4000,
            )
            
            analysis_text = response.choices[0].message.content
            return self._parse_analysis_response(analysis_text)
            
        except Exception as e:
            logger.error(f"Groq analysis failed: {str(e)}")
            raise
    
    def _analyze_with_openai(self, paper_content: str) -> Dict[str, Any]:
        """Analyze paper using OpenAI API."""
        try:
            prompt = self._create_analysis_prompt(paper_content)
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert research analyst specializing in academic paper analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=4000,
            )
            
            analysis_text = response.choices[0].message.content
            return self._parse_analysis_response(analysis_text)
            
        except Exception as e:
            logger.error(f"OpenAI analysis failed: {str(e)}")
            raise
    
    def _create_analysis_prompt(self, paper_content: str) -> str:
        """Create a comprehensive analysis prompt."""
        return f"""
Please provide a comprehensive analysis of the following academic paper. Your analysis should be structured and detailed.

Paper Content:
{paper_content}

Please provide your analysis in the following JSON format:

{{
    "summary": "A concise summary of the paper's main contributions and findings (2-3 paragraphs)",
    "key_findings": [
        "Key finding 1",
        "Key finding 2",
        "Key finding 3"
    ],
    "methodology": "Detailed description of the research methodology, approach, and techniques used",
    "limitations": "Discussion of the paper's limitations, constraints, and potential issues",
    "future_work": "Suggestions for future research directions and potential improvements",
    "impact_assessment": "Assessment of the paper's potential impact on the field",
    "methodology_type": "Type of methodology (e.g., 'Machine Learning', 'Survey', 'Experimental', 'Theoretical')",
    "dataset_info": "Information about datasets used, if any",
    "evaluation_metrics": [
        "Metric 1: value",
        "Metric 2: value"
    ],
    "confidence_score": 0.85
}}

Please ensure your analysis is:
1. Accurate and based on the provided content
2. Comprehensive but concise
3. Well-structured and professional
4. Focused on the paper's contributions and significance

Respond only with the JSON analysis, no additional text.
"""
    
    def _parse_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the AI response into structured data."""
        try:
            # Try to extract JSON from the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_text = response_text[start_idx:end_idx]
                analysis_data = json.loads(json_text)
                
                # Ensure all required fields are present
                required_fields = [
                    'summary', 'key_findings', 'methodology', 'limitations',
                    'future_work', 'impact_assessment', 'methodology_type',
                    'dataset_info', 'evaluation_metrics', 'confidence_score'
                ]
                
                for field in required_fields:
                    if field not in analysis_data:
                        analysis_data[field] = '' if field != 'key_findings' and field != 'evaluation_metrics' else []
                
                return analysis_data
            else:
                # Fallback: create basic structure
                return {
                    'summary': response_text,
                    'key_findings': [],
                    'methodology': '',
                    'limitations': '',
                    'future_work': '',
                    'impact_assessment': '',
                    'methodology_type': '',
                    'dataset_info': '',
                    'evaluation_metrics': [],
                    'confidence_score': 0.7
                }
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse analysis response: {str(e)}")
            # Return basic structure with the raw response as summary
            return {
                'summary': response_text,
                'key_findings': [],
                'methodology': '',
                'limitations': '',
                'future_work': '',
                'impact_assessment': '',
                'methodology_type': '',
                'dataset_info': '',
                'evaluation_metrics': [],
                'confidence_score': 0.6
            }
    
    def generate_summary(self, paper: Paper) -> str:
        """Generate a concise summary of the paper."""
        paper_content = self._prepare_paper_content(paper)
        
        prompt = f"""
Please provide a concise summary (2-3 paragraphs) of the following academic paper:

{paper_content}

Focus on the main contributions, methodology, and key findings.
"""
        
        try:
            if self.groq_client:
                response = self.groq_client.chat.completions.create(
                    model=settings.PRIMARY_MODEL,
                    messages=[
                        {"role": "system", "content": "You are an expert at summarizing academic papers."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500,
                )
                return response.choices[0].message.content
            elif self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an expert at summarizing academic papers."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500,
                )
                return response.choices[0].message.content
            else:
                return "Summary generation not available."
                
        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}")
            return "Error generating summary."
    
    def extract_key_findings(self, paper: Paper) -> List[str]:
        """Extract key findings from the paper."""
        paper_content = self._prepare_paper_content(paper)
        
        prompt = f"""
Please extract the key findings from the following academic paper. Return them as a numbered list:

{paper_content}

Format your response as:
1. Finding 1
2. Finding 2
3. Finding 3
"""
        
        try:
            if self.groq_client:
                response = self.groq_client.chat.completions.create(
                    model=settings.PRIMARY_MODEL,
                    messages=[
                        {"role": "system", "content": "You are an expert at extracting key findings from academic papers."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=300,
                )
                return self._parse_findings_list(response.choices[0].message.content)
            elif self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an expert at extracting key findings from academic papers."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=300,
                )
                return self._parse_findings_list(response.choices[0].message.content)
            else:
                return []
                
        except Exception as e:
            logger.error(f"Key findings extraction failed: {str(e)}")
            return []
    
    def _parse_findings_list(self, text: str) -> List[str]:
        """Parse numbered findings list from text."""
        findings = []
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('•') or line.startswith('-') or line.startswith('*')):
                # Remove numbering/bullets and clean up
                finding = line.lstrip('0123456789.•-* ').strip()
                if finding:
                    findings.append(finding)
        
        return findings 