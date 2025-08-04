"""
Comparative Analysis Service for comparing multiple research papers.
"""

import logging
from typing import List, Dict, Any
from django.conf import settings
from groq import Groq
from .rag_service import RAGService

logger = logging.getLogger(__name__)

class ComparativeAnalysisService:
    """Service for comparing multiple research papers."""
    
    def __init__(self):
        self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
        self.rag_service = RAGService()
    
    def compare_papers(self, papers: List, comparison_aspects: List[str] = None) -> Dict[str, Any]:
        """Compare multiple papers across different aspects."""
        try:
            if len(papers) < 2:
                raise ValueError("At least 2 papers are required for comparison")
            
            if not comparison_aspects:
                comparison_aspects = [
                    "methodology",
                    "key_findings", 
                    "limitations",
                    "datasets_used",
                    "novelty_contribution"
                ]
            
            # Prepare paper summaries for comparison
            paper_summaries = []
            for paper in papers:
                summary = {
                    'id': str(paper.id),
                    'title': paper.title,
                    'authors': paper.authors,
                    'abstract': paper.abstract[:500] if paper.abstract else "",
                    'publication_date': paper.publication_date.year if paper.publication_date else "Unknown"
                }
                
                # Get full text if available
                if paper.full_text:
                    summary['content'] = paper.full_text[:2000]  # Truncate for API limits
                elif paper.abstract:
                    summary['content'] = paper.abstract
                else:
                    summary['content'] = "No content available"
                
                paper_summaries.append(summary)
            
            # Generate comparative analysis
            comparison_result = self._generate_comparison(paper_summaries, comparison_aspects)
            
            return {
                'papers': paper_summaries,
                'comparison_aspects': comparison_aspects,
                'analysis': comparison_result,
                'total_papers': len(papers),
                'generated_at': logger.info(f"Generated comparison for {len(papers)} papers")
            }
            
        except Exception as e:
            logger.error(f"Error in comparative analysis: {str(e)}")
            raise e
    
    def _generate_comparison(self, paper_summaries: List[Dict], aspects: List[str]) -> Dict[str, Any]:
        """Generate detailed comparison using AI."""
        
        # Prepare papers data for prompt
        papers_text = ""
        for i, paper in enumerate(paper_summaries, 1):
            papers_text += f"""
Paper {i}: {paper['title']}
Authors: {paper['authors']}
Year: {paper['publication_date']}
Abstract/Content: {paper['content']}

---
"""
        
        prompt = f"""
Compare the following {len(paper_summaries)} research papers across these aspects: {', '.join(aspects)}.

{papers_text}

Provide a comprehensive comparative analysis with the following structure:

1. OVERVIEW COMPARISON
   - Brief summary of each paper's main contribution
   - How the papers relate to each other

2. DETAILED COMPARISON BY ASPECT:
   {chr(10).join([f'   - {aspect.replace("_", " ").title()}' for aspect in aspects])}

3. STRENGTHS AND WEAKNESSES
   - What each paper does well
   - What each paper lacks

4. SYNTHESIS
   - Key insights from comparing these papers
   - Gaps that could be addressed by future work
   - Which paper(s) might be most relevant for different use cases

5. RECOMMENDATIONS
   - For researchers interested in this area
   - For practitioners looking to apply these methods

Be specific, analytical, and provide concrete examples from the papers.
"""
        
        try:
            response = self.groq_client.chat.completions.create(
                model=settings.PRIMARY_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert research analyst who specializes in comparative analysis of academic papers. Provide detailed, objective, and insightful comparisons."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=settings.MAX_TOKENS,
                temperature=0.1
            )
            
            analysis_text = response.choices[0].message.content
            
            # Structure the response
            return {
                'full_analysis': analysis_text,
                'summary': self._extract_summary(analysis_text),
                'comparison_matrix': self._create_comparison_matrix(paper_summaries, aspects),
                'model_used': settings.PRIMARY_MODEL
            }
            
        except Exception as e:
            logger.error(f"Error generating comparison: {str(e)}")
            return {
                'full_analysis': f"Error generating comparison: {str(e)}",
                'summary': "Comparison could not be generated due to an error.",
                'comparison_matrix': {},
                'model_used': settings.PRIMARY_MODEL
            }
    
    def _extract_summary(self, analysis_text: str) -> str:
        """Extract a brief summary from the full analysis."""
        lines = analysis_text.split('\n')
        summary_lines = []
        
        for line in lines[:10]:  # Take first 10 lines as summary
            if line.strip() and not line.strip().startswith('#'):
                summary_lines.append(line.strip())
        
        return ' '.join(summary_lines)[:500] + "..."
    
    def _create_comparison_matrix(self, papers: List[Dict], aspects: List[str]) -> Dict[str, Any]:
        """Create a structured comparison matrix."""
        matrix = {}
        
        for aspect in aspects:
            matrix[aspect] = {}
            for paper in papers:
                # This is a simplified matrix - in a real implementation,
                # you might extract specific data for each aspect
                matrix[aspect][paper['title'][:50]] = {
                    'present': True,  # Whether this aspect is discussed
                    'strength': 'Medium',  # Relative strength in this aspect
                    'details': 'Analysis pending'  # Specific details
                }
        
        return matrix
    
    def generate_literature_review(self, papers: List, topic: str = None) -> Dict[str, Any]:
        """Generate a literature review from multiple papers."""
        try:
            if not papers:
                raise ValueError("No papers provided for literature review")
            
            # Prepare papers for review
            papers_content = []
            for paper in papers:
                content = {
                    'title': paper.title,
                    'authors': paper.authors,
                    'year': paper.publication_date.year if paper.publication_date else "Unknown",
                    'abstract': paper.abstract[:500] if paper.abstract else "",
                    'key_content': paper.full_text[:1500] if paper.full_text else paper.abstract[:1500] if paper.abstract else ""
                }
                papers_content.append(content)
            
            topic_text = f" on the topic of '{topic}'" if topic else ""
            
            # Generate literature review
            papers_text = ""
            for paper in papers_content:
                papers_text += f"""
Title: {paper['title']}
Authors: {paper['authors']} ({paper['year']})
Content: {paper['key_content']}

---
"""
            
            prompt = f"""
Write a comprehensive literature review{topic_text} based on the following {len(papers)} research papers:

{papers_text}

Structure the literature review as follows:

1. INTRODUCTION
   - Brief overview of the research area
   - Scope of this review

2. BACKGROUND AND CONTEXT
   - Historical development of the field
   - Current state of research

3. METHODOLOGICAL APPROACHES
   - Different methodologies employed across studies
   - Evolution of research methods

4. KEY FINDINGS AND CONTRIBUTIONS
   - Major discoveries and innovations
   - Consistent findings across studies
   - Conflicting results and debates

5. RESEARCH GAPS AND LIMITATIONS
   - What hasn't been addressed adequately
   - Methodological limitations
   - Areas needing further investigation

6. FUTURE DIRECTIONS
   - Promising research avenues
   - Methodological improvements needed
   - Practical applications to explore

7. CONCLUSION
   - Summary of the current state
   - Key takeaways for researchers

Use proper academic writing style with clear transitions between sections. Cite the papers by title when referencing specific findings.
"""
            
            response = self.groq_client.chat.completions.create(
                model=settings.PRIMARY_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert academic writer specializing in literature reviews. Write comprehensive, well-structured reviews that synthesize research effectively."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=settings.MAX_TOKENS,
                temperature=0.2
            )
            
            review_text = response.choices[0].message.content
            
            return {
                'literature_review': review_text,
                'papers_included': len(papers),
                'topic': topic or "General Research Area",
                'generated_at': "Just now",
                'model_used': settings.PRIMARY_MODEL,
                'word_count': len(review_text.split())
            }
            
        except Exception as e:
            logger.error(f"Error generating literature review: {str(e)}")
            raise e