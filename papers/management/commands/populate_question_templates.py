from django.core.management.base import BaseCommand
from papers.models import QuestionTemplate

class Command(BaseCommand):
    help = 'Populate database with pre-built question templates'

    def handle(self, *args, **options):
        templates = [
            # Methodology Questions
            {
                'question_text': 'What methodology was used in this research?',
                'category': 'methodology',
                'description': 'General methodology overview'
            },
            {
                'question_text': 'What are the key steps in the proposed approach?',
                'category': 'methodology',
                'description': 'Step-by-step methodology breakdown'
            },
            {
                'question_text': 'What datasets were used in this study?',
                'category': 'methodology',
                'description': 'Information about data sources'
            },
            {
                'question_text': 'How was the experiment designed and conducted?',
                'category': 'methodology',
                'description': 'Experimental design details'
            },
            
            # Results & Findings Questions
            {
                'question_text': 'What are the main findings of this research?',
                'category': 'results',
                'description': 'Summary of key results'
            },
            {
                'question_text': 'What were the performance metrics and results?',
                'category': 'results',
                'description': 'Quantitative results and metrics'
            },
            {
                'question_text': 'How do the results compare to previous work?',
                'category': 'results',
                'description': 'Comparison with baseline methods'
            },
            {
                'question_text': 'What statistical significance was achieved?',
                'category': 'results',
                'description': 'Statistical analysis of results'
            },
            
            # Limitations Questions
            {
                'question_text': 'What are the limitations of this study?',
                'category': 'limitations',
                'description': 'Study limitations and constraints'
            },
            {
                'question_text': 'What assumptions were made in this research?',
                'category': 'limitations',
                'description': 'Underlying assumptions'
            },
            {
                'question_text': 'What are potential sources of bias or error?',
                'category': 'limitations',
                'description': 'Bias and error analysis'
            },
            
            # Future Work Questions
            {
                'question_text': 'What future research directions are suggested?',
                'category': 'future_work',
                'description': 'Proposed future research'
            },
            {
                'question_text': 'What improvements could be made to this approach?',
                'category': 'future_work',
                'description': 'Potential improvements'
            },
            {
                'question_text': 'What are the next steps for this research?',
                'category': 'future_work',
                'description': 'Immediate next steps'
            },
            
            # Background & Literature Questions
            {
                'question_text': 'What is the research problem being addressed?',
                'category': 'background',
                'description': 'Problem statement and motivation'
            },
            {
                'question_text': 'What prior work is this research building upon?',
                'category': 'background',
                'description': 'Literature review and foundations'
            },
            {
                'question_text': 'What gap in knowledge does this research fill?',
                'category': 'background',
                'description': 'Research gap identification'
            },
            
            # Impact & Applications Questions
            {
                'question_text': 'What are the practical applications of this research?',
                'category': 'impact',
                'description': 'Real-world applications'
            },
            {
                'question_text': 'What is the significance and impact of these findings?',
                'category': 'impact',
                'description': 'Research impact assessment'
            },
            {
                'question_text': 'How could this research benefit society or industry?',
                'category': 'impact',
                'description': 'Societal and industrial benefits'
            },
            
            # Data & Analysis Questions
            {
                'question_text': 'How was the data collected and processed?',
                'category': 'data',
                'description': 'Data collection methodology'
            },
            {
                'question_text': 'What data analysis techniques were employed?',
                'category': 'data',
                'description': 'Analysis methods used'
            },
            {
                'question_text': 'What was the sample size and selection criteria?',
                'category': 'data',
                'description': 'Sampling methodology'
            },
            
            # Comparison & Evaluation Questions
            {
                'question_text': 'How does this approach compare to existing methods?',
                'category': 'comparison',
                'description': 'Comparative analysis'
            },
            {
                'question_text': 'What evaluation criteria were used?',
                'category': 'comparison',
                'description': 'Evaluation methodology'
            },
            {
                'question_text': 'What are the advantages and disadvantages of this method?',
                'category': 'comparison',
                'description': 'Pros and cons analysis'
            },
        ]
        
        created_count = 0
        for template_data in templates:
            template, created = QuestionTemplate.objects.get_or_create(
                question_text=template_data['question_text'],
                defaults={
                    'category': template_data['category'],
                    'description': template_data['description']
                }
            )
            if created:
                created_count += 1
        
        self.stdout.write(
            self.style.SUCCESS(
                f'Successfully created {created_count} question templates'
            )
        )