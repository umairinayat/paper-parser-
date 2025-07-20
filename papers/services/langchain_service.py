import os
import logging
from typing import Dict, List, Any, Optional
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from django.conf import settings

logger = logging.getLogger(__name__)

class PaperAnalysisService:
    def __init__(self):
        self.groq_api_key = settings.GROQ_API_KEY
        self.primary_model = settings.PRIMARY_MODEL
        self.fallback_models = settings.FALLBACK_MODELS
        self.max_tokens = settings.MAX_TOKENS
        
        # Initialize models with fallback
        self.models = self._initialize_models()
        
        # Analysis prompts
        self.prompts = self._create_prompts()
    
    def _initialize_models(self) -> List[ChatGroq]:
        """Initialize Groq models with fallback chain"""
        models = []
        
        # Primary model
        try:
            primary_llm = ChatGroq(
                api_key=self.groq_api_key,
                model_name=self.primary_model,
                max_tokens=self.max_tokens
            )
            models.append(primary_llm)
        except Exception as e:
            logger.error(f"Failed to initialize primary model: {e}")
        
        # Fallback models
        for model_name in self.fallback_models:
            try:
                fallback_llm = ChatGroq(
                    api_key=self.groq_api_key,
                    model_name=model_name,
                    max_tokens=self.max_tokens
                )
                models.append(fallback_llm)
            except Exception as e:
                logger.error(f"Failed to initialize fallback model {model_name}: {e}")
        
        return models
    
    def _create_prompts(self) -> Dict[str, PromptTemplate]:
        """Create analysis prompts for different sections"""
        return {
            'abstract_summary': PromptTemplate(
                input_variables=["text"],
                template="""Analyze the following research paper abstract and provide a concise summary (2-3 sentences):

{text}

Summary:"""
            ),
            
            'main_findings': PromptTemplate(
                input_variables=["text"],
                template="""Extract the main findings and results from this research paper. List them as bullet points:

{text}

Main Findings:"""
            ),
            
            'methodology': PromptTemplate(
                input_variables=["text"],
                template="""Analyze the methodology section and extract:
1. Study design
2. Research objectives
3. Theoretical framework
4. Research questions
5. Hypotheses tested

{text}

Analysis:"""
            ),
            
            'intervention': PromptTemplate(
                input_variables=["text"],
                template="""Extract information about interventions and outcomes:
1. Intervention description
2. Intervention effects
3. Outcome measures
4. Measurement methods

{text}

Analysis:"""
            ),
            
            'findings': PromptTemplate(
                input_variables=["text"],
                template="""Extract research findings:
1. Primary outcomes
2. Secondary outcomes
3. Statistical significance
4. Effect sizes

{text}

Analysis:"""
            ),
            
            'critical_analysis': PromptTemplate(
                input_variables=["text"],
                template="""Identify critical aspects:
1. Study limitations
2. Research gaps
3. Future research directions
4. Methodological constraints

{text}

Analysis:"""
            ),
            
            'discussion': PromptTemplate(
                input_variables=["text"],
                template="""Summarize discussion sections:
1. Introduction summary
2. Discussion summary
3. Key arguments
4. Implications

{text}

Analysis:"""
            )
        }
    
    def _call_model_with_fallback(self, prompt: str, model_index: int = 0) -> Optional[str]:
        """Call model with fallback to next model if failed"""
        for i in range(model_index, len(self.models)):
            try:
                model = self.models[i]
                response = model.invoke([HumanMessage(content=prompt)])
                return response.content
            except Exception as e:
                logger.warning(f"Model {i} failed: {e}")
                if i < len(self.models) - 1:
                    continue
                else:
                    logger.error("All models failed")
                    return None
        return None
    
    def analyze_paper(self, paper_text: str, sections: Dict[str, str]) -> Dict[str, Any]:
        """Analyze a research paper and extract comprehensive information"""
        analysis_results = {}
        confidence_scores = {}
        
        # Analyze abstract and summary
        if sections.get('abstract'):
            abstract_summary = self._call_model_with_fallback(
                self.prompts['abstract_summary'].format(text=sections['abstract'])
            )
            analysis_results['abstract_summary'] = abstract_summary or "Abstract summary not available."
            
            main_findings = self._call_model_with_fallback(
                self.prompts['main_findings'].format(text=sections['abstract'])
            )
            analysis_results['main_findings'] = self._parse_list_response(main_findings)
        
        # Analyze methodology
        methodology_text = sections.get('methods', '') + ' ' + sections.get('introduction', '')
        if methodology_text.strip():
            methodology_analysis = self._call_model_with_fallback(
                self.prompts['methodology'].format(text=methodology_text)
            )
            parsed_methodology = self._parse_methodology(methodology_analysis)
            analysis_results.update(parsed_methodology)
        
        # Analyze interventions and outcomes
        results_text = sections.get('results', '') + ' ' + sections.get('methods', '')
        if results_text.strip():
            intervention_analysis = self._call_model_with_fallback(
                self.prompts['intervention'].format(text=results_text)
            )
            parsed_intervention = self._parse_intervention(intervention_analysis)
            analysis_results.update(parsed_intervention)
        
        # Analyze findings
        if sections.get('results'):
            findings_analysis = self._call_model_with_fallback(
                self.prompts['findings'].format(text=sections['results'])
            )
            parsed_findings = self._parse_findings(findings_analysis)
            analysis_results.update(parsed_findings)
        
        # Analyze critical aspects
        discussion_text = sections.get('discussion', '') + ' ' + sections.get('conclusion', '')
        if discussion_text.strip():
            critical_analysis = self._call_model_with_fallback(
                self.prompts['critical_analysis'].format(text=discussion_text)
            )
            parsed_critical = self._parse_critical_analysis(critical_analysis)
            analysis_results.update(parsed_critical)
        
        # Analyze discussion
        if discussion_text.strip():
            discussion_analysis = self._call_model_with_fallback(
                self.prompts['discussion'].format(text=discussion_text)
            )
            parsed_discussion = self._parse_discussion(discussion_analysis)
            analysis_results.update(parsed_discussion)
        
        return analysis_results
    
    def _parse_list_response(self, response: str) -> List[str]:
        """Parse list response from AI model"""
        if not response:
            return []
        
        lines = response.strip().split('\n')
        items = []
        for line in lines:
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                items.append(line[1:].strip())
            elif line and line[0].isdigit() and '.' in line:
                items.append(line.split('.', 1)[1].strip())
        
        return items if items else [response.strip()]
    
    def _parse_methodology(self, response: str) -> Dict[str, Any]:
        """Parse methodology analysis response"""
        if not response:
            return {
                'study_design': 'Not specified',
                'study_objectives': [],
                'theoretical_framework': 'Not specified',
                'research_question': 'Not specified',
                'hypotheses_tested': []
            }
        
        # Simple parsing - in production, you'd want more sophisticated parsing
        lines = response.split('\n')
        result = {
            'study_design': 'Not specified',
            'study_objectives': [],
            'theoretical_framework': 'Not specified',
            'research_question': 'Not specified',
            'hypotheses_tested': []
        }
        
        current_section = None
        for line in lines:
            line = line.strip()
            if 'study design' in line.lower():
                result['study_design'] = line.split(':', 1)[1].strip() if ':' in line else line
            elif 'objectives' in line.lower():
                result['study_objectives'] = self._parse_list_response(line)
            elif 'framework' in line.lower():
                result['theoretical_framework'] = line.split(':', 1)[1].strip() if ':' in line else line
            elif 'question' in line.lower():
                result['research_question'] = line.split(':', 1)[1].strip() if ':' in line else line
            elif 'hypothesis' in line.lower():
                result['hypotheses_tested'] = self._parse_list_response(line)
        
        return result
    
    def _parse_intervention(self, response: str) -> Dict[str, Any]:
        """Parse intervention analysis response"""
        if not response:
            return {
                'intervention': 'Not specified',
                'intervention_effects': 'Not specified',
                'outcome_measured': [],
                'measurement_methods': []
            }
        
        # Simple parsing
        lines = response.split('\n')
        result = {
            'intervention': 'Not specified',
            'intervention_effects': 'Not specified',
            'outcome_measured': [],
            'measurement_methods': []
        }
        
        for line in lines:
            line = line.strip()
            if 'intervention' in line.lower():
                result['intervention'] = line.split(':', 1)[1].strip() if ':' in line else line
            elif 'effect' in line.lower():
                result['intervention_effects'] = line.split(':', 1)[1].strip() if ':' in line else line
            elif 'outcome' in line.lower():
                result['outcome_measured'] = self._parse_list_response(line)
            elif 'measurement' in line.lower():
                result['measurement_methods'] = self._parse_list_response(line)
        
        return result
    
    def _parse_findings(self, response: str) -> Dict[str, Any]:
        """Parse findings analysis response"""
        if not response:
            return {
                'primary_outcomes': [],
                'secondary_outcomes': [],
                'statistical_significance': 'Not specified',
                'effect_sizes': []
            }
        
        lines = response.split('\n')
        result = {
            'primary_outcomes': [],
            'secondary_outcomes': [],
            'statistical_significance': 'Not specified',
            'effect_sizes': []
        }
        
        for line in lines:
            line = line.strip()
            if 'primary' in line.lower():
                result['primary_outcomes'] = self._parse_list_response(line)
            elif 'secondary' in line.lower():
                result['secondary_outcomes'] = self._parse_list_response(line)
            elif 'significance' in line.lower():
                result['statistical_significance'] = line.split(':', 1)[1].strip() if ':' in line else line
            elif 'effect size' in line.lower():
                result['effect_sizes'] = self._parse_list_response(line)
        
        return result
    
    def _parse_critical_analysis(self, response: str) -> Dict[str, Any]:
        """Parse critical analysis response"""
        if not response:
            return {
                'limitations': [],
                'research_gaps': [],
                'future_research': [],
                'methodological_constraints': []
            }
        
        lines = response.split('\n')
        result = {
            'limitations': [],
            'research_gaps': [],
            'future_research': [],
            'methodological_constraints': []
        }
        
        for line in lines:
            line = line.strip()
            if 'limitation' in line.lower():
                result['limitations'] = self._parse_list_response(line)
            elif 'gap' in line.lower():
                result['research_gaps'] = self._parse_list_response(line)
            elif 'future' in line.lower():
                result['future_research'] = self._parse_list_response(line)
            elif 'constraint' in line.lower():
                result['methodological_constraints'] = self._parse_list_response(line)
        
        return result
    
    def _parse_discussion(self, response: str) -> Dict[str, Any]:
        """Parse discussion analysis response"""
        if not response:
            return {
                'introduction_summary': 'Not available',
                'discussion_summary': 'Not available',
                'key_arguments': [],
                'implications': []
            }
        
        lines = response.split('\n')
        result = {
            'introduction_summary': 'Not available',
            'discussion_summary': 'Not available',
            'key_arguments': [],
            'implications': []
        }
        
        for line in lines:
            line = line.strip()
            if 'introduction' in line.lower():
                result['introduction_summary'] = line.split(':', 1)[1].strip() if ':' in line else line
            elif 'discussion' in line.lower():
                result['discussion_summary'] = line.split(':', 1)[1].strip() if ':' in line else line
            elif 'argument' in line.lower():
                result['key_arguments'] = self._parse_list_response(line)
            elif 'implication' in line.lower():
                result['implications'] = self._parse_list_response(line)
        
        return result 