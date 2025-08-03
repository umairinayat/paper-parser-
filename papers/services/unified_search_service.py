"""
Unified search service that aggregates results from multiple academic sources.
Similar to Elicit.com functionality for comprehensive paper search.
"""

import logging
import asyncio
import concurrent.futures
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict

from django.db import transaction
from django.utils import timezone

from .arxiv_scraper import ArXivScraper
from .semantic_scholar_scraper import SemanticScholarScraper
from .pubmed_scraper import PubMedScraper
from .crossref_scraper import CrossrefScraper
from .ieee_scraper import IEEEScraper
from .doaj_scraper import DOAJScraper
from papers.models import Paper, SearchQuery, ScrapingTask

logger = logging.getLogger(__name__)


@dataclass
class SearchFilter:
    """Filter options for unified search."""
    sources: List[str] = None  # ['arxiv', 'semantic_scholar', 'pubmed', 'crossref']
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    publishers: List[str] = None
    subject_areas: List[str] = None
    min_citations: Optional[int] = None
    open_access_only: bool = False
    publication_types: List[str] = None  # ['journal-article', 'conference-paper', 'preprint']


@dataclass
class SearchResult:
    """Standardized search result from any source."""
    source: str
    paper_data: Dict[str, Any]
    relevance_score: float = 0.0
    confidence_score: float = 1.0


class PaperDeduplicator:
    """Handles deduplication of papers from multiple sources."""
    
    def __init__(self):
        self.similarity_threshold = 0.9  # More conservative to avoid over-deduplication
    
    def deduplicate_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate papers based on DOI, ArXiv ID, PMID, and title similarity."""
        unique_papers = []
        seen_identifiers = set()
        seen_titles = {}
        
        for paper in papers:
            # Check for exact identifier matches
            identifiers = self._extract_identifiers(paper)
            
            # Skip if we've seen any of these identifiers
            if any(identifier in seen_identifiers for identifier in identifiers):
                continue
            
            # Check for title similarity
            title = (paper.get('title') or '').lower().strip()
            if title:
                is_duplicate = False
                for seen_title, seen_paper in seen_titles.items():
                    if self._calculate_title_similarity(title, seen_title) > self.similarity_threshold:
                        # Merge information from duplicate
                        merged_paper = self._merge_paper_data(seen_paper, paper)
                        seen_titles[seen_title] = merged_paper
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    seen_titles[title] = paper
                    unique_papers.append(paper)
                    seen_identifiers.update(identifiers)
            else:
                # Add papers without titles if they have unique identifiers
                if identifiers:
                    unique_papers.append(paper)
                    seen_identifiers.update(identifiers)
        
        return unique_papers
    
    def _extract_identifiers(self, paper: Dict[str, Any]) -> Set[str]:
        """Extract all available identifiers from a paper."""
        identifiers = set()
        
        if paper.get('doi'):
            identifiers.add(f"doi:{paper['doi']}")
        if paper.get('arxiv_id'):
            identifiers.add(f"arxiv:{paper['arxiv_id']}")
        if paper.get('pmid'):
            identifiers.add(f"pmid:{paper['pmid']}")
        
        return identifiers
    
    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles using simple token matching."""
        # Remove common words and punctuation
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        def tokenize(title):
            import re
            tokens = re.findall(r'\b\w+\b', title.lower())
            return set(token for token in tokens if token not in stop_words and len(token) > 2)
        
        tokens1 = tokenize(title1)
        tokens2 = tokenize(title2)
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _merge_paper_data(self, paper1: Dict[str, Any], paper2: Dict[str, Any]) -> Dict[str, Any]:
        """Merge data from two duplicate papers, preferring more complete information."""
        merged = paper1.copy()
        
        # Prefer non-empty values
        for key, value in paper2.items():
            if key not in merged or not merged[key]:
                merged[key] = value
            elif key == 'citation_count':
                # Take the higher citation count
                merged[key] = max(merged.get(key, 0), value or 0)
            elif key == 'authors' and isinstance(value, list) and len(value) > len(merged.get(key, [])):
                # Prefer more complete author list
                merged[key] = value
        
        return merged


class PaperRankingService:
    """Ranks papers based on relevance, citations, and other factors."""
    
    def __init__(self):
        self.weights = {
            'relevance': 0.4,
            'citations': 0.3,
            'recency': 0.2,
            'source_quality': 0.1
        }
    
    def rank_papers(self, papers: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Rank papers based on multiple factors."""
        scored_papers = []
        
        for paper in papers:
            score = self._calculate_score(paper, query)
            paper['_ranking_score'] = score
            scored_papers.append(paper)
        
        # Sort by score descending
        return sorted(scored_papers, key=lambda x: x['_ranking_score'], reverse=True)
    
    def _calculate_score(self, paper: Dict[str, Any], query: str) -> float:
        """Calculate a ranking score for a paper."""
        relevance_score = self._calculate_relevance_score(paper, query)
        citation_score = self._calculate_citation_score(paper)
        recency_score = self._calculate_recency_score(paper)
        source_score = self._calculate_source_score(paper)
        
        total_score = (
            relevance_score * self.weights['relevance'] +
            citation_score * self.weights['citations'] +
            recency_score * self.weights['recency'] +
            source_score * self.weights['source_quality']
        )
        
        return total_score
    
    def _calculate_relevance_score(self, paper: Dict[str, Any], query: str) -> float:
        """Calculate relevance score based on title and abstract matching."""
        query_terms = set(query.lower().split())
        
        # Check title match
        title = paper.get('title', '') or ''
        title = title.lower() if title else ''
        title_matches = sum(1 for term in query_terms if term in title)
        title_score = title_matches / len(query_terms) if query_terms else 0
        
        # Check abstract match
        abstract = paper.get('abstract', '') or ''
        abstract = abstract.lower() if abstract else ''
        abstract_matches = sum(1 for term in query_terms if term in abstract)
        abstract_score = abstract_matches / len(query_terms) if query_terms else 0
        
        # Weight title more heavily than abstract
        return (title_score * 0.7) + (abstract_score * 0.3)
    
    def _calculate_citation_score(self, paper: Dict[str, Any]) -> float:
        """Calculate score based on citation count (normalized)."""
        citations = paper.get('citation_count', 0)
        # Use log scale to prevent extremely high citations from dominating
        import math
        return math.log(max(citations, 1)) / math.log(1000)  # Normalize to ~0-1 range
    
    def _calculate_recency_score(self, paper: Dict[str, Any]) -> float:
        """Calculate score based on publication recency."""
        pub_date = paper.get('publication_date')
        if not pub_date:
            return 0.0
        
        if isinstance(pub_date, str):
            try:
                pub_date = datetime.fromisoformat(pub_date).date()
            except:
                return 0.0
        
        # Calculate years since publication
        current_year = datetime.now().year
        pub_year = pub_date.year if hasattr(pub_date, 'year') else 2000
        years_since = current_year - pub_year
        
        # Recent papers get higher scores
        return max(0, 1 - (years_since / 10))  # Papers older than 10 years get 0
    
    def _calculate_source_score(self, paper: Dict[str, Any]) -> float:
        """Calculate score based on source quality."""
        source_scores = {
            'semantic_scholar': 1.0,
            'pubmed': 0.95,
            'crossref': 0.9,
            'arxiv': 0.85
        }
        
        source = paper.get('source', 'unknown')
        return source_scores.get(source, 0.5)


class UnifiedSearchService:
    """Main service for searching papers across multiple sources."""
    
    def __init__(self):
        self.scrapers = {
            'arxiv': ArXivScraper,
            'semantic_scholar': SemanticScholarScraper,
            'pubmed': PubMedScraper,
            'crossref': CrossrefScraper,
            'ieee': IEEEScraper,
            'doaj': DOAJScraper
        }
        self.deduplicator = PaperDeduplicator()
        self.ranker = PaperRankingService()
    
    def search_papers(self, query: str, max_results: int = 50, 
                     search_filter: SearchFilter = None, 
                     user=None) -> Dict[str, Any]:
        """
        Perform unified search across multiple sources.
        Returns aggregated and ranked results.
        """
        if search_filter is None:
            search_filter = SearchFilter()
        
        # Default to all sources if none specified
        if not search_filter.sources:
            search_filter.sources = ['arxiv', 'crossref', 'pubmed', 'ieee', 'doaj', 'semantic_scholar']  # All sources
        
        # Create search query record
        search_query = SearchQuery.objects.create(
            user=user,
            query=query,
            filters={
                'sources': search_filter.sources,
                'year_from': search_filter.year_from,
                'year_to': search_filter.year_to,
                'max_results': max_results
            }
        )
        
        # Search across all sources in parallel
        all_papers = []
        results_by_source = {}
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_source = {}
                
                for source in search_filter.sources:
                    if source in self.scrapers:
                        scraper_class = self.scrapers[source]
                        future = executor.submit(
                            self._search_single_source,
                            scraper_class, query, max_results, search_filter
                        )
                        future_to_source[future] = source
                
                # Collect results
                for future in concurrent.futures.as_completed(future_to_source):
                    source = future_to_source[future]
                    try:
                        papers = future.result(timeout=60)  # 60 second timeout per source
                        results_by_source[source] = papers
                        all_papers.extend(papers)
                        logger.info(f"Found {len(papers)} papers from {source}")
                        for i, paper in enumerate(papers[:3]):  # Log first 3 paper titles
                            logger.info(f"  {source} paper {i+1}: {paper.get('title', 'No title')[:50]}")
                    except Exception as e:
                        logger.error(f"Search failed for {source}: {str(e)}")
                        results_by_source[source] = []
            
            # Deduplicate papers
            logger.info(f"Before deduplication: {len(all_papers)} papers total")
            unique_papers = self.deduplicator.deduplicate_papers(all_papers)
            logger.info(f"After deduplication: {len(unique_papers)} papers from {len(all_papers)} total")
            
            # Apply additional filters
            filtered_papers = self._apply_filters(unique_papers, search_filter)
            logger.info(f"After filtering: {len(filtered_papers)} papers")
            
            # Rank papers
            ranked_papers = self.ranker.rank_papers(filtered_papers, query)
            logger.info(f"After ranking: {len(ranked_papers)} papers")
            
            # Limit results
            final_papers = ranked_papers[:max_results]
            
            # Save papers to database and create search results
            saved_papers = []
            logger.info(f"Attempting to save {len(final_papers)} papers to database")
            for i, paper_data in enumerate(final_papers):
                try:
                    with transaction.atomic():
                        # Create paper using base scraper logic - use the appropriate scraper
                        source_name = paper_data.get('source', 'arxiv')
                        if source_name in self.scrapers:
                            scraper = self.scrapers[source_name]()
                        else:
                            scraper = ArXivScraper()  # Fallback
                        paper = scraper._create_paper(paper_data)
                        saved_papers.append(paper)
                        
                        # Create search result
                        from papers.models import SearchResult
                        SearchResult.objects.create(
                            search_query=search_query,
                            paper=paper,
                            relevance_score=paper_data.get('_ranking_score', 0.0),
                            rank=i + 1
                        )
                        logger.info(f"Successfully saved paper {i+1}: {paper.title[:50]}")
                except Exception as e:
                    logger.error(f"Failed to save paper {paper_data.get('title', 'Unknown')}: {str(e)}")
                    continue
            
            # Update search query
            search_query.results_count = len(saved_papers)
            search_query.save()
            
            return {
                'search_query': search_query,
                'papers': saved_papers,
                'total_found': len(all_papers),
                'total_unique': len(unique_papers),
                'total_returned': len(saved_papers),
                'results_by_source': {k: len(v) for k, v in results_by_source.items()},
                'search_time': timezone.now()
            }
            
        except Exception as e:
            logger.error(f"Unified search failed: {str(e)}")
            raise
    
    def _search_single_source(self, scraper_class, query: str, max_results: int, 
                             search_filter: SearchFilter) -> List[Dict[str, Any]]:
        """Search a single source."""
        try:
            scraper = scraper_class()
            
            # Prepare filters for this source
            filters = {}
            if search_filter.year_from:
                if scraper.source_name == 'pubmed':
                    filters['date_from'] = f"{search_filter.year_from}/01/01"
                elif scraper.source_name == 'crossref':
                    filters['from_year'] = search_filter.year_from
            
            if search_filter.year_to:
                if scraper.source_name == 'pubmed':
                    filters['date_to'] = f"{search_filter.year_to}/12/31"
                elif scraper.source_name == 'crossref':
                    filters['to_year'] = search_filter.year_to
            
            # Search papers
            papers = scraper.search_papers(query, max_results, filters)
            
            # Add source information
            for paper in papers:
                paper['source'] = scraper.source_name
            
            return papers
            
        except Exception as e:
            logger.error(f"Failed to search {scraper_class.__name__}: {str(e)}")
            return []
    
    def _apply_filters(self, papers: List[Dict[str, Any]], 
                      search_filter: SearchFilter) -> List[Dict[str, Any]]:
        """Apply additional filters to papers."""
        filtered = papers
        
        # Filter by minimum citations
        if search_filter.min_citations:
            filtered = [p for p in filtered 
                       if (p.get('citation_count') or 0) >= search_filter.min_citations]
        
        # Filter by subject areas
        if search_filter.subject_areas:
            filtered = [p for p in filtered 
                       if any(area.lower() in str(p.get('subject_areas', []) or []).lower() 
                             for area in search_filter.subject_areas)]
        
        # Filter by publication types (if supported)
        if search_filter.publication_types:
            # This would need to be implemented based on how each source provides type info
            pass
        
        return filtered
    
    def get_similar_papers(self, paper_id: str, max_results: int = 20) -> List[Paper]:
        """Find papers similar to a given paper."""
        try:
            paper = Paper.objects.get(id=paper_id)
            
            # Use title and keywords for similarity search
            query_terms = []
            if paper.title:
                query_terms.extend(paper.title.split())
            if paper.keywords:
                if isinstance(paper.keywords, list):
                    query_terms.extend(paper.keywords)
                else:
                    query_terms.extend(str(paper.keywords).split(','))
            
            query = ' '.join(query_terms[:10])  # Limit query length
            
            search_filter = SearchFilter(sources=['semantic_scholar', 'crossref'])
            results = self.search_papers(query, max_results, search_filter)
            
            # Filter out the original paper
            similar_papers = [p for p in results['papers'] if str(p.id) != paper_id]
            
            return similar_papers[:max_results]
            
        except Paper.DoesNotExist:
            logger.error(f"Paper {paper_id} not found")
            return []
        except Exception as e:
            logger.error(f"Failed to find similar papers: {str(e)}")
            return []