"""
Intent Classification for Natural Language Queries.

This module classifies user intents from natural language queries to determine
the appropriate SQL operation type and query structure.
"""

import logging
import re
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Enumeration of possible query intents."""
    
    # Data Retrieval
    SELECT_ALL = "select_all"
    SELECT_FILTERED = "select_filtered" 
    SELECT_AGGREGATED = "select_aggregated"
    SELECT_JOINED = "select_joined"
    SELECT_ORDERED = "select_ordered"
    SELECT_GROUPED = "select_grouped"
    
    # Data Modification
    INSERT_SINGLE = "insert_single"
    INSERT_BULK = "insert_bulk"
    UPDATE_FILTERED = "update_filtered"
    DELETE_FILTERED = "delete_filtered"
    
    # Schema Operations
    CREATE_TABLE = "create_table"
    DROP_TABLE = "drop_table"
    ALTER_TABLE = "alter_table"
    CREATE_INDEX = "create_index"
    
    # Analysis Operations
    COUNT_RECORDS = "count_records"
    CALCULATE_STATISTICS = "calculate_statistics"
    FIND_DUPLICATES = "find_duplicates"
    ANALYZE_DISTRIBUTION = "analyze_distribution"
    
    # Utility Operations
    DESCRIBE_TABLE = "describe_table"
    EXPLAIN_QUERY = "explain_query"
    SHOW_TABLES = "show_tables"
    
    # Unknown/Ambiguous
    UNKNOWN = "unknown"


@dataclass
class IntentClassification:
    """Result of intent classification."""
    intent: QueryIntent
    confidence: float
    supporting_patterns: List[str]
    suggested_sql_template: Optional[str] = None
    required_entities: List[str] = None


class IntentClassifier:
    """Classifies natural language queries to determine SQL intent."""
    
    def __init__(self):
        """Initialize the intent classifier."""
        self.nlp = self._load_spacy_model()
        
        # Intent patterns and keywords
        self.intent_patterns = self._build_intent_patterns()
        
        # ML classifier (initialized when training data is available)
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        self.classifier = MultinomialNB()
        self._is_trained = False
    
    def _load_spacy_model(self):
        """Load spaCy NLP model."""
        try:
            # Try to load English model
            return spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy English model not found. Using basic processing.")
            # Return None if model not available
            return None
    
    def _build_intent_patterns(self) -> Dict[QueryIntent, Dict[str, any]]:
        """Build patterns for intent classification."""
        return {
            # Data Retrieval Intents
            QueryIntent.SELECT_ALL: {
                "keywords": ["show", "display", "list", "get", "retrieve", "all", "everything"],
                "patterns": [
                    r"show\s+(me\s+)?all",
                    r"list\s+all",
                    r"get\s+all",
                    r"display\s+all",
                    r"retrieve\s+all"
                ],
                "sql_template": "SELECT * FROM {table}",
                "entities": ["table"]
            },
            
            QueryIntent.SELECT_FILTERED: {
                "keywords": ["where", "filter", "find", "search", "with", "having"],
                "patterns": [
                    r"where\s+",
                    r"filter\s+by",
                    r"find\s+.*\s+where",
                    r"search\s+for",
                    r"with\s+.*\s+(equals?|is|contains?)"
                ],
                "sql_template": "SELECT * FROM {table} WHERE {condition}",
                "entities": ["table", "condition"]
            },
            
            QueryIntent.SELECT_AGGREGATED: {
                "keywords": ["count", "sum", "average", "avg", "max", "min", "total"],
                "patterns": [
                    r"count\s+",
                    r"how\s+many",
                    r"total\s+",
                    r"sum\s+of",
                    r"average\s+",
                    r"maximum\s+",
                    r"minimum\s+"
                ],
                "sql_template": "SELECT {aggregate_function}({column}) FROM {table}",
                "entities": ["table", "column", "aggregate_function"]
            },
            
            QueryIntent.SELECT_JOINED: {
                "keywords": ["join", "combine", "merge", "together", "from", "and"],
                "patterns": [
                    r"join\s+",
                    r"combine\s+.*\s+with",
                    r"from\s+.*\s+and\s+",
                    r"merge\s+.*\s+with"
                ],
                "sql_template": "SELECT * FROM {table1} JOIN {table2} ON {condition}",
                "entities": ["table1", "table2", "condition"]
            },
            
            QueryIntent.SELECT_ORDERED: {
                "keywords": ["sort", "order", "arrange", "by", "ascending", "descending"],
                "patterns": [
                    r"sort\s+by",
                    r"order\s+by",
                    r"arrange\s+by",
                    r"sorted\s+by"
                ],
                "sql_template": "SELECT * FROM {table} ORDER BY {column}",
                "entities": ["table", "column"]
            },
            
            QueryIntent.SELECT_GROUPED: {
                "keywords": ["group", "grouped", "by", "category", "categories"],
                "patterns": [
                    r"group\s+by",
                    r"grouped\s+by",
                    r"by\s+category",
                    r"categorize\s+by"
                ],
                "sql_template": "SELECT {column}, COUNT(*) FROM {table} GROUP BY {column}",
                "entities": ["table", "column"]
            },
            
            # Data Modification Intents
            QueryIntent.INSERT_SINGLE: {
                "keywords": ["add", "insert", "create", "new", "record"],
                "patterns": [
                    r"add\s+.*\s+record",
                    r"insert\s+",
                    r"create\s+.*\s+record",
                    r"add\s+new"
                ],
                "sql_template": "INSERT INTO {table} ({columns}) VALUES ({values})",
                "entities": ["table", "columns", "values"]
            },
            
            QueryIntent.UPDATE_FILTERED: {
                "keywords": ["update", "change", "modify", "set", "edit"],
                "patterns": [
                    r"update\s+",
                    r"change\s+.*\s+to",
                    r"modify\s+",
                    r"set\s+.*\s+to",
                    r"edit\s+"
                ],
                "sql_template": "UPDATE {table} SET {column} = {value} WHERE {condition}",
                "entities": ["table", "column", "value", "condition"]
            },
            
            QueryIntent.DELETE_FILTERED: {
                "keywords": ["delete", "remove", "drop", "eliminate"],
                "patterns": [
                    r"delete\s+",
                    r"remove\s+",
                    r"drop\s+.*\s+where",
                    r"eliminate\s+"
                ],
                "sql_template": "DELETE FROM {table} WHERE {condition}",
                "entities": ["table", "condition"]
            },
            
            # Analysis Intents
            QueryIntent.COUNT_RECORDS: {
                "keywords": ["count", "how many", "number of", "total"],
                "patterns": [
                    r"count\s+",
                    r"how\s+many\s+",
                    r"number\s+of\s+",
                    r"total\s+.*\s+records"
                ],
                "sql_template": "SELECT COUNT(*) FROM {table}",
                "entities": ["table"]
            },
            
            QueryIntent.CALCULATE_STATISTICS: {
                "keywords": ["statistics", "stats", "summary", "analyze", "distribution"],
                "patterns": [
                    r"statistics\s+",
                    r"stats\s+for",
                    r"summary\s+of",
                    r"analyze\s+",
                    r"distribution\s+of"
                ],
                "sql_template": "SELECT AVG({column}), MIN({column}), MAX({column}) FROM {table}",
                "entities": ["table", "column"]
            },
            
            # Utility Intents
            QueryIntent.DESCRIBE_TABLE: {
                "keywords": ["describe", "structure", "schema", "columns", "fields"],
                "patterns": [
                    r"describe\s+",
                    r"structure\s+of",
                    r"schema\s+of",
                    r"columns\s+in",
                    r"fields\s+in"
                ],
                "sql_template": "DESCRIBE {table}",
                "entities": ["table"]
            },
            
            QueryIntent.SHOW_TABLES: {
                "keywords": ["show tables", "list tables", "available tables", "tables"],
                "patterns": [
                    r"show\s+tables",
                    r"list\s+tables",
                    r"available\s+tables",
                    r"what\s+tables"
                ],
                "sql_template": "SHOW TABLES",
                "entities": []
            }
        }
    
    def classify_intent(self, query: str) -> IntentClassification:
        """
        Classify the intent of a natural language query.
        
        Args:
            query: Natural language query
            
        Returns:
            Intent classification result
        """
        query_lower = query.lower().strip()
        
        # Score each intent
        intent_scores = {}
        supporting_patterns = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            matched_patterns = []
            
            # Check keyword matches
            keywords = patterns.get("keywords", [])
            keyword_matches = sum(1 for keyword in keywords if keyword in query_lower)
            score += keyword_matches * 0.5
            
            # Check regex pattern matches
            regex_patterns = patterns.get("patterns", [])
            for pattern in regex_patterns:
                if re.search(pattern, query_lower):
                    score += 1.0
                    matched_patterns.append(pattern)
            
            if score > 0:
                intent_scores[intent] = score
                supporting_patterns[intent] = matched_patterns
        
        # Use ML classifier if trained
        if self._is_trained:
            try:
                ml_scores = self._get_ml_scores(query)
                # Combine rule-based and ML scores
                for intent, ml_score in ml_scores.items():
                    if intent in intent_scores:
                        intent_scores[intent] = (intent_scores[intent] * 0.7) + (ml_score * 0.3)
                    else:
                        intent_scores[intent] = ml_score * 0.3
            except Exception as e:
                logger.warning(f"ML classification failed: {e}")
        
        # Find best intent
        if not intent_scores:
            return IntentClassification(
                intent=QueryIntent.UNKNOWN,
                confidence=0.0,
                supporting_patterns=[],
                required_entities=["query_analysis_needed"]
            )
        
        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        intent = best_intent[0]
        raw_score = best_intent[1]
        
        # Normalize confidence score
        max_possible_score = 3.0  # Rough estimate
        confidence = min(raw_score / max_possible_score, 1.0)
        
        # Get template and entities
        pattern_info = self.intent_patterns.get(intent, {})
        
        return IntentClassification(
            intent=intent,
            confidence=confidence,
            supporting_patterns=supporting_patterns.get(intent, []),
            suggested_sql_template=pattern_info.get("sql_template"),
            required_entities=pattern_info.get("entities", [])
        )
    
    def train_classifier(self, training_data: List[Tuple[str, QueryIntent]]) -> None:
        """
        Train the ML classifier with labeled data.
        
        Args:
            training_data: List of (query, intent) pairs
        """
        if len(training_data) < 10:
            logger.warning("Insufficient training data for ML classifier")
            return
        
        try:
            queries, intents = zip(*training_data)
            
            # Vectorize queries
            X = self.vectorizer.fit_transform(queries)
            y = [intent.value for intent in intents]
            
            # Train classifier
            self.classifier.fit(X, y)
            self._is_trained = True
            
            logger.info(f"Trained ML classifier with {len(training_data)} samples")
            
        except Exception as e:
            logger.error(f"Failed to train ML classifier: {e}")
    
    def _get_ml_scores(self, query: str) -> Dict[QueryIntent, float]:
        """Get ML classification scores."""
        try:
            X = self.vectorizer.transform([query])
            probabilities = self.classifier.predict_proba(X)[0]
            classes = self.classifier.classes_
            
            scores = {}
            for i, class_name in enumerate(classes):
                try:
                    intent = QueryIntent(class_name)
                    scores[intent] = probabilities[i]
                except ValueError:
                    # Skip unknown intent values
                    pass
            
            return scores
            
        except Exception as e:
            logger.error(f"ML scoring failed: {e}")
            return {}
    
    def get_intent_suggestions(self, query: str, top_k: int = 3) -> List[IntentClassification]:
        """
        Get multiple intent suggestions ranked by confidence.
        
        Args:
            query: Natural language query
            top_k: Number of suggestions to return
            
        Returns:
            List of intent classifications ordered by confidence
        """
        query_lower = query.lower().strip()
        all_scores = {}
        all_patterns = {}
        
        # Score all intents
        for intent, patterns in self.intent_patterns.items():
            score = 0
            matched_patterns = []
            
            # Keyword scoring
            keywords = patterns.get("keywords", [])
            keyword_matches = sum(1 for keyword in keywords if keyword in query_lower)
            score += keyword_matches * 0.5
            
            # Pattern scoring
            regex_patterns = patterns.get("patterns", [])
            for pattern in regex_patterns:
                if re.search(pattern, query_lower):
                    score += 1.0
                    matched_patterns.append(pattern)
            
            all_scores[intent] = score
            all_patterns[intent] = matched_patterns
        
        # Sort by score and take top_k
        sorted_intents = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        suggestions = []
        for intent, score in sorted_intents:
            if score > 0:  # Only include intents with positive scores
                max_score = 3.0
                confidence = min(score / max_score, 1.0)
                
                pattern_info = self.intent_patterns.get(intent, {})
                
                suggestions.append(IntentClassification(
                    intent=intent,
                    confidence=confidence,
                    supporting_patterns=all_patterns.get(intent, []),
                    suggested_sql_template=pattern_info.get("sql_template"),
                    required_entities=pattern_info.get("entities", [])
                ))
        
        return suggestions
    
    def analyze_query_complexity(self, query: str) -> Dict[str, any]:
        """
        Analyze the complexity of a natural language query.
        
        Args:
            query: Natural language query
            
        Returns:
            Complexity analysis results
        """
        query_lower = query.lower().strip()
        
        complexity_indicators = {
            "joins": len(re.findall(r'\b(join|combine|merge|with|and)\b', query_lower)),
            "conditions": len(re.findall(r'\b(where|having|if|when)\b', query_lower)),
            "aggregations": len(re.findall(r'\b(count|sum|avg|max|min|average)\b', query_lower)),
            "grouping": len(re.findall(r'\b(group|category|by)\b', query_lower)),
            "sorting": len(re.findall(r'\b(sort|order|arrange)\b', query_lower)),
            "subqueries": len(re.findall(r'\b(in|exists|any|all)\b', query_lower)),
            "word_count": len(query.split()),
            "question_words": len(re.findall(r'\b(what|where|when|who|how|which)\b', query_lower))
        }
        
        # Calculate complexity score
        complexity_score = (
            complexity_indicators["joins"] * 2 +
            complexity_indicators["conditions"] * 1 +
            complexity_indicators["aggregations"] * 1.5 +
            complexity_indicators["grouping"] * 1.5 +
            complexity_indicators["sorting"] * 1 +
            complexity_indicators["subqueries"] * 3 +
            (complexity_indicators["word_count"] / 10)  # Normalize word count
        )
        
        # Classify complexity level
        if complexity_score < 2:
            complexity_level = "simple"
        elif complexity_score < 5:
            complexity_level = "moderate"
        else:
            complexity_level = "complex"
        
        return {
            "complexity_score": round(complexity_score, 2),
            "complexity_level": complexity_level,
            "indicators": complexity_indicators,
            "estimated_sql_clauses": self._estimate_sql_clauses(complexity_indicators)
        }
    
    def _estimate_sql_clauses(self, indicators: Dict[str, int]) -> List[str]:
        """Estimate which SQL clauses will be needed."""
        clauses = ["SELECT", "FROM"]
        
        if indicators["joins"] > 0:
            clauses.append("JOIN")
        
        if indicators["conditions"] > 0:
            clauses.append("WHERE")
        
        if indicators["grouping"] > 0:
            clauses.append("GROUP BY")
        
        if indicators["aggregations"] > 0 and indicators["grouping"] > 0:
            clauses.append("HAVING")
        
        if indicators["sorting"] > 0:
            clauses.append("ORDER BY")
        
        return clauses