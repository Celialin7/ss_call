"""
============================================================
English Resource Module for Call Coverage Analysis
============================================================

This module provides English-specific text processing capabilities for the call coverage analysis system.
It implements lazy loading with fallback strategies to ensure the system runs even when optional dependencies are missing.

Key Features:
- Lazy loading of optional dependencies (nltk, contractions, flashtext)
- Built-in fallback resources when dependencies are unavailable
- Contract expansion (don't -> do not, won't -> will not)
- Phrase/synonym normalization (KPI variants, refund variants)
- Negation bigram detection (not_refund, no_warranty)
- Minimal stopwords and business keywords

============================================================
"""

import re
import unicodedata
from typing import List, Set, Dict, Iterable, Any
import warnings

# Global flags for dependency availability (lazy loaded)
_CONTRACTIONS_AVAILABLE = None
_GENSIM_AVAILABLE = None
_FLASHTEXT_AVAILABLE = None
_DEPENDENCIES_CHECKED = False

# Built-in fallback resources (minimal sets)
_BUILTIN_STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it',
    'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with', 'you', 'your', 'we', 'our',
    'this', 'these', 'those', 'they', 'them', 'their', 'have', 'had', 'can', 'could', 'would',
    'should', 'may', 'might', 'must', 'shall', 'do', 'does', 'did', 'am', 'been', 'being'
}

_BUILTIN_CONTRACTIONS = {
    "don't": "do not", "won't": "will not", "can't": "can not", "couldn't": "could not",
    "wouldn't": "would not", "shouldn't": "should not", "mustn't": "must not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "I'm": "I am", "you're": "you are", "he's": "he is", "she's": "she is", "it's": "it is",
    "we're": "we are", "they're": "they are", "I've": "I have", "you've": "you have",
    "we've": "we have", "they've": "they have", "I'll": "I will", "you'll": "you will",
    "he'll": "he will", "she'll": "she will", "we'll": "we will", "they'll": "they will"
}

_BUILTIN_SYNONYMS = {
    'kpi': ['key performance indicator', 'performance indicator', 'key indicator'],
    'refund': ['refunds', 'refunding', 'refunded', 'reimbursement', 'reimburse'],
    'upgrade': ['upgrades', 'upgrading', 'upgraded', 'enhancement', 'improve'],
    'payment': ['payments', 'pay', 'paying', 'paid', 'transaction'],
    'warranty': ['warranties', 'guarantee', 'guarantees', 'coverage'],
    'customer': ['customers', 'client', 'clients', 'user', 'users'],
    'service': ['services', 'support', 'assistance', 'help']
}

_IMPORTANT_KEYWORDS_EN = {
    'refund', 'warranty', 'guarantee', 'payment', 'upgrade', 'service', 'customer', 'support',
    'price', 'cost', 'fee', 'charge', 'billing', 'account', 'contract', 'agreement',
    'terms', 'conditions', 'policy', 'coverage', 'benefit', 'feature', 'product',
    'kpi', 'performance', 'quality', 'satisfaction', 'experience', 'issue', 'problem'
}

_NEGATION_WORDS = {'not', 'no', 'without', 'never', 'none', 'nothing', 'neither', 'nor'}

def _check_dependencies():
    """Check availability of optional dependencies with one-time warning."""
    global _CONTRACTIONS_AVAILABLE, _GENSIM_AVAILABLE, _FLASHTEXT_AVAILABLE, _DEPENDENCIES_CHECKED
    
    if _DEPENDENCIES_CHECKED:
        return
    
    # Check contractions
    try:
        import contractions
        _CONTRACTIONS_AVAILABLE = True
    except ImportError:
        _CONTRACTIONS_AVAILABLE = False
        warnings.warn("contractions package not available, using built-in minimal contractions", UserWarning)
    
    # Check gensim
    try:
        from gensim.parsing.preprocessing import STOPWORDS
        _GENSIM_AVAILABLE = True
    except ImportError:
        _GENSIM_AVAILABLE = False
        warnings.warn("gensim package not available, using built-in minimal stopwords", UserWarning)
    
    # Check flashtext
    try:
        from flashtext import KeywordProcessor
        _FLASHTEXT_AVAILABLE = True
    except ImportError:
        _FLASHTEXT_AVAILABLE = False
        warnings.warn("flashtext package not available, using built-in minimal synonym mapping", UserWarning)
    
    _DEPENDENCIES_CHECKED = True

def normalize_and_tokenize_en(text: str) -> List[str]:
    """
    Normalize and tokenize English text with contract expansion and basic preprocessing.
    Uses gensim stopwords but preserves negation words (not, no, without, never) for negation bigram detection.
    
    Args:
        text: Input English text
        
    Returns:
        List of normalized tokens, filtered for stopwords and single characters
    """
    _check_dependencies()
    
    # Step 1: NFKC normalization
    normalized = unicodedata.normalize('NFKC', str(text))
    
    # Step 2: Contract expansion
    if _CONTRACTIONS_AVAILABLE:
        try:
            import contractions
            expanded = contractions.fix(normalized)
        except Exception:
            # Fallback to built-in contractions
            expanded = _expand_contractions_builtin(normalized)
    else:
        expanded = _expand_contractions_builtin(normalized)
    
    # Step 3: Basic tokenization (whitespace + common punctuation)
    # Preserve numeric patterns like $1,200, 5%, 10k, 10x
    tokens = re.findall(r'\$[\d,]+\.?\d*|[\d,]+\.?\d*%|\d+[kKxX]|\w+', expanded.lower())
    
    # Step 4: Filter stopwords and single characters, but keep numbers and symbols
    # Use gensim stopwords but preserve negation words for negation bigram detection
    if _GENSIM_AVAILABLE:
        try:
            from gensim.parsing.preprocessing import STOPWORDS
            # Remove negation words from stopwords to preserve them for negation bigrams
            english_stopwords = STOPWORDS - {'not', 'no', 'without', 'never'}
        except Exception:
            english_stopwords = _BUILTIN_STOPWORDS
    else:
        english_stopwords = _BUILTIN_STOPWORDS
    
    # Filter tokens: remove stopwords and single letters, but keep numbers/symbols
    filtered_tokens = []
    for token in tokens:
        if len(token) == 1 and token.isalpha():  # Remove single letters only
            continue
        if token in english_stopwords:
            continue
        filtered_tokens.append(token)
    
    return filtered_tokens

def _expand_contractions_builtin(text: str) -> str:
    """Expand contractions using built-in mapping."""
    result = text
    for contraction, expansion in _BUILTIN_CONTRACTIONS.items():
        # Case-insensitive replacement
        result = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, result, flags=re.IGNORECASE)
    return result

def expand_phrases_and_synonyms_en(tokens: Iterable[str]) -> Set[str]:
    """
    Expand tokens with phrase normalization and synonym mapping.
    
    Args:
        tokens: Input tokens to expand
        
    Returns:
        Set of expanded tokens including original tokens and their synonyms
    """
    _check_dependencies()
    
    expanded_tokens = set(tokens)
    
    if _FLASHTEXT_AVAILABLE:
        try:
            from flashtext import KeywordProcessor
            # Use flashtext for efficient phrase matching
            processor = KeywordProcessor()
            for canonical, variants in _BUILTIN_SYNONYMS.items():
                for variant in variants:
                    processor.add_keyword(variant, canonical)
            
            # Process each token
            for token in tokens:
                matches = processor.extract_keywords(token)
                expanded_tokens.update(matches)
        except Exception:
            # Fallback to simple mapping
            _expand_synonyms_builtin(tokens, expanded_tokens)
    else:
        _expand_synonyms_builtin(tokens, expanded_tokens)
    
    return expanded_tokens

def _expand_synonyms_builtin(tokens: Iterable[str], expanded_tokens: Set[str]):
    """Expand synonyms using built-in mapping."""
    for token in tokens:
        token_lower = token.lower()
        for canonical, variants in _BUILTIN_SYNONYMS.items():
            if token_lower in variants or token_lower == canonical:
                expanded_tokens.add(canonical)
                expanded_tokens.update(variants)

def detect_negation_bigrams_en(tokens: Iterable[str]) -> Set[str]:
    """
    Detect negation patterns and create negation bigrams.
    
    Args:
        tokens: Input tokens to analyze
        
    Returns:
        Set of negation bigrams (e.g., 'not_refund', 'no_warranty')
    """
    tokens_list = list(tokens)
    negation_bigrams = set()
    
    for i in range(len(tokens_list) - 1):
        current_token = tokens_list[i].lower()
        next_token = tokens_list[i + 1].lower()
        
        # Check if current token is a negation word and next token is important
        if current_token in _NEGATION_WORDS and next_token in _IMPORTANT_KEYWORDS_EN:
            negation_bigram = f"{current_token}_{next_token}"
            negation_bigrams.add(negation_bigram)
    
    return negation_bigrams

def get_en_business_resources() -> Dict[str, Any]:
    """
    Get English business resources including keywords and synonym mappings.
    
    Returns:
        Dictionary containing important_keywords_EN and synonyms_map_EN
    """
    return {
        'important_keywords_EN': _IMPORTANT_KEYWORDS_EN,
        'synonyms_map_EN': _BUILTIN_SYNONYMS,
        'negation_words_EN': _NEGATION_WORDS
    }

def get_en_language_weights() -> Dict[str, float]:
    """
    Get English language-specific weights for scoring.
    
    Returns:
        Dictionary with scoring weights for English analysis
    """
    return {
        'SBERT': 0.60,
        'expanded_overlap': 0.13,
        'rouge_l': 0.12,
        'keyword_coverage': 0.10,
        'fuzzy_similarity': 0.05  # If used in the future
    }

def conservative_stem_tokens(tokens: Iterable[str], min_len: int = 5, whitelist: set = None) -> List[str]:
    """
    Conservative stemming for English tokens using NLTK SnowballStemmer.
    Only stems tokens longer than min_len or in whitelist. Falls back to no-op if NLTK unavailable.
    
    Args:
        tokens: Input tokens to stem
        min_len: Minimum token length to apply stemming (default: 5)
        whitelist: Set of tokens to always stem regardless of length
        
    Returns:
        List of stemmed tokens, or original tokens if NLTK unavailable
    """
    if whitelist is None:
        whitelist = set()
    
    tokens_list = list(tokens)
    
    # Try to use NLTK SnowballStemmer
    try:
        from nltk.stem import SnowballStemmer
        stemmer = SnowballStemmer('english')
        
        stemmed_tokens = []
        for token in tokens_list:
            # Apply stemming only if token length >= min_len or in whitelist
            if len(token) >= min_len or token.lower() in whitelist:
                try:
                    stemmed_token = stemmer.stem(token)
                    stemmed_tokens.append(stemmed_token)
                except Exception:
                    # If stemming fails for individual token, keep original
                    stemmed_tokens.append(token)
            else:
                # Keep short tokens unchanged
                stemmed_tokens.append(token)
        
        return stemmed_tokens
        
    except ImportError:
        # NLTK not available, return original tokens (no-op)
        return tokens_list
    except Exception:
        # Any other error, return original tokens (no-op)
        return tokens_list