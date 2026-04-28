"""
============================================================
Dictionaries and Configuration for Call Coverage Analysis
============================================================

This file contains all the dictionaries and configuration data used by the call coverage analysis system.
It can be modified by the dynamic script analysis tool to update term importance weights.

Structure:
- cantonese_synonyms: Synonym mappings for Cantonese business terms
- error_patterns: Speech-to-text error correction patterns
- important_keywords: Business-critical keywords for analysis
- term_importance: Term importance weights (can be updated by dynamic analysis)
- stopwords: Words to filter out from analysis
- dynamic weights: Generated fresh by analysis tool each run

============================================================
"""

import pycantonese as pc
import pandas as pd
import os
from typing import Optional, Dict

# ==========================================================
# Cantonese Synonyms Dictionary
# ==========================================================
cantonese_synonyms = {
    # Identity verification
    '客戶': ['客人', '顧客', '客戶', '陳先生', '陳生'],
    '身份': ['身份證', 'ID', '證件', '身份證號碼'],
    '核實': ['確認', '核對', '驗證', 'check', '保障'],
    '香港身份證': ['身份證', 'ID', '證件'],
    # Product information
    '產品': ['product', '投資產品', '基金', '股票基金', '環球科技股票基金'],
    '信息': ['資料', '詳情', '內容', '紀錄'],
    '確認': ['確認', '核對', '驗證', '係咪'],
    # Financial assessment
    '財務': ['收入', '資產', '經濟狀況', '投資金額'],
    '狀況': ['情況', '狀況', '狀態'],
    '評估': ['評估', '了解', '確認'],
    # Risk assessment
    '風險': ['風險評估', '風險取向', '風險承受能力'],
    '取向': ['取向', '承受能力', '偏好'],
    '問卷': ['問卷', '評估', '測試'],
    # Transaction details
    '交易': ['買賣', '投資', '購買', '認購'],
    '價格': ['價錢', '費用', '成本', '投資金額'],
    '數量': ['份額', '單位', '數量', '金額'],
    '港幣': ['港幣', '港紙', 'HKD'],
    # Important declarations
    '重要': ['重要事項', '重要聲明', '注意事項'],
    '事項': ['事項', '聲明', '條款'],
    '聲明': ['聲明', '條款', '事項'],
    # Final authorization
    '授權': ['同意', '確認', '授權', 'OK'],
    '最終': ['最後', '最終', '最後確認'],
    '同意': ['同意', '確認', 'OK', '好']
}

# ==========================================================
# Speech-to-Text Error Patterns
# ==========================================================
error_patterns = {
    '嘅': ['嘅', '既', '的', '嘅'],
    '係': ['係', '是', '在'],
    '唔': ['唔', '不', '沒'],
    '咗': ['咗', '了', '過'],
    '咁': ['咁', '這樣', '這麼'],
    '嚟': ['嚟', '來'],
    '返': ['返', '回'],
    '曬': ['曬', '了'],
    '蚊': ['蚊', '元', '塊']
}

# ==========================================================
# Business-Critical Keywords
# ==========================================================
important_keywords = {
    '客戶', '身份', '核實', '產品', '信息', '財務', '風險', 
    '交易', '價格', '數量', '重要', '授權', '投資', '基金',
    '確認', '香港身份證', '投資金額', '認購', '港幣', '收入',
    '流動資產', '風險評估', '風險取向', '問卷', '股票基金',
    '環球科技', '投資產品', '客戶經理', '身份證號碼', '年度收入',
    '財務狀況', '投資目標', '現金', '股票', '總值', '年收入',
    '收入範圍', '資產狀況', '經濟狀況', '投資能力'
}

# ==========================================================
# Stopwords for Different Languages
# ==========================================================

# Cantonese stopwords (Traditional Chinese)
cantonese_stopwords = set(pc.stop_words()) | {
    '嘅', '咗', '咁', '嚟', '返', '曬', '蚊', '我', '係', '唔',
    '好', '都', '有', '冇', '係', '就', '會', '要', '想', '可以',
    '應該', '可能', '一定', '當然', '其實', '所以', '因為', '如果',
    '然後', '之後', '之前', '現在', '今天', '明天', '昨天',
    # Function words that shouldn't be considered meaningful keywords
    '呢個', '呢啲', '嗰個', '嗰啲',  # this, these, that, those
    '一下', '少少', '咪', '啦', '㗎',  # a bit, a little, particles
    '目前', '而家', '宜家', '依家',  # now, currently
    '先生', '小姐', '女士',  # titles (Mr., Miss, Ms.)
    '第一', '第二', '第三', '第四', '第五',  # ordinal numbers
    '其他', '另外', '同埋', '同時',  # other, additionally, and, meanwhile
    '包括', '例如', '比如',  # including, for example
    '通常', '一般', '基本',  # usually, generally, basically
    '主要', '重要', '特別',  # mainly, important, special (too generic)
    '完全', '非常', '相當',  # completely, very, quite
    '開始', '結束', '完成',  # start, end, complete (too generic)
    '記住', '知道', '明白', '了解',  # remember, know, understand
    '繼續', '停止', '保持',  # continue, stop, maintain
    '需要', '必須', '應該',  # need, must, should
    '可能', '或者', '如果',  # maybe, or, if
    '但係', '不過', '雖然',  # but, however, although
    '咁樣', '咁嘅', '點樣',  # like this, such, how
    '邊個', '邊度', '幾時', '點解',  # who, where, when, why
    '多少', '幾多', '幾個',  # how much, how many
    '第啲', '其餘', '剩係',  # the rest, remaining, only
    # Additional stopwords as requested
    '我們', '香港', '今日', '大約', '一段'
}

# Mandarin stopwords (Simplified Chinese) - Based on Cantonese equivalents
mandarin_stopwords = {
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
    '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
    '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '可以',
    '应该', '可能', '一定', '当然', '其实', '所以', '因为', '如果',
    '然后', '之后', '之前', '现在', '今天', '明天', '昨天',
    # Function words that shouldn't be considered meaningful keywords
    '这个', '这些', '那个', '那些',  # this, these, that, those
    '一下', '一点', '一些', '啊', '呢',  # a bit, a little, particles
    '目前', '现在',  # now, currently
    '先生', '小姐', '女士',  # titles (Mr., Miss, Ms.)
    '第一', '第二', '第三', '第四', '第五',  # ordinal numbers
    '其他', '另外', '以及', '同时',  # other, additionally, and, meanwhile
    '包括', '例如', '比如',  # including, for example
    '通常', '一般', '基本',  # usually, generally, basically
    '主要', '重要', '特别',  # mainly, important, special (too generic)
    '完全', '非常', '相当',  # completely, very, quite
    '开始', '结束', '完成',  # start, end, complete (too generic)
    '记住', '知道', '明白', '了解',  # remember, know, understand
    '继续', '停止', '保持',  # continue, stop, maintain
    '需要', '必须', '应该',  # need, must, should
    '可能', '或者', '如果',  # maybe, or, if
    '但是', '不过', '虽然',  # but, however, although
    '这样', '那样', '怎样',  # like this, like that, how
    '谁', '哪里', '什么时候', '为什么',  # who, where, when, why
    '多少', '几个', '一些',  # how much, how many, some
    '其余', '剩下', '只是',  # the rest, remaining, only
    # Additional business context stopwords
    '我们', '香港', '今日', '大约', '一段', '客户', '服务',
    # Common mandarin particles and connectors
    '而且', '并且', '然而', '尽管', '除了', '除非',
    '根据', '按照', '通过', '经过', '关于', '对于', '由于', '因此',
    # Time and location words
    '时候', '情况', '地方', '时间', '方面', '过程', '结果',
    # Modal and auxiliary words
    '能够', '愿意', '希望', '打算', '准备', '开始', '继续',
    # Additional stopwords found in company scripts
    '如同', '以前', '首先','除非','做出','任何','以免','','最近','如同',
}

# Function to get appropriate stopwords based on language
def get_stopwords(language='CAN'):
    """
    Get appropriate stopwords for the specified language.

    Args:
        language: 'CAN' for Cantonese, 'MAN' for Mandarin, 'ENG' for English

    Returns:
        set: Stopwords for the specified language
    """
    if language == 'CAN':
        return cantonese_stopwords
    elif language == 'MAN':
        return mandarin_stopwords
    elif language == 'ENG':
        # ENG: 使用gensim英文停用词，统一入口
        try:
            from gensim.parsing.preprocessing import STOPWORDS
            return STOPWORDS
        except ImportError:
            # 兜底：使用基础英文停用词
            return {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with', 'you', 'your', 'we', 'our'}
    else:
        # For unknown languages, use Mandarin as fallback
        return mandarin_stopwords

# For backward compatibility, keep the original stopwords variable pointing to Cantonese
stopwords = cantonese_stopwords

# ==========================================================
# Term Importance Weights (Manual + Dynamic)
# ==========================================================

# ==========================================================
# Term Importance Data Loader (Lazy Loading from External CSV)
# ==========================================================

import pandas as pd
import gzip
import os
from typing import Dict, Optional

# Cache for loaded term importance data
_term_importance_cache: Optional[Dict] = None

def _load_term_importance_data():
    """
    Lazy load term importance data from external CSV.gz file.
    Returns structured dictionary for backward compatibility.
    """
    global _term_importance_cache
    
    if _term_importance_cache is not None:
        return _term_importance_cache
    
    # Import config here to avoid circular imports
    try:
        from config import TERM_IMPORTANCE_DIR, TERM_IMPORTANCE_CSV
        csv_path = os.path.join(TERM_IMPORTANCE_DIR, TERM_IMPORTANCE_CSV)
    except ImportError:
        # Fallback paths if config import fails
        csv_path = os.path.join("Generated/SS_project", "term_importance.csv.gz")
    
    # Initialize empty cache
    _term_importance_cache = {'CAN': {}, 'MAN': {}, 'ENG': {}}
    
    # Try to load CSV data
    if os.path.exists(csv_path):
        try:
            # Load compressed CSV
            df = pd.read_csv(csv_path, compression='gzip')
            
            # Build nested dictionary structure
            for _, row in df.iterrows():
                language = row['language']
                product = row['product']
                point = row['point']
                term = row['term']
                weight = row['weight']
                
                # Initialize nested structure
                if language not in _term_importance_cache:
                    _term_importance_cache[language] = {}
                if product not in _term_importance_cache[language]:
                    _term_importance_cache[language][product] = {}
                if point not in _term_importance_cache[language][product]:
                    _term_importance_cache[language][product][point] = {}
                
                # Store term weight
                _term_importance_cache[language][product][point][term] = weight
                
            print(f"✅ Loaded term importance data from {csv_path}")
            
        except Exception as e:
            print(f"⚠️  Error loading term importance data: {e}")
            print("   Using empty fallback data")
    else:
        print(f"⚠️  Term importance file not found: {csv_path}")
        print("   Using empty fallback data")
    
    return _term_importance_cache

# Backward compatibility: expose as dynamic_term_importance_by_product
@property
def dynamic_term_importance_by_product():
    """Property to maintain backward compatibility"""
    return _load_term_importance_data()

# Make it accessible as module attribute
import sys
sys.modules[__name__].dynamic_term_importance_by_product = dynamic_term_importance_by_product

def get_product_weights(language, product_type):
    """
    Get vocabulary weights for specified language and product type (backward compatibility)
    Args:
        language: "CAN", "MAN", or "ENG"
        product_type: Product name, e.g. "Bond", "Caller Linear Note"
    
    Returns:
        dict: Flattened vocabulary weight dictionary, returns general weights if not found
    """
    data = _load_term_importance_data()
    
    if (language in data and product_type in data[language]):
        # Flatten point-specific weights for backward compatibility
        product_data = data[language][product_type]
        flattened = {}
        for point_weights in product_data.values():
            for word, score in point_weights.items():
                # Keep highest score if word appears in multiple points
                if word not in flattened or score > flattened[word]:
                    flattened[word] = score
        return flattened
    else:
        print(f"⚠️  FALLBACK: Using general term_importance weights. "
              f"Product-specific weights not found for {language}:{product_type}")
        return term_importance  # fallback to general weights

def get_point_specific_weights(language, product_type, point_name):
    """
    Get vocabulary weights for a specific discussion point (NEW: Point-Specific)
    Args:
        language: "CAN", "MAN", or "ENG"
        product_type: Product name, e.g. "Bond", "Caller Linear Note"
        point_name: Discussion point name, e.g. "产品介绍", "风险告知"
    
    Returns:
        dict: Point-specific vocabulary weights, falls back to general weights if not found
    """
    data = _load_term_importance_data()
    
    if (language in data and 
        product_type in data[language] and
        point_name in data[language][product_type]):
        return data[language][product_type][point_name]
    else:
        # Fallback to product-level weights first
        product_weights = get_product_weights(language, product_type)
        if product_weights != term_importance:  # If product weights exist
            return product_weights
        else:
            # Ultimate fallback to general weights
            return term_importance

def detect_product_type_from_script(script_df):
    """
    Detect product type from script DataFrame
    Can be based on keywords or other features in script content
    """
    # TODO: Add keyword-based product detection logic here
    # For now, return None to use fallback weights
    return None

# Initial manual weights (can be updated by dynamic analysis)
term_importance = {
    '基金': 2.197,
    '避免': 1.099,
    '能力': 1.099,
    '承受': 1.099,
    '透明': 1.099,
    '管理': 1.099,
    '收益': 0.405,
    '流動資產': 3.0,
    '年度收入': 3.0,
    '財務狀況': 3.0,
    '風險評估': 3.0,
    '投資金額': 2.5,
    '股票基金': 2.5,
    '香港身份證': 2.5,
    '身份證號碼': 2.5,
    '產品': 2.0,
    '投資': 2.0,
    '交易': 2.0,
    '確認': 1.0,
    '同意': 1.0,
    '係': 0.5,
}

# End of dictionaries.py file

