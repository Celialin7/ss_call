"""
============================================================
Dictionaries and Configuration for Call Coverage Analysis
============================================================

This file contains all the dictionaries and configuration data used by the call coverage analysis system.
It can be modified by the dynamic script analysis tool to update term importance weights.

Structure:
- mandarin_synonyms: Synonym mappings for Mandarin business terms
- error_patterns: Speech-to-text error correction patterns
- important_keywords: Business-critical keywords for analysis
- term_importance: Term importance weights (can be updated by dynamic analysis)
- stopwords: Words to filter out from analysis
- dynamic_term_importance_cache: Cache for dynamic weights (updated by analysis tool)
- script_hash_cache: Hash to detect script changes

============================================================
"""

import jieba
import hashlib

# ==========================================================
# Mandarin Synonyms Dictionary
# ==========================================================
mandarin_synonyms = {
    # Identity verification
    '客户': ['客人', '顾客', '客户', '陈先生', '先生'],
    '身份': ['身份证', 'ID', '证件', '身份证号码'],
    '核实': ['确认', '核对', '验证', 'check', '保障'],
    '身份证': ['身份证', 'ID', '证件'],
    # Product information
    '产品': ['product', '投资产品', '基金', '股票基金', '全球科技股票基金'],
    '信息': ['资料', '详情', '内容', '记录'],
    '确认': ['确认', '核对', '验证', '是不是'],
    # Financial assessment
    '财务': ['收入', '资产', '经济状况', '投资金额'],
    '状况': ['情况', '状况', '状态'],
    '评估': ['评估', '了解', '确认'],
    # Risk assessment
    '风险': ['风险评估', '风险取向', '风险承受能力'],
    '取向': ['取向', '承受能力', '偏好'],
    '问卷': ['问卷', '评估', '测试'],
    # Transaction details
    '交易': ['买卖', '投资', '购买', '认购'],
    '价格': ['价钱', '费用', '成本', '投资金额'],
    '数量': ['份额', '单位', '数量', '金额'],
    '人民币': ['人民币', '块钱', 'CNY'],
    # Important declarations
    '重要': ['重要事项', '重要声明', '注意事项'],
    '事项': ['事项', '声明', '条款'],
    '声明': ['声明', '条款', '事项'],
    # Final authorization
    '授权': ['同意', '确认', '授权', 'OK'],
    '最终': ['最后', '最终', '最后确认'],
    '同意': ['同意', '确认', 'OK', '好的']
}

# ==========================================================
# Speech-to-Text Error Patterns
# ==========================================================
error_patterns = {
    '的': ['的', '地', '得'],
    '是': ['是', '系', '对'],
    '不': ['不', '没', '别'],
    '了': ['了', '过', '着'],
    '这样': ['这样', '这么', '这种'],
    '来': ['来', '到'],
    '回': ['回', '返'],
    '完': ['完', '了'],
    '块': ['块', '元', '块钱']
}

# ==========================================================
# Business-Critical Keywords
# ==========================================================
important_keywords = {
    '客户', '身份', '核实', '产品', '信息', '财务', '风险', 
    '交易', '价格', '数量', '重要', '授权', '投资', '基金',
    '确认', '身份证', '投资金额', '认购', '人民币', '收入',
    '流动资产', '风险评估', '风险取向', '问卷', '股票基金',
    '全球科技', '投资产品', '客户经理', '身份证号码', '年度收入',
    '财务状况', '投资目标', '现金', '股票', '总值', '年收入',
    '收入范围', '资产状况', '经济状况', '投资能力'
}

# ==========================================================
# Term Importance Weights (Manual + Dynamic)
# ==========================================================
# Initial manual weights (can be updated by dynamic analysis)
term_importance = {
    '身份证号码': 4.159,
    '身份': 4.159,
    '收入': 4.159,
    '流动资产': 4.159,
    '问卷': 4.159,
    '投资': 4.159,
    '价格': 4.159,
    '认购': 4.159,
    '数量': 4.159,
    '授权': 4.159,
    '产品': 2.773,
    '确认': 2.773,
    '风险': 2.773,
    '交易': 2.773,
    '全名': 2.079,
    '核对': 2.079,
    '提供': 2.079,
    '名称': 2.079,
    '将要': 2.079,
    '处理': 2.079,
    '编号': 2.079,
    '状况': 2.079,
    '年度': 2.079,
    '一次': 2.079,
    '评估': 2.079,
    '结果': 2.079,
    '重温': 2.079,
    '确保': 2.079,
    '取向': 2.079,
    '执行': 2.079,
    '最终': 2.079,
    '买入': 2.079,
    '参考': 2.079,
    '相等': 2.079,
    '决定': 2.079,
    '金额': 2.079,
    '播放': 2.079,
    '披露': 2.079,
    '条款': 2.079,
    '录音': 2.079,
    '包含': 2.079,
    '接下': 2.079,
    '系统': 2.079,
    '说出': 2.079,
    '清楚': 2.079,
    '进行': 2.079,
    '同意': 2.079,
    '单位': 1.386,
    '年度收入': 3.0,
    '财务状况': 3.0,
    '风险评估': 3.0,
    '投资金额': 2.5,
    '股票基金': 2.5,
    '身份证': 2.5,
    '基金': 2.0,
    '是': 0.5,
}

# ==========================================================
# Stopwords (Words to Filter Out)
# ==========================================================
stopwords = set(jieba.get_stop_words()) | {
    '的', '了', '这样', '来', '回', '完', '块', '我', '是', '不',
    '好', '都', '有', '没有', '就是', '就', '会', '要', '想', '可以',
    '应该', '可能', '一定', '当然', '其实', '所以', '因为', '如果',
    '然后', '之后', '之前', '现在', '今天', '明天', '昨天',
    # Function words that shouldn't be considered meaningful keywords
    '这个', '这些', '那个', '那些',  # this, these, that, those
    '一下', '一点', '嘛', '啦', '的',  # a bit, a little, particles
    '目前', '现在', '当前',  # now, currently
    '先生', '小姐', '女士',  # titles (Mr., Miss, Ms.)
    '第一', '第二', '第三', '第四', '第五',  # ordinal numbers
    '其他', '另外', '还有', '同时',  # other, additionally, and, meanwhile
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
    '这样', '这种', '怎么',  # like this, such, how
    '谁', '哪里', '什么时候', '为什么',  # who, where, when, why
    '多少', '几个', '多少个',  # how much, how many
    '这些', '其余', '只是',  # the rest, remaining, only
    # Additional stopwords as requested
    '我们', '中国', '今日', '大约', '一段'
}

# ==========================================================
# Dynamic Analysis Cache (Updated by dynamic_script_analysis.py)
# ==========================================================
dynamic_term_importance_cache = {'身份证号码': 4.1588830833596715, '身份': 4.1588830833596715, '提供': 2.0794415416798357, '全名': 2.0794415416798357, '核对': 2.0794415416798357, '产品': 2.772588722239781, '将要': 2.0794415416798357, '处理': 2.0794415416798357, '名称': 2.0794415416798357, '编号': 2.0794415416798357, '收入': 4.1588830833596715, '流动资产': 4.1588830833596715, '确认': 2.772588722239781, '年度': 2.0794415416798357, '状况': 2.0794415416798357, '问卷': 4.1588830833596715, '投资': 4.1588830833596715, '风险': 2.772588722239781, '重温': 2.0794415416798357, '一次': 2.0794415416798357, '评估': 2.0794415416798357, '结果': 2.0794415416798357, '确保': 2.0794415416798357, '取向': 2.0794415416798357, '价格': 4.1588830833596715, '交易': 2.772588722239781, '参考': 2.0794415416798357, '买入': 2.0794415416798357, '最终': 2.0794415416798357, '执行': 2.0794415416798357, '单位': 1.3862943611198906, '认购': 4.1588830833596715, '数量': 4.1588830833596715, '决定': 2.0794415416798357, '金额': 2.0794415416798357, '相等': 2.0794415416798357, '接下': 2.0794415416798357, '系统': 2.0794415416798357, '播放': 2.0794415416798357, '包含': 2.0794415416798357, '披露': 2.0794415416798357, '条款': 2.0794415416798357, '录音': 2.0794415416798357, '授权': 4.1588830833596715, '清楚': 2.0794415416798357, '说出': 2.0794415416798357, '同意': 2.0794415416798357, '进行': 2.0794415416798357}
script_hash_cache = '69b918d70e0379e6ebb752b714c35d6c'

# ==========================================================
# Utility Functions
# ==========================================================
def calculate_script_hash(required_points_df):
    """Calculate a hash of the script content to detect changes for cache invalidation."""
    script_content = ""
    for _, row in required_points_df.iterrows():
        script_content += str(row['Required_Discussion_Point']) + str(row['Standard_Script'])
    return hashlib.md5(script_content.encode('utf-8')).hexdigest()

def update_term_importance_with_dynamic_weights(dynamic_weights):
    """
    Update term_importance dictionary with dynamic weights from script analysis.
    Replace manual weights with dynamic weights for words that have dynamic weights.
    Keep manual weights for words without dynamic weights.
    """
    global term_importance
    
    # Replace manual weights with dynamic weights where available
    for word, dynamic_weight in dynamic_weights.items():
        term_importance[word] = dynamic_weight
    
    return term_importance

def save_dynamic_weights_to_cache(dynamic_weights, script_hash):
    """Save dynamic weights to cache for future use."""
    global dynamic_term_importance_cache, script_hash_cache
    dynamic_term_importance_cache = dynamic_weights
    script_hash_cache = script_hash

def get_cached_dynamic_weights():
    """Get cached dynamic weights if available."""
    return dynamic_term_importance_cache, script_hash_cache

def clear_dynamic_cache():
    """Clear the dynamic weights cache."""
    global dynamic_term_importance_cache, script_hash_cache
    dynamic_term_importance_cache = {'身份证号码': 4.1588830833596715, '身份': 4.1588830833596715, '提供': 2.0794415416798357, '全名': 2.0794415416798357, '核对': 2.0794415416798357, '产品': 2.772588722239781, '将要': 2.0794415416798357, '处理': 2.0794415416798357, '名称': 2.0794415416798357, '编号': 2.0794415416798357, '收入': 4.1588830833596715, '流动资产': 4.1588830833596715, '确认': 2.772588722239781, '年度': 2.0794415416798357, '状况': 2.0794415416798357, '问卷': 4.1588830833596715, '投资': 4.1588830833596715, '风险': 2.772588722239781, '重温': 2.0794415416798357, '一次': 2.0794415416798357, '评估': 2.0794415416798357, '结果': 2.0794415416798357, '确保': 2.0794415416798357, '取向': 2.0794415416798357, '价格': 4.1588830833596715, '交易': 2.772588722239781, '参考': 2.0794415416798357, '买入': 2.0794415416798357, '最终': 2.0794415416798357, '执行': 2.0794415416798357, '单位': 1.3862943611198906, '认购': 4.1588830833596715, '数量': 4.1588830833596715, '决定': 2.0794415416798357, '金额': 2.0794415416798357, '相等': 2.0794415416798357, '接下': 2.0794415416798357, '系统': 2.0794415416798357, '播放': 2.0794415416798357, '包含': 2.0794415416798357, '披露': 2.0794415416798357, '条款': 2.0794415416798357, '录音': 2.0794415416798357, '授权': 4.1588830833596715, '清楚': 2.0794415416798357, '说出': 2.0794415416798357, '同意': 2.0794415416798357, '进行': 2.0794415416798357}
    script_hash_cache = '69b918d70e0379e6ebb752b714c35d6c'