#!/usr/bin/env python3
"""
English Integration Test Script
测试英文分析端到端流程
"""

def test_english_resource():
    """测试english_resource模块功能"""
    print("=== Testing english_resource module ===")
    
    try:
        import english_resource
        
        # 测试文本
        test_text = "We don't offer a refund for 10k upgrades (5%)."
        print(f"Test text: {test_text}")
        
        # 测试normalize_and_tokenize_en
        tokens = english_resource.normalize_and_tokenize_en(test_text)
        print(f"Tokens: {tokens}")
        
        # 测试expand_phrases_and_synonyms_en
        expanded = english_resource.expand_phrases_and_synonyms_en(tokens)
        print(f"Expanded: {expanded}")
        
        # 测试detect_negation_bigrams_en
        negations = english_resource.detect_negation_bigrams_en(tokens)
        print(f"Negations: {negations}")
        
        # 测试权重
        weights = english_resource.get_en_language_weights()
        print(f"Weights: {weights}")
        
        print("✅ english_resource tests passed")
        return True
        
    except Exception as e:
        print(f"❌ english_resource test failed: {e}")
        return False

def test_config_weights():
    """测试config中的权重配置"""
    print("\n=== Testing config weights ===")
    
    try:
        from config import get_similarity_weights
        
        # 测试各语言权重
        for lang in ['MAN', 'CAN', 'ENG']:
            weights = get_similarity_weights(lang)
            print(f"{lang} weights: {weights}")
        
        print("✅ config weights tests passed")
        return True
        
    except Exception as e:
        print(f"❌ config weights test failed: {e}")
        return False

def test_mandarin_analyzer_eng():
    """测试Mandarin analyzer处理英文"""
    print("\n=== Testing Mandarin analyzer with ENG ===")
    
    try:
        import sys
        sys.path.append('Mandarin')
        import improved_call_coverage_checker_M as mandarin_analyzer
        
        # 创建analyzer实例
        checker = mandarin_analyzer.SbertCallCoverageChecker('./multilingual_sbert/paraphrase-multilingual-MiniLM-L12-v2')
        
        # 设置为英文模式
        checker.current_language = 'ENG'
        
        # 测试预处理
        test_text = "We don't offer a refund for 10k upgrades (5%)."
        processed = checker.preprocess_text(test_text)
        print(f"Preprocessed: '{test_text}' -> '{processed}'")
        
        # 测试分词
        tokens = checker.tokenize_text(test_text)
        print(f"Tokenized: {tokens}")
        
        # 测试关键词扩展
        expanded = checker.expand_keywords(test_text)
        print(f"Expanded: {expanded}")
        
        print("✅ Mandarin analyzer ENG tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Mandarin analyzer ENG test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting English Integration Tests")
    
    results = []
    results.append(test_english_resource())
    results.append(test_config_weights())
    results.append(test_mandarin_analyzer_eng())
    
    print(f"\n📊 Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 All tests passed! English integration is ready.")
    else:
        print("❌ Some tests failed. Please check the errors above.")