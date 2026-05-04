#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SBERT模型测试脚本
用于验证模型加载和相似度计算功能
"""

import os
import sys
import time

def test_sbert_model():
    """测试SBERT模型的加载和计算功能"""
    
    # 模型路径
    model_path = "/Users/CeliaLin_1/Desktop/paraphrase-multilingual-MiniLM-L12-v2"
    
    print("=" * 80)
    print("🧪 SBERT模型测试")
    print("=" * 80)
    
    # 检查模型路径
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        return False
    
    print(f"✅ 模型路径存在: {model_path}")
    
    # 检查模型文件
    required_files = [
        "config.json",
        "pytorch_model.bin", 
        "sentence_bert_config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.txt"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} (缺失)")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  缺失文件: {missing_files}")
        print("模型可能不完整")
    
    # 测试句子
    sentence1 = "债券票据的类别为高级无抵押"
    sentence2 = "非常感谢。"
    
    print(f"\n📝 测试句子:")
    print(f"  句子1: {sentence1}")
    print(f"  句子2: {sentence2}")
    
    try:
        print(f"\n🔄 加载SBERT模型...")
        start_time = time.time()
        
        # 导入必要的库
        from sentence_transformers import SentenceTransformer
        from sentence_transformers import util
        
        # 加载模型
        model = SentenceTransformer(model_path)
        load_time = time.time() - start_time
        
        print(f"✅ 模型加载成功 (耗时: {load_time:.2f}秒)")
        
        # 计算相似度
        print(f"\n🔄 计算句子相似度...")
        start_time = time.time()
        
        # 方法1: 公司电脑使用的计算方式
        print(f"📊 方法1 (公司电脑方式):")
        embeddings = model.encode([sentence1, sentence2])
        embedding1 = embeddings[0]  # 提取 sentence1 嵌入
        embedding2 = embeddings[1]  # 提取 sentence2 嵌入
        # 计算余弦相似度，util.cos_sim 输出是二维张量，取 [0][0] 拿到标量相似度值
        cosine_score = util.cos_sim(embedding1, embedding2)[0][0]
        similarity_score_company = float(cosine_score)
        print(f"  相似度分数: {similarity_score_company:.4f}")
        
        # 方法2: 当前脚本的计算方式
        print(f"📊 方法2 (当前脚本方式):")
        embeddings1 = model.encode(sentence1, convert_to_tensor=True)
        embeddings2 = model.encode(sentence2, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
        similarity_score_current = float(cosine_scores[0][0])
        print(f"  相似度分数: {similarity_score_current:.4f}")
        
        # 使用公司电脑的方式作为主要结果
        similarity_score = similarity_score_company
        
        calc_time = time.time() - start_time
        
        print(f"✅ 相似度计算完成 (耗时: {calc_time:.2f}秒)")
        print(f"📊 相似度分数: {similarity_score:.4f}")
        
        # 分析结果
        print(f"\n📈 结果分析:")
        if similarity_score > 0.8:
            print(f"  🟢 高相似度 (>0.8): 句子语义非常相似")
        elif similarity_score > 0.6:
            print(f"  🟡 中等相似度 (0.6-0.8): 句子有一定相似性")
        elif similarity_score > 0.4:
            print(f"  🟠 低相似度 (0.4-0.6): 句子相似性较低")
        else:
            print(f"  🔴 极低相似度 (<0.4): 句子几乎不相似")
        
        # 预期分析
        print(f"\n💡 预期分析:")
        print(f"  句子1: '{sentence1}' - 金融产品描述")
        print(f"  句子2: '{sentence2}' - 礼貌用语")
        print(f"  预期: 这两个句子在语义上应该差异很大，相似度应该较低")
        
        if similarity_score < 0.5:
            print(f"  ✅ 结果符合预期: 相似度低，模型工作正常")
        else:
            print(f"  ⚠️  结果异常: 相似度过高，可能需要检查模型")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请安装sentence-transformers: pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"❌ 模型加载或计算错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_additional_sentences():
    """测试更多句子对"""
    
    model_path = "/Users/CeliaLin_1/Desktop/paraphrase-multilingual-MiniLM-L12-v2"
    
    if not os.path.exists(model_path):
        print("❌ 模型路径不存在，跳过额外测试")
        return
    
    try:
        from sentence_transformers import SentenceTransformer
        from sentence_transformers import util
        
        print(f"\n" + "=" * 80)
        print("🧪 额外句子测试")
        print("=" * 80)
        
        model = SentenceTransformer(model_path)
        
        # 测试句子对
        test_pairs = [
            ("债券票据的类别为高级无抵押", "债券票据的类别为高级无抵押"),  # 完全相同
            ("债券票据的类别为高级无抵押", "债券票据类别是高级无抵押"),    # 相似表达
            ("债券票据的类别为高级无抵押", "基金产品的收益很稳定"),        # 相关金融
            ("债券票据的类别为高级无抵押", "今天天气很好"),               # 完全不相关
            ("非常感谢。", "谢谢您"),                                    # 相似礼貌用语
            ("非常感谢。", "债券票据的类别为高级无抵押"),                # 完全不相关
        ]
        
        print(f"📊 测试结果 (使用公司电脑计算方式):")
        for i, (s1, s2) in enumerate(test_pairs, 1):
            # 使用公司电脑的计算方式
            embeddings = model.encode([s1, s2])
            embedding1 = embeddings[0]
            embedding2 = embeddings[1]
            cosine_score = util.cos_sim(embedding1, embedding2)[0][0]
            similarity = float(cosine_score)
            
            print(f"  {i}. '{s1}' vs '{s2}'")
            print(f"     相似度: {similarity:.4f}")
            
            # 简单判断
            if similarity > 0.9:
                print(f"     → 几乎相同")
            elif similarity > 0.7:
                print(f"     → 很相似")
            elif similarity > 0.5:
                print(f"     → 有一定相似性")
            else:
                print(f"     → 不相似")
            print()
        
    except Exception as e:
        print(f"❌ 额外测试失败: {e}")

if __name__ == "__main__":
    print("🎯 SBERT模型完整性测试")
    print("用于验证公司内网模型是否正确加载")
    
    success = test_sbert_model()
    
    if success:
        print(f"\n✅ 基础测试完成")
        # 询问是否进行额外测试
        try:
            response = input("\n是否进行额外句子测试? (y/n): ").lower().strip()
            if response in ['y', 'yes', '是']:
                test_additional_sentences()
        except KeyboardInterrupt:
            print("\n测试结束")
    else:
        print(f"\n❌ 测试失败，请检查模型路径和依赖")
    
    print(f"\n" + "=" * 80)
    print("🎯 测试完成")
    print("=" * 80)
