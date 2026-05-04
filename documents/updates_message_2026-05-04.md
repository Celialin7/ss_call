# 更新说明（2026-05-04）

本次更新聚焦于普通话/英文 SBERT 缓存命中稳定性修复，以及文档同步。

## 涉及文件与改动

### 1) `Mandarin/improved_call_coverage_checker_M.py`
- 修复 SBERT 缓存 key 语义不一致问题：将相似度缓存与 embedding 查找统一到 `clean_text`。
- 新增 `normalize_script_embedding_keys()`，对历史 `script_embeddings.pkl` 进行兼容归一化，避免旧数据失效。
- 调整运行时通话向量映射：由“原文文本 key”改为“预处理 clean_text key”，与编码输入一致。
- 清理旧的无用中间映射变量，减少脆弱耦合与维护成本。

### 2) `dynamic_script_analysis.py`
- 调整 `generate_script_embeddings()` 产物格式：embedding 字典改为 `clean_text -> embedding`。
- 删除旧的“原文 -> 预处理文本 -> embedding”回填路径中不再需要的中间结构，避免冗余。

### 3) `README.md`
- 增补 MAN/ENG 路径缓存机制说明：key 统一为 `clean_text`，降低格式差异导致的缓存 miss。
- 增补旧 embedding 文件兼容说明：加载时会做 key 归一化。
- 在动态资源刷新流程中补充：MAN/ENG embeddings 使用 `clean_text` key。

### 4) `documents/standard_script_flow_detailed.md`
- 新增完整流程记录文档：详细解释 `Standard_Script` 从进入、比较、判定 `Covered` 到形成 `Best_Matching_Variation` 的全链路。

