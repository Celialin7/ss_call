# Standard_Script 处理链路详细记录

本文记录系统中 `Standard_Script` 的完整进入流程，包括：

- 进入后做了哪些处理
- 比较时具体如何比较
- 如何判定 `Covered`
- 如何产出 `Best_Matching_Variation`
- 粤语（TF-IDF 路径）与普通话（SBERT 路径）的关键差异

---

## 1. 数据入口与上下文

1. 批处理入口会根据文件名、mapping、语言与 call type，确定本次分析使用的脚本 sheet。
2. `run_analysis` 读取 `Scripts.xlsx` 指定 sheet，得到脚本表（含 `Required_Discussion_Point` 与 `Standard_Script`）。
3. 通话文本先被聚合成 `grouped_lines`（按规则分组后的对话段），后续比较都在“脚本文本 vs 对话组文本”之间进行，而不是逐行 raw transcript 对比。

---

## 2. Standard_Script 进入后做了哪些处理

### 2.1 脚本变体解析（两条路径都使用）

`Standard_Script` 会进入 `parse_script_variations` 进行标准化与拆分：

1. 比较模式预处理（`mode='comparison'`）
   - 去除无关字符（尤其英文/杂质字符）
   - 应用 ASR 常见错误归一
2. 第一次切分
   - 按标点/换行/版本标记等切分（例如 `版本A`、`Version B`）
3. 短片段合并
   - 长度很短（<=5）的碎片会前后合并，避免噪声片段
4. 第二次切分
   - 对过长片段按逗号再切分，并再次做短片段合并
5. 去重
   - 得到用于 Granular 比较的变体列表

这一步的结果意味着：比较对象通常不是 Excel 单元格原文，而是“清洗+切分后的脚本变体”。

---

## 3. 粤语路径（TF-IDF）如何比较

文件：`improved_call_coverage_checker.py`

### 3.1 point 级脚本聚合

`check_coverage` 先按 `Required_Discussion_Point` 聚合脚本，把同一点下多行 `Standard_Script` 收集在一起。

### 3.2 双轨比较：Holistic + Granular

对每个讨论点、每个对话组，执行：

1. **Holistic**
   - 将该点所有脚本行拼成一个完整脚本串（`' '.join(scripts)`）后比较
2. **Granular**
   - 对每个 `script_variation` 分别比较，取该组内最高分
3. 比较 Holistic 与 Granular 的分数，取更高者作为该组该点的最终分数
4. 应用 pattern enhancement（日期/数字等加分逻辑，最终分封顶 1.0）

### 3.3 具体分数构成（语义相似度）

`calculate_semantic_similarity` 的加权构成为：

- TF-IDF + Cosine（char n-gram）: 55%
- Expanded overlap（同义词扩展后重叠）: 15%
- ROUGE-L: 20%
- Keyword coverage: 10%

最终 `weighted_score` 是上述加权和，再可能经过 pattern enhancement 增强。

### 3.4 Covered 判定

对一个 discussion point：

1. 遍历所有对话组，找到增强后分数最高的“全通话最佳组”
2. 若 `best_score >= threshold`，判定 `Covered`，否则 `Not Covered`

默认阈值（粤语）通常是 `0.3`。

### 3.5 Best_Matching_Variation 的来源

- 若最终是 Holistic 胜出：`Best_Matching_Variation` 通常是拼接后的完整脚本串
- 若最终是 Granular 胜出：`Best_Matching_Variation` 是某个具体变体片段
- 若为空有 fallback：使用 `scripts[0]`

---

## 4. 普通话路径（SBERT）如何比较

文件：`Mandarin/improved_call_coverage_checker_M.py`

### 4.1 核心组织方式：先算全量 pairwise

`compute_pairwise_matches` 会对所有 `(group, point)` 配对统一计算指标（一次算好，多处复用）：

1. 先按 point 聚合脚本、展开变体、构建完整 Holistic 文本
2. 对每个 `(group, point)`：
   - 先算 Holistic
   - 若 Holistic 增强后已过阈值，可直接早停选 Holistic
   - 否则再算 Granular 并与 Holistic 比较，择优

### 4.2 具体分数构成（SBERT 混合）

`calculate_semantic_similarity` 的加权构成为：

- SBERT semantic score: 默认 60%（可由 config 按语言读取）
- Expanded overlap: 默认 15%
- ROUGE-L: 默认 15%
- Keyword coverage: 默认 10%

### 4.3 Covered 判定

`check_coverage` 在 pairwise 结果中，按每个 point 选 `weighted_score` 最大的一行作为全局最佳匹配：

- `best_match['weighted_score'] >= threshold` -> `Covered`
- 否则 `Not Covered`

默认阈值（普通话）通常是 `0.4`。

### 4.4 Best_Matching_Variation 的输出层处理

普通话结果里 `Best_Matching_Variation` 在输出前还会：

1. 再做一次 comparison 预处理
2. 长度超过 100 时做截断并加 `...`

因此报表中显示值可能不是内存中的原始 best_variation 全文。

---

## 5. 关于 Executive Summary 的语义提醒

在当前实现中，明细里会把 `Best_Matching_Variation` 重命名展示为 `Standard_Script`。  
这在业务展示上容易误解为“原始脚本列”，但它实际是“最终匹配命中的脚本文本（可能是拼接串或切分片段）”。

---

## 6. 普通话路径的缓存脆弱耦合（当前已识别）

当前普通话路径存在一个容易被忽略的耦合点：

1. SBERT 输入是 `clean1/clean2`（即预处理后的文本）
2. 但 embedding cache 的 key 却使用原始 `text1/text2`

这会导致：只要原始文本有无关差异（空格、符号、格式细节等），即使清洗后完全相同，也可能 miss 预计算缓存并现场 encode，带来不稳定命中与额外耗时。

---

## 7. 后续修复目标（原则）

修复应遵循：

- **Simple**：改动点小，尽量复用现有函数与数据结构
- **Robust**：只要 `clean_text` 相同就稳定命中缓存
- **Non-over-engineered**：不引入复杂新层，不改业务判定逻辑

核心思想：让“钥匙（cache key）”与“内容（SBERT 输入）”统一成同一套 `clean_text` 语义。

