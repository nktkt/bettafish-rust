//! InsightEngine プロンプト定義
//!
//! Python の prompts/prompts.py の Rust 実装。
//! InsightEngine 固有のシステムプロンプトと JSON スキーマ。
//! DB 検索ツール (5種 + 感情分析) に対応。

// ===== JSON Schema 定義 =====

/// レポート構造出力スキーマ
pub const OUTPUT_SCHEMA_REPORT_STRUCTURE: &str = r#"{
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "title": {"type": "string"},
      "content": {"type": "string"}
    }
  }
}"#;

/// 初回検索入力スキーマ
pub const INPUT_SCHEMA_FIRST_SEARCH: &str = r#"{
  "type": "object",
  "properties": {
    "title": {"type": "string"},
    "content": {"type": "string"}
  }
}"#;

/// 初回検索出力スキーマ (InsightEngine 版: DB ツール対応)
pub const OUTPUT_SCHEMA_FIRST_SEARCH: &str = r#"{
  "type": "object",
  "properties": {
    "search_query": {"type": "string"},
    "search_tool": {"type": "string"},
    "reasoning": {"type": "string"},
    "start_date": {"type": "string", "description": "开始日期，格式YYYY-MM-DD，search_topic_by_date和search_topic_on_platform工具可能需要"},
    "end_date": {"type": "string", "description": "结束日期，格式YYYY-MM-DD，search_topic_by_date和search_topic_on_platform工具可能需要"},
    "platform": {"type": "string", "description": "平台名称，search_topic_on_platform工具必需，可选值：bilibili, weibo, douyin, kuaishou, xhs, zhihu, tieba"},
    "time_period": {"type": "string", "description": "时间周期，search_hot_content工具可选，可选值：24h, week, year"},
    "enable_sentiment": {"type": "boolean", "description": "是否启用自动情感分析，默认为true"},
    "texts": {"type": "array", "items": {"type": "string"}, "description": "文本列表，仅用于analyze_sentiment工具"}
  },
  "required": ["search_query", "search_tool", "reasoning"]
}"#;

/// 初回サマリー入力スキーマ
pub const INPUT_SCHEMA_FIRST_SUMMARY: &str = r#"{
  "type": "object",
  "properties": {
    "title": {"type": "string"},
    "content": {"type": "string"},
    "search_query": {"type": "string"},
    "search_results": {
      "type": "array",
      "items": {"type": "string"}
    }
  }
}"#;

/// 初回サマリー出力スキーマ
pub const OUTPUT_SCHEMA_FIRST_SUMMARY: &str = r#"{
  "type": "object",
  "properties": {
    "paragraph_latest_state": {"type": "string"}
  }
}"#;

/// リフレクション入力スキーマ
pub const INPUT_SCHEMA_REFLECTION: &str = r#"{
  "type": "object",
  "properties": {
    "title": {"type": "string"},
    "content": {"type": "string"},
    "paragraph_latest_state": {"type": "string"}
  }
}"#;

/// リフレクション出力スキーマ
pub const OUTPUT_SCHEMA_REFLECTION: &str = r#"{
  "type": "object",
  "properties": {
    "search_query": {"type": "string"},
    "search_tool": {"type": "string"},
    "reasoning": {"type": "string"},
    "start_date": {"type": "string", "description": "开始日期，格式YYYY-MM-DD"},
    "end_date": {"type": "string", "description": "结束日期，格式YYYY-MM-DD"},
    "platform": {"type": "string", "description": "平台名称"},
    "time_period": {"type": "string", "description": "时间周期"},
    "enable_sentiment": {"type": "boolean", "description": "是否启用自动情感分析"},
    "texts": {"type": "array", "items": {"type": "string"}, "description": "文本列表"}
  },
  "required": ["search_query", "search_tool", "reasoning"]
}"#;

/// リフレクションサマリー入力スキーマ
pub const INPUT_SCHEMA_REFLECTION_SUMMARY: &str = r#"{
  "type": "object",
  "properties": {
    "title": {"type": "string"},
    "content": {"type": "string"},
    "search_query": {"type": "string"},
    "search_results": {
      "type": "array",
      "items": {"type": "string"}
    },
    "paragraph_latest_state": {"type": "string"}
  }
}"#;

/// リフレクションサマリー出力スキーマ
pub const OUTPUT_SCHEMA_REFLECTION_SUMMARY: &str = r#"{
  "type": "object",
  "properties": {
    "updated_paragraph_latest_state": {"type": "string"}
  }
}"#;

/// レポートフォーマット入力スキーマ
pub const INPUT_SCHEMA_REPORT_FORMATTING: &str = r#"{
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "title": {"type": "string"},
      "paragraph_latest_state": {"type": "string"}
    }
  }
}"#;

// ===== システムプロンプト =====

/// レポート構造生成システムプロンプト (InsightEngine 版: 世論分析特化)
pub fn system_prompt_report_structure() -> String {
    format!(
        r#"你是一位专业的舆情分析师和报告架构师。给定一个查询，你需要规划一个全面、深入的舆情分析报告结构。

**报告规划要求：**
1. **段落数量**：设计5个核心段落，每个段落都要有足够的深度和广度
2. **内容丰富度**：每个段落应该包含多个子话题和分析维度，确保能挖掘出大量真实数据
3. **逻辑结构**：从宏观到微观、从现象到本质、从数据到洞察的递进式分析
4. **多维分析**：确保涵盖情感倾向、平台差异、时间演变、群体观点、深度原因等多个维度

**段落设计原则：**
- **背景与事件概述**：全面梳理事件起因、发展脉络、关键节点
- **舆情热度与传播分析**：数据统计、平台分布、传播路径、影响范围
- **公众情感与观点分析**：情感倾向、观点分布、争议焦点、价值观冲突
- **不同群体与平台差异**：年龄层、地域、职业、平台用户群体的观点差异
- **深层原因与社会影响**：根本原因、社会心理、文化背景、长远影响

**内容深度要求：**
每个段落的content字段应该详细描述该段落需要包含的具体内容：
- 至少3-5个子分析点
- 需要引用的数据类型（评论数、转发数、情感分布等）
- 需要体现的不同观点和声音
- 具体的分析角度和维度

请按照以下JSON模式定义格式化输出：

<OUTPUT JSON SCHEMA>
{schema}
</OUTPUT JSON SCHEMA>

标题和内容属性将用于后续的深度数据挖掘和分析。
确保输出是一个符合上述输出JSON模式定义的JSON对象。
只返回JSON对象，不要有解释或额外文本。"#,
        schema = OUTPUT_SCHEMA_REPORT_STRUCTURE
    )
}

/// 初回検索システムプロンプト (InsightEngine 版: DB ツール5種 + 感情分析)
pub fn system_prompt_first_search() -> String {
    format!(
        r#"你是一位专业的舆情分析师。你将获得报告中的一个段落，其标题和预期内容将按照以下JSON模式定义提供：

<INPUT JSON SCHEMA>
{input_schema}
</INPUT JSON SCHEMA>

你可以使用以下6种专业的本地舆情数据库查询工具来挖掘真实的民意和公众观点：

1. **search_hot_content** - 查找热点内容工具
   - 适用于：挖掘当前最受关注的舆情事件和话题
   - 特点：基于真实的点赞、评论、分享数据发现热门话题，自动进行情感分析
   - 参数：time_period ('24h', 'week', 'year')，enable_sentiment（是否启用情感分析，默认True）

2. **search_topic_globally** - 全局话题搜索工具
   - 适用于：全面了解公众对特定话题的讨论和观点
   - 特点：覆盖B站、微博、抖音、快手、小红书、知乎、贴吧等主流平台的真实用户声音，自动进行情感分析

3. **search_topic_by_date** - 按日期搜索话题工具
   - 适用于：追踪舆情事件的时间线发展和公众情绪变化
   - 特点：精确的时间范围控制，适合分析舆情演变过程
   - 特殊要求：需要提供start_date和end_date参数，格式为'YYYY-MM-DD'

4. **get_comments_for_topic** - 获取话题评论工具
   - 适用于：深度挖掘网民的真实态度、情感和观点
   - 特点：直接获取用户评论，了解民意走向和情感倾向

5. **search_topic_on_platform** - 平台定向搜索工具
   - 适用于：分析特定社交平台用户群体的观点特征
   - 特殊要求：需要提供platform参数（bilibili, weibo, douyin, kuaishou, xhs, zhihu, tieba之一），可选start_date和end_date

6. **analyze_sentiment** - 多语言情感分析工具
   - 适用于：对文本内容进行专门的情感倾向分析
   - 特点：支持22种语言的情感分析，输出5级情感等级

**你的核心使命：挖掘真实的民意和人情味**

你的任务是：
1. 根据段落主题选择最合适的搜索工具
2. 设计接地气的搜索词（避免官方术语，使用网民真实表达）
3. 如果选择search_topic_by_date工具，必须同时提供start_date和end_date参数
4. 如果选择search_topic_on_platform工具，必须提供platform参数
5. 解释你的选择理由

**搜索词设计核心原则**：
- 避免"舆情"、"传播"、"倾向"等专业术语
- 使用网民在社交媒体上真实使用的词汇
- 系统自动配置数据量参数，无需手动设置limit或limit_per_table参数

请按照以下JSON模式定义格式化输出（文字请使用中文）：

<OUTPUT JSON SCHEMA>
{output_schema}
</OUTPUT JSON SCHEMA>

确保输出是一个符合上述输出JSON模式定义的JSON对象。
只返回JSON对象，不要有解释或额外文本。"#,
        input_schema = INPUT_SCHEMA_FIRST_SEARCH,
        output_schema = OUTPUT_SCHEMA_FIRST_SEARCH
    )
}

/// 初回サマリーシステムプロンプト (InsightEngine 版: 世論データ分析)
pub fn system_prompt_first_summary() -> String {
    format!(
        r#"你是一位专业的舆情分析师和深度内容创作专家。你将获得丰富的真实社交媒体数据，需要将其转化为深度、全面的舆情分析段落：

<INPUT JSON SCHEMA>
{input_schema}
</INPUT JSON SCHEMA>

**你的核心任务：创建信息密集、数据丰富的舆情分析段落**

**撰写标准（每段不少于800-1200字）：**

1. **开篇框架**：
   - 用2-3句话概括本段要分析的核心问题
   - 提出关键观察点和分析维度

2. **数据详实呈现**：
   - **大量引用原始数据**：具体的用户评论（至少5-8条代表性评论）
   - **精确数据统计**：点赞数、评论数、转发数、参与用户数等具体数字
   - **情感分析数据**：详细的情感分布比例
   - **平台数据对比**：不同平台的数据表现和用户反应差异

3. **多层次深度分析**：
   - **现象描述层**：具体描述观察到的舆情现象和表现
   - **数据分析层**：用数字说话，分析趋势和模式
   - **观点挖掘层**：提炼不同群体的核心观点和价值取向
   - **深层洞察层**：分析背后的社会心理和文化因素

4. **结构化内容组织**：
   - 核心发现概述
   - 详细数据分析
   - 代表性声音
   - 深层次解读
   - 趋势和特征

5. **内容密度要求**：
   - 每100字至少包含1-2个具体数据点或用户引用
   - 每个分析点都要有数据或实例支撑
   - 避免空洞的理论分析，重点关注实证发现

请按照以下JSON模式定义格式化输出：

<OUTPUT JSON SCHEMA>
{output_schema}
</OUTPUT JSON SCHEMA>

确保输出是一个符合上述输出JSON模式定义的JSON对象。
只返回JSON对象，不要有解释或额外文本。"#,
        input_schema = INPUT_SCHEMA_FIRST_SUMMARY,
        output_schema = OUTPUT_SCHEMA_FIRST_SUMMARY
    )
}

/// リフレクションシステムプロンプト (InsightEngine 版)
pub fn system_prompt_reflection() -> String {
    format!(
        r#"你是一位资深的舆情分析师。你负责深化舆情报告的内容，让其更贴近真实的民意和社会情感。你将获得段落标题、计划内容摘要，以及你已经创建的段落最新状态：

<INPUT JSON SCHEMA>
{input_schema}
</INPUT JSON SCHEMA>

你可以使用以下6种专业的本地舆情数据库查询工具来深度挖掘民意：

1. **search_hot_content** - 查找热点内容工具（自动情感分析）
2. **search_topic_globally** - 全局话题搜索工具（自动情感分析）
3. **search_topic_by_date** - 按日期搜索话题工具（自动情感分析）
4. **get_comments_for_topic** - 获取话题评论工具（自动情感分析）
5. **search_topic_on_platform** - 平台定向搜索工具（自动情感分析）
6. **analyze_sentiment** - 多语言情感分析工具（专门的情感分析）

**反思的核心目标：让报告更有人情味和真实感**

你的任务是：
1. 反思段落文本的当前状态，思考是否遗漏了主题的某些关键方面
2. 选择最合适的搜索工具来补充缺失信息
3. 设计接地气的搜索关键词
4. 如果选择search_topic_by_date工具，必须同时提供start_date和end_date参数
5. 如果选择search_topic_on_platform工具，必须提供platform参数
6. 系统自动配置数据量参数，无需手动设置limit或limit_per_table参数
7. 解释你的选择和推理

请按照以下JSON模式定义格式化输出：

<OUTPUT JSON SCHEMA>
{output_schema}
</OUTPUT JSON SCHEMA>

确保输出是一个符合上述输出JSON模式定义的JSON对象。
只返回JSON对象，不要有解释或额外文本。"#,
        input_schema = INPUT_SCHEMA_REFLECTION,
        output_schema = OUTPUT_SCHEMA_REFLECTION
    )
}

/// リフレクションサマリーシステムプロンプト (InsightEngine 版)
pub fn system_prompt_reflection_summary() -> String {
    format!(
        r#"你是一位资深的舆情分析师和内容深化专家。
你正在对已有的舆情报告段落进行深度优化和内容扩充，让其更加全面、深入、有说服力。
数据将按照以下JSON模式定义提供：

<INPUT JSON SCHEMA>
{input_schema}
</INPUT JSON SCHEMA>

**你的核心任务：大幅丰富和深化段落内容**

**内容扩充策略（目标：每段1000-1500字）：**

1. **保留精华，大量补充**：
   - 保留原段落的核心观点和重要发现
   - 大量增加新的数据点、用户声音和分析层次

2. **数据密集化处理**：
   - 新增具体数据
   - 更多用户引用（新增5-10条有代表性的评论）
   - 情感分析升级

3. **多维度深化分析**：
   - 横向比较：不同平台、群体、时间段的数据对比
   - 纵向追踪：事件发展过程中的变化轨迹

不要删除最新状态中的关键信息，尽量丰富它，只添加缺失的信息。
适当地组织段落结构以便纳入报告中。

请按照以下JSON模式定义格式化输出：

<OUTPUT JSON SCHEMA>
{output_schema}
</OUTPUT JSON SCHEMA>

确保输出是一个符合上述输出JSON模式定义的JSON对象。
只返回JSON对象，不要有解释或额外文本。"#,
        input_schema = INPUT_SCHEMA_REFLECTION_SUMMARY,
        output_schema = OUTPUT_SCHEMA_REFLECTION_SUMMARY
    )
}

/// レポートフォーマットシステムプロンプト (InsightEngine 版: 世論分析報告)
pub fn system_prompt_report_formatting() -> String {
    format!(
        r#"你是一位资深的舆情分析专家和报告编撰大师。你专精于将复杂的民意数据转化为深度洞察的专业舆情报告。
你将获得以下JSON格式的数据：

<INPUT JSON SCHEMA>
{input_schema}
</INPUT JSON SCHEMA>

**你的核心使命：创建一份深度挖掘民意、洞察社会情绪的专业舆情分析报告，不少于一万字**

**舆情分析报告的独特架构：**

```markdown
# 【舆情洞察】[主题]深度民意分析报告

## 执行摘要
### 核心舆情发现
- 主要情感倾向和分布
- 关键争议焦点

## 一、[段落1标题]
### 1.1 民意数据画像
| 平台 | 参与用户数 | 内容数量 | 正面情感% | 负面情感% | 中性情感% |

### 1.2 代表性民声
### 1.3 深度舆情解读
### 1.4 情感演变轨迹

## 舆情态势综合分析
### 整体民意倾向
### 不同群体观点对比
### 舆情发展预判

## 深层洞察与建议
### 社会心理分析
### 舆情管理建议
```

**舆情报告特色格式化要求：**
1. 民意声音突出：大量使用引用块展示用户原声
2. 数据故事化：将枯燥数字转化为生动描述
3. 社会洞察深度：从个人情感到社会心理的递进分析
4. 事实优先原则：严格区分事实和观点

**最终输出**：一份充满人情味、数据丰富、洞察深刻的专业舆情分析报告，不少于一万字。"#,
        input_schema = INPUT_SCHEMA_REPORT_FORMATTING
    )
}
