//! KeywordOptimizer - キーワード最適化ミドルウェア
//!
//! Python の tools/keyword_optimizer.py の Rust 実装。
//! LLM を使用して Agent 生成の検索クエリを
//! インターネットスラングに近い世論 DB 向けキーワードに変換。

use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use tracing::{error, info, warn};

use bettafish_common::text_processing::clean_json_tags;
use bettafish_llm::{InvokeOptions, LLMClient};

/// キーワード最適化レスポンス
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeywordOptimizationResponse {
    /// 元のクエリ
    pub original_query: String,
    /// 最適化されたキーワードリスト
    pub optimized_keywords: Vec<String>,
    /// 最適化理由
    pub reasoning: String,
    /// 成功フラグ
    pub success: bool,
    /// エラーメッセージ
    pub error_message: String,
}

/// キーワード最適化器
///
/// LLM を使用して、Agent が生成したフォーマルな検索クエリを
/// ネットユーザーが実際に使う言葉遣いに変換する。
pub struct KeywordOptimizer {
    llm_client: LLMClient,
}

impl KeywordOptimizer {
    /// 新しい KeywordOptimizer を作成
    pub fn new(llm_client: LLMClient) -> Self {
        Self { llm_client }
    }

    /// 設定から KeywordOptimizer を作成
    pub fn from_config(config: &bettafish_config::Settings) -> Result<Self> {
        let llm_client = LLMClient::new(
            &config.keyword_optimizer_api_key,
            &config.keyword_optimizer_model_name,
            config.keyword_optimizer_base_url.as_deref(),
        )?;
        Ok(Self { llm_client })
    }

    /// 検索キーワードを最適化
    ///
    /// Agent が生成した元のクエリを、世論 DB 検索に適したキーワードに変換。
    ///
    /// # Arguments
    /// * `original_query` - 元の検索クエリ
    /// * `context` - 追加のコンテキスト情報
    pub async fn optimize_keywords(
        &self,
        original_query: &str,
        context: &str,
    ) -> KeywordOptimizationResponse {
        info!("キーワード最適化ミドルウェア: クエリ '{}' を処理中", original_query);

        match self.try_optimize(original_query, context).await {
            Ok(response) => response,
            Err(e) => {
                error!("キーワード最適化失敗: {}", e);
                let fallback = self.fallback_keyword_extraction(original_query);
                KeywordOptimizationResponse {
                    original_query: original_query.to_string(),
                    optimized_keywords: fallback,
                    reasoning: "システムエラー、バックアップキーワード抽出を使用".to_string(),
                    success: false,
                    error_message: e.to_string(),
                }
            }
        }
    }

    /// 最適化を試行（内部）
    async fn try_optimize(
        &self,
        original_query: &str,
        context: &str,
    ) -> Result<KeywordOptimizationResponse> {
        let system_prompt = Self::build_system_prompt();
        let user_prompt = Self::build_user_prompt(original_query, context);

        let options = InvokeOptions {
            temperature: Some(0.7),
            ..Default::default()
        };

        let response = self
            .llm_client
            .stream_invoke_to_string(&system_prompt, &user_prompt, &options)
            .await?;

        // レスポンスをパース
        let cleaned = clean_json_tags(&response);

        let (keywords, reasoning) = if cleaned.trim().starts_with('{') {
            match serde_json::from_str::<serde_json::Value>(&cleaned) {
                Ok(parsed) => {
                    let kw = parsed
                        .get("keywords")
                        .and_then(|v| v.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default();
                    let r = parsed
                        .get("reasoning")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    (kw, r)
                }
                Err(e) => {
                    warn!("JSON パース失敗、バックアップ方式を使用: {}", e);
                    let kw = self.extract_keywords_from_text(&cleaned);
                    (kw, cleaned.clone())
                }
            }
        } else {
            let kw = self.extract_keywords_from_text(&cleaned);
            (kw, cleaned.clone())
        };

        let validated = self.validate_keywords(&keywords);

        if !validated.is_empty() {
            let mut msg = format!("最適化成功: {} 個のキーワード", validated.len());
            for (i, k) in validated.iter().enumerate() {
                msg.push_str(&format!("\n   {}. '{}'", i + 1, k));
            }
            info!("{}", msg);
        }

        Ok(KeywordOptimizationResponse {
            original_query: original_query.to_string(),
            optimized_keywords: validated,
            reasoning,
            success: true,
            error_message: String::new(),
        })
    }

    /// システムプロンプトを構築
    fn build_system_prompt() -> String {
        r#"你是一位专业的舆情数据挖掘专家。你的任务是将用户提供的搜索查询优化为更适合在社交媒体舆情数据库中查找的关键词。

**核心原则**：
1. **贴近网民语言**：使用普通网友在社交媒体上会使用的词汇
2. **避免专业术语**：不使用"舆情"、"传播"、"倾向"、"展望"等官方词汇
3. **简洁具体**：每个关键词要非常简洁明了，便于数据库匹配
4. **情感丰富**：包含网民常用的情感表达词汇
5. **数量控制**：最少提供10个关键词，最多提供20个关键词
6. **避免重复**：不要脱离初始查询的主题

**重要提醒**：每个关键词都必须是一个不可分割的独立词条，严禁在词条内部包含空格。

**输出格式**：
请以JSON格式返回结果：
{
    "keywords": ["关键词1", "关键词2", "关键词3"],
    "reasoning": "选择这些关键词的理由"
}

**示例**：
输入："武汉大学舆情管理 未来展望 发展趋势"
输出：
{
    "keywords": ["武大", "武汉大学", "学校管理", "武大教育"],
    "reasoning": "选择'武大'和'武汉大学'作为核心词汇..."
}"#
            .to_string()
    }

    /// ユーザープロンプトを構築
    fn build_user_prompt(original_query: &str, context: &str) -> String {
        let mut prompt = format!(
            "请将以下搜索查询优化为适合舆情数据库查询的关键词：\n\n原始查询：{}",
            original_query
        );

        if !context.is_empty() {
            prompt.push_str(&format!("\n\n上下文信息：{}", context));
        }

        prompt.push_str(
            "\n\n请记住：要使用网民在社交媒体上真实使用的词汇，避免官方术语和专业词汇。",
        );

        prompt
    }

    /// テキストからキーワードを抽出（JSON パース失敗時のフォールバック）
    fn extract_keywords_from_text(&self, text: &str) -> Vec<String> {
        let mut keywords = Vec::new();

        // コロン区切りのキーワードを検索
        for line in text.lines() {
            let line = line.trim();
            let parts: Vec<&str> = if line.contains('\u{ff1a}') {
                // 全角コロン
                line.splitn(2, '\u{ff1a}').collect()
            } else if line.contains(':') {
                line.splitn(2, ':').collect()
            } else {
                continue;
            };

            if parts.len() > 1 {
                let potential = parts[1].trim();
                if potential.contains('\u{3001}') {
                    // 読点区切り
                    keywords.extend(
                        potential
                            .split('\u{3001}')
                            .map(|k| k.trim().to_string()),
                    );
                } else if potential.contains(',') {
                    keywords.extend(potential.split(',').map(|k| k.trim().to_string()));
                } else {
                    keywords.push(potential.to_string());
                }
            }
        }

        // 引用符内のコンテンツを検索
        if keywords.is_empty() {
            if let Ok(re) = Regex::new(r#"["""\']([^"""\']+)["""\']"#) {
                for cap in re.captures_iter(text) {
                    if let Some(m) = cap.get(1) {
                        keywords.push(m.as_str().to_string());
                    }
                }
            }
        }

        // クリーニングと検証
        keywords
            .into_iter()
            .take(20)
            .map(|k| {
                k.trim()
                    .trim_matches(|c: char| c == '"' || c == '\'' || c == '\u{201c}' || c == '\u{201d}')
                    .to_string()
            })
            .filter(|k| !k.is_empty() && k.len() <= 60)
            .collect()
    }

    /// キーワードを検証・クリーニング
    fn validate_keywords(&self, keywords: &[String]) -> Vec<String> {
        let bad_keywords = [
            "态度分析",
            "公众反应",
            "情绪倾向",
            "未来展望",
            "发展趋势",
            "战略规划",
            "政策导向",
            "管理机制",
        ];

        keywords
            .iter()
            .filter_map(|k| {
                let k = k
                    .trim()
                    .trim_matches(|c: char| c == '"' || c == '\'' || c == '\u{201c}' || c == '\u{201d}')
                    .to_string();

                if k.is_empty() || k.len() > 60 {
                    return None;
                }

                if bad_keywords.iter().any(|bw| k.contains(bw)) {
                    return None;
                }

                Some(k)
            })
            .take(20)
            .collect()
    }

    /// バックアップキーワード抽出方式
    fn fallback_keyword_extraction(&self, original_query: &str) -> Vec<String> {
        if let Ok(re) = Regex::new(r"[\s，。！？；：、]+") {
            let tokens: Vec<String> = re
                .split(original_query)
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty() && s.chars().count() >= 2)
                .collect();

            if !tokens.is_empty() {
                return tokens.into_iter().take(20).collect();
            }
        }

        // 最終フォールバック
        let first_word = original_query
            .split_whitespace()
            .next()
            .unwrap_or(original_query);
        if first_word.is_empty() {
            vec!["热门".to_string()]
        } else {
            vec![first_word.to_string()]
        }
    }
}
