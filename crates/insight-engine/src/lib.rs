//! BettaFish InsightEngine (スタブ)
//!
//! ローカルデータベースの世論データを分析する InsightEngine のインターフェース定義。
//! 完全実装は今後のフェーズで行う。

use async_trait::async_trait;

/// InsightEngine トレイト
///
/// InsightEngine はローカル DB の SNS データを使用して分析を行う。
/// ツール: MediaCrawlerDB (5種), KeywordOptimizer, SentimentAnalyzer
#[async_trait]
pub trait InsightEngine: Send + Sync {
    /// 深度研究を実行
    async fn research(&mut self, query: &str) -> anyhow::Result<String>;

    /// 進捗サマリーを取得
    fn get_progress_summary(&self) -> serde_json::Value;
}

/// InsightEngine 設定
#[derive(Debug, Clone, serde::Deserialize)]
pub struct InsightEngineConfig {
    pub api_key: String,
    pub base_url: Option<String>,
    pub model_name: String,
    pub max_reflections: usize,
    pub max_paragraphs: usize,
}

impl Default for InsightEngineConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: Some("https://api.moonshot.cn/v1".to_string()),
            model_name: "kimi-k2-0711-preview".to_string(),
            max_reflections: 3,
            max_paragraphs: 6,
        }
    }
}
