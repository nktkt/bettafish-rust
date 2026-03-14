//! BettaFish MediaEngine (スタブ)
//!
//! インターネット上のマルチモーダルコンテンツを検索する MediaEngine のインターフェース定義。
//! 完全実装は今後のフェーズで行う。

use async_trait::async_trait;

/// MediaEngine トレイト
///
/// MediaEngine は Bocha/Anspire API を使用してウェブ全体を検索する。
/// ツール: BochaMultimodalSearch / AnspireAISearch (5種)
#[async_trait]
pub trait MediaEngine: Send + Sync {
    /// 深度研究を実行
    async fn research(&mut self, query: &str) -> anyhow::Result<String>;

    /// 進捗サマリーを取得
    fn get_progress_summary(&self) -> serde_json::Value;
}

/// MediaEngine 設定
#[derive(Debug, Clone, serde::Deserialize)]
pub struct MediaEngineConfig {
    pub api_key: String,
    pub base_url: Option<String>,
    pub model_name: String,
    pub search_tool_type: String,
    pub max_reflections: usize,
}

impl Default for MediaEngineConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: Some("https://aihubmix.com/v1".to_string()),
            model_name: "gemini-2.5-pro".to_string(),
            search_tool_type: "BochaAPI".to_string(),
            max_reflections: 3,
        }
    }
}
