//! BettaFish ForumEngine (スタブ)
//!
//! マルチエージェント討論の調整と合成を行う ForumEngine のインターフェース定義。
//! 完全実装は今後のフェーズで行う。

use async_trait::async_trait;

/// ForumHost トレイト
///
/// ForumHost は LLM パワードのモデレーターとして機能し、
/// 各 Agent の発言を総合して討論を指導する。
#[async_trait]
pub trait ForumHost: Send + Sync {
    /// Agent 発言からホスト発言を生成
    async fn generate_host_speech(&self, forum_logs: &[String]) -> anyhow::Result<Option<String>>;

    /// フォーラムモニタリングを開始
    async fn start_monitoring(&mut self) -> anyhow::Result<()>;

    /// フォーラムモニタリングを停止
    async fn stop_monitoring(&mut self) -> anyhow::Result<()>;
}

/// ForumEngine 設定
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ForumEngineConfig {
    pub api_key: String,
    pub base_url: Option<String>,
    pub model_name: String,
}

impl Default for ForumEngineConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: None,
            model_name: "Qwen3-235B-A22B".to_string(),
        }
    }
}
