//! BettaFish MindSpider (スタブ)
//!
//! AI クローラー集群のインターフェース定義。
//! データ収集、トピック抽出、プラットフォームクローラー管理を含む。
//! 完全実装は今後のフェーズで行う。

use async_trait::async_trait;

/// MindSpider データ収集エンジントレイト
#[async_trait]
pub trait Spider: Send + Sync {
    /// 深度センチメントクローリングを開始
    async fn start_deep_sentiment_crawling(
        &mut self,
        keywords: &[String],
        platforms: &[String],
    ) -> anyhow::Result<()>;

    /// 広域トピック抽出を実行
    async fn extract_broad_topics(&self) -> anyhow::Result<Vec<Topic>>;

    /// クローリングタスクの状態を取得
    async fn get_task_status(&self, task_id: &str) -> anyhow::Result<serde_json::Value>;
}

/// トピックデータ
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Topic {
    pub topic_id: String,
    pub topic_name: String,
    pub description: String,
    pub keywords: Vec<String>,
    pub relevance_score: f64,
    pub news_count: usize,
}

/// サポートされるプラットフォーム
pub const SUPPORTED_PLATFORMS: &[&str] = &[
    "xhs",     // 小紅書
    "dy",      // 抖音
    "ks",      // 快手
    "bili",    // ビリビリ
    "wb",      // 微博
    "tieba",   // 百度貼吧
    "zhihu",   // 知乎
];

/// MindSpider 設定
#[derive(Debug, Clone, serde::Deserialize)]
pub struct MindSpiderConfig {
    pub api_key: String,
    pub base_url: Option<String>,
    pub model_name: String,
    pub db_dialect: String,
    pub db_host: String,
    pub db_port: u16,
    pub db_user: String,
    pub db_password: String,
    pub db_name: String,
}

impl Default for MindSpiderConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: None,
            model_name: "deepseek-chat".to_string(),
            db_dialect: "postgresql".to_string(),
            db_host: "localhost".to_string(),
            db_port: 5432,
            db_user: "root".to_string(),
            db_password: String::new(),
            db_name: "bettafish".to_string(),
        }
    }
}
