//! BettaFish ReportEngine (スタブ)
//!
//! テンプレートベースのレポート生成エンジンのインターフェース定義。
//! HTML/PDF/Markdown レンダリング、チャートSVG変換、IR スキーマ検証を含む。
//! 完全実装は今後のフェーズで行う。

use async_trait::async_trait;

/// ReportEngine トレイト
///
/// ReportEngine はテンプレート解析、章生成、ドキュメント合成、
/// 複数フォーマットレンダリング（HTML/PDF/Markdown）を行う。
#[async_trait]
pub trait ReportEngine: Send + Sync {
    /// レポート生成リクエストをキューに追加
    async fn queue_report(&mut self, query: &str, reports: &[String]) -> anyhow::Result<String>;

    /// タスクステータスを取得
    async fn get_status(&self, task_id: &str) -> anyhow::Result<serde_json::Value>;

    /// 生成されたレポートをダウンロード
    async fn download(&self, task_id: &str, format: &str) -> anyhow::Result<Vec<u8>>;
}

/// IR ドキュメントスキーマバージョン
pub const IR_VERSION: &str = "1.0";

/// サポートされるブロックタイプ
pub const ALLOWED_BLOCK_TYPES: &[&str] = &[
    "heading", "paragraph", "list", "table", "swotTable", "pestTable",
    "blockquote", "engineQuote", "figure", "code", "math",
    "widget", "kpiGrid", "callout", "toc", "hr",
];

/// サポートされるインラインマークタイプ
pub const ALLOWED_INLINE_MARKS: &[&str] = &[
    "bold", "italic", "underline", "strike", "code", "link",
    "color", "font", "highlight", "subscript", "superscript", "math",
];

/// ReportEngine 設定
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ReportEngineConfig {
    pub api_key: String,
    pub base_url: Option<String>,
    pub model_name: String,
}

impl Default for ReportEngineConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: None,
            model_name: "gemini-2.5-pro".to_string(),
        }
    }
}
