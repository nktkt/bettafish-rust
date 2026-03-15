//! SentimentAnalyzer - 多言語感情分析ツール
//!
//! Python の tools/sentiment_analyzer.py の Rust 実装。
//! PyTorch/Transformers は Rust では直接使用できないため、
//! トレイトとして定義し、スタブ実装を提供する。
//!
//! 将来的に ONNX Runtime や candle 等で実モデル推論に置換可能。

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{info, warn};

// ---------------------------------------------------------------------------
// データ構造
// ---------------------------------------------------------------------------

/// 感情分析結果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentResult {
    /// 入力テキスト
    pub text: String,
    /// 感情ラベル ("非常负面", "负面", "中性", "正面", "非常正面")
    pub sentiment_label: String,
    /// 信頼度 (0.0 ~ 1.0)
    pub confidence: f64,
    /// 確率分布
    pub probability_distribution: HashMap<String, f64>,
    /// 成功フラグ
    pub success: bool,
    /// エラーメッセージ
    pub error_message: Option<String>,
    /// 分析が実行されたか
    pub analysis_performed: bool,
}

impl Default for SentimentResult {
    fn default() -> Self {
        Self {
            text: String::new(),
            sentiment_label: "中性".to_string(),
            confidence: 0.0,
            probability_distribution: HashMap::new(),
            success: false,
            error_message: None,
            analysis_performed: false,
        }
    }
}

/// バッチ感情分析結果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSentimentResult {
    /// 個別結果リスト
    pub results: Vec<SentimentResult>,
    /// 処理総数
    pub total_processed: usize,
    /// 成功数
    pub success_count: usize,
    /// 失敗数
    pub failed_count: usize,
    /// 平均信頼度
    pub average_confidence: f64,
    /// 分析が実行されたか
    pub analysis_performed: bool,
}

impl Default for BatchSentimentResult {
    fn default() -> Self {
        Self {
            results: Vec::new(),
            total_processed: 0,
            success_count: 0,
            failed_count: 0,
            average_confidence: 0.0,
            analysis_performed: false,
        }
    }
}

// ---------------------------------------------------------------------------
// 感情ラベルマッピング (5段階)
// ---------------------------------------------------------------------------

/// 感情ラベルのマッピング
pub const SENTIMENT_MAP: &[(i32, &str)] = &[
    (0, "非常负面"),
    (1, "负面"),
    (2, "中性"),
    (3, "正面"),
    (4, "非常正面"),
];

/// サポート言語一覧 (22言語)
pub const SUPPORTED_LANGUAGES: &[&str] = &[
    "中文", "英文", "西班牙文", "阿拉伯文", "日文", "韩文", "德文", "法文",
    "意大利文", "葡萄牙文", "俄文", "荷兰文", "波兰文", "土耳其文", "丹麦文",
    "希腊文", "芬兰文", "瑞典文", "挪威文", "匈牙利文", "捷克文", "保加利亚文",
];

// ---------------------------------------------------------------------------
// SentimentAnalyzer トレイト
// ---------------------------------------------------------------------------

/// 多言語感情分析トレイト
///
/// PyTorch ベースのモデルは Rust で直接使用できないため、
/// トレイトとして抽象化し、将来的な実装 (ONNX Runtime 等) を可能にする。
#[async_trait]
pub trait SentimentAnalyzer: Send + Sync {
    /// 初期化
    fn initialize(&mut self) -> bool;

    /// 初期化状態
    fn is_initialized(&self) -> bool;

    /// 無効化状態
    fn is_disabled(&self) -> bool;

    /// 単一テキストの感情分析
    async fn analyze_single_text(&self, text: &str) -> SentimentResult;

    /// バッチ感情分析
    async fn analyze_batch(
        &self,
        texts: &[String],
        show_progress: bool,
    ) -> BatchSentimentResult;

    /// クエリ結果の感情分析
    ///
    /// MediaCrawlerDB から返されたクエリ結果に対して感情分析を実行。
    async fn analyze_query_results(
        &self,
        query_results: &[HashMap<String, serde_json::Value>],
        text_field: &str,
        min_confidence: f64,
    ) -> HashMap<String, serde_json::Value>;

    /// モデル情報を取得
    fn get_model_info(&self) -> HashMap<String, serde_json::Value>;
}

// ---------------------------------------------------------------------------
// スタブ実装
// ---------------------------------------------------------------------------

/// スタブ感情分析器
///
/// 実際のモデル推論は行わず、「分析未実行」の結果を返す。
/// 将来 ONNX Runtime や candle 統合時に実装を差し替える。
pub struct StubSentimentAnalyzer {
    initialized: bool,
    disabled: bool,
    #[allow(dead_code)]
    disable_reason: Option<String>,
}

impl StubSentimentAnalyzer {
    /// 新しいスタブ分析器を作成
    pub fn new() -> Self {
        info!("StubSentimentAnalyzer: スタブ感情分析器を作成（モデル推論なし）");
        Self {
            initialized: false,
            disabled: false,
            disable_reason: None,
        }
    }

    /// 透過結果を構築（感情分析不可時）
    fn build_passthrough_analysis(
        &self,
        original_data: &[HashMap<String, serde_json::Value>],
        reason: &str,
    ) -> HashMap<String, serde_json::Value> {
        let total_items = original_data.len();
        let mut analysis = HashMap::new();
        analysis.insert(
            "available".to_string(),
            serde_json::Value::Bool(false),
        );
        analysis.insert(
            "reason".to_string(),
            serde_json::Value::String(reason.to_string()),
        );
        analysis.insert(
            "total_analyzed".to_string(),
            serde_json::json!(0),
        );
        analysis.insert(
            "success_rate".to_string(),
            serde_json::Value::String(format!("0/{}", total_items)),
        );
        analysis.insert(
            "average_confidence".to_string(),
            serde_json::json!(0.0),
        );
        analysis.insert(
            "sentiment_distribution".to_string(),
            serde_json::json!({}),
        );
        analysis.insert(
            "high_confidence_results".to_string(),
            serde_json::json!([]),
        );
        analysis.insert(
            "summary".to_string(),
            serde_json::Value::String(format!("情感分析未执行：{}", reason)),
        );

        let mut result = HashMap::new();
        result.insert(
            "sentiment_analysis".to_string(),
            serde_json::json!(analysis),
        );
        result
    }
}

impl Default for StubSentimentAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SentimentAnalyzer for StubSentimentAnalyzer {
    fn initialize(&mut self) -> bool {
        warn!("StubSentimentAnalyzer: スタブ実装のため初期化をスキップ");
        self.initialized = true;
        true
    }

    fn is_initialized(&self) -> bool {
        self.initialized
    }

    fn is_disabled(&self) -> bool {
        self.disabled
    }

    async fn analyze_single_text(&self, text: &str) -> SentimentResult {
        SentimentResult {
            text: text.to_string(),
            sentiment_label: "情感分析未执行".to_string(),
            confidence: 0.0,
            probability_distribution: HashMap::new(),
            success: false,
            error_message: Some(
                "Rust版ではスタブ実装です。ONNX Runtime統合時に有効化されます。".to_string(),
            ),
            analysis_performed: false,
        }
    }

    async fn analyze_batch(
        &self,
        texts: &[String],
        _show_progress: bool,
    ) -> BatchSentimentResult {
        let results: Vec<SentimentResult> = texts
            .iter()
            .map(|text| SentimentResult {
                text: text.clone(),
                sentiment_label: "情感分析未执行".to_string(),
                confidence: 0.0,
                probability_distribution: HashMap::new(),
                success: false,
                error_message: Some("スタブ実装".to_string()),
                analysis_performed: false,
            })
            .collect();

        BatchSentimentResult {
            total_processed: texts.len(),
            success_count: 0,
            failed_count: texts.len(),
            average_confidence: 0.0,
            results,
            analysis_performed: false,
        }
    }

    async fn analyze_query_results(
        &self,
        query_results: &[HashMap<String, serde_json::Value>],
        _text_field: &str,
        _min_confidence: f64,
    ) -> HashMap<String, serde_json::Value> {
        self.build_passthrough_analysis(
            query_results,
            "Rust版ではスタブ実装。将来ONNX Runtime統合時に有効化。",
        )
    }

    fn get_model_info(&self) -> HashMap<String, serde_json::Value> {
        let mut info = HashMap::new();
        info.insert(
            "model_name".to_string(),
            serde_json::Value::String(
                "tabularisai/multilingual-sentiment-analysis (stub)".to_string(),
            ),
        );
        info.insert(
            "supported_languages".to_string(),
            serde_json::json!(SUPPORTED_LANGUAGES),
        );
        info.insert(
            "sentiment_levels".to_string(),
            serde_json::json!(SENTIMENT_MAP.iter().map(|(_, l)| l).collect::<Vec<_>>()),
        );
        info.insert(
            "is_initialized".to_string(),
            serde_json::Value::Bool(self.initialized),
        );
        info.insert(
            "device".to_string(),
            serde_json::Value::String("stub (no device)".to_string()),
        );
        info
    }
}
