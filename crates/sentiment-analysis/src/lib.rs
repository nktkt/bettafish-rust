//! BettaFish Sentiment Analysis Model
//!
//! Python の SentimentAnalysisModel パッケージの Rust 実装。
//! 多言語感情分析、BERT トピック検出、ファインチューンモデル設定、
//! トレーニングパイプライン、ML モデルインターフェースを提供。
//!
//! ## モジュール構成
//! - **core**: SentimentLevel, SentimentResult, BatchSentimentResult
//! - **analyzer**: MultilingualSentimentAnalyzer トレイト + StubAnalyzer
//! - **bert_topic**: BertTopicDetector トレイト + StubBertDetector
//! - **model_config**: ModelConfig, WeiboSentimentConfig, SmallQwenConfig
//! - **training**: TrainingConfig, TrainingResult, Trainer トレイト
//! - **ml_models**: MLModelType, MLModelConfig, MLPredictor トレイト

#![allow(dead_code)]

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use tracing::{info, warn};

// =============================================================================
// Sentiment Analysis Core
// =============================================================================

/// 感情レベル (5段階)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SentimentLevel {
    /// 非常負面 (0)
    VeryNegative = 0,
    /// 負面 (1)
    Negative = 1,
    /// 中性 (2)
    Neutral = 2,
    /// 正面 (3)
    Positive = 3,
    /// 非常正面 (4)
    VeryPositive = 4,
}

impl SentimentLevel {
    /// 数値からレベルに変換
    pub fn from_value(value: i32) -> Self {
        match value {
            0 => Self::VeryNegative,
            1 => Self::Negative,
            2 => Self::Neutral,
            3 => Self::Positive,
            4 => Self::VeryPositive,
            _ => Self::Neutral,
        }
    }

    /// 中国語ラベルを取得
    pub fn chinese_label(&self) -> &'static str {
        match self {
            Self::VeryNegative => "非常负面",
            Self::Negative => "负面",
            Self::Neutral => "中性",
            Self::Positive => "正面",
            Self::VeryPositive => "非常正面",
        }
    }

    /// 数値を取得
    pub fn value(&self) -> i32 {
        *self as i32
    }

    /// 全てのレベルを返す
    pub fn all() -> &'static [SentimentLevel] {
        &[
            Self::VeryNegative,
            Self::Negative,
            Self::Neutral,
            Self::Positive,
            Self::VeryPositive,
        ]
    }
}

impl std::fmt::Display for SentimentLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.chinese_label())
    }
}

/// 感情分析結果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentResult {
    /// 入力テキスト
    pub text: String,
    /// 感情ラベル
    pub sentiment_label: String,
    /// 信頼度 (0.0 ~ 1.0)
    pub confidence: f64,
    /// 確率分布 (ラベル -> 確率)
    pub probability_distribution: HashMap<String, f64>,
    /// 成功フラグ
    pub success: bool,
    /// エラーメッセージ
    pub error_message: Option<String>,
}

impl Default for SentimentResult {
    fn default() -> Self {
        Self {
            text: String::new(),
            sentiment_label: SentimentLevel::Neutral.chinese_label().to_string(),
            confidence: 0.0,
            probability_distribution: HashMap::new(),
            success: false,
            error_message: None,
        }
    }
}

impl SentimentResult {
    /// 新しい成功結果を作成
    pub fn new_success(
        text: &str,
        level: SentimentLevel,
        confidence: f64,
        distribution: HashMap<String, f64>,
    ) -> Self {
        Self {
            text: text.to_string(),
            sentiment_label: level.chinese_label().to_string(),
            confidence,
            probability_distribution: distribution,
            success: true,
            error_message: None,
        }
    }

    /// 新しい失敗結果を作成
    pub fn new_error(text: &str, error: &str) -> Self {
        Self {
            text: text.to_string(),
            sentiment_label: SentimentLevel::Neutral.chinese_label().to_string(),
            confidence: 0.0,
            probability_distribution: HashMap::new(),
            success: false,
            error_message: Some(error.to_string()),
        }
    }

    /// 感情レベルを取得
    pub fn sentiment_level(&self) -> SentimentLevel {
        match self.sentiment_label.as_str() {
            "非常负面" => SentimentLevel::VeryNegative,
            "负面" => SentimentLevel::Negative,
            "中性" => SentimentLevel::Neutral,
            "正面" => SentimentLevel::Positive,
            "非常正面" => SentimentLevel::VeryPositive,
            _ => SentimentLevel::Neutral,
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
}

impl Default for BatchSentimentResult {
    fn default() -> Self {
        Self {
            results: Vec::new(),
            total_processed: 0,
            success_count: 0,
            failed_count: 0,
            average_confidence: 0.0,
        }
    }
}

impl BatchSentimentResult {
    /// 結果から統計を計算
    pub fn from_results(results: Vec<SentimentResult>) -> Self {
        let total_processed = results.len();
        let success_count = results.iter().filter(|r| r.success).count();
        let failed_count = total_processed - success_count;
        let average_confidence = if success_count > 0 {
            results
                .iter()
                .filter(|r| r.success)
                .map(|r| r.confidence)
                .sum::<f64>()
                / success_count as f64
        } else {
            0.0
        };

        Self {
            results,
            total_processed,
            success_count,
            failed_count,
            average_confidence,
        }
    }

    /// 感情分布を取得
    pub fn sentiment_distribution(&self) -> HashMap<String, usize> {
        let mut distribution = HashMap::new();
        for result in &self.results {
            if result.success {
                *distribution.entry(result.sentiment_label.clone()).or_insert(0) += 1;
            }
        }
        distribution
    }
}

/// サポート言語一覧 (22言語)
pub const SUPPORTED_LANGUAGES: &[&str] = &[
    "中文",
    "英文",
    "西班牙文",
    "阿拉伯文",
    "日文",
    "韩文",
    "德文",
    "法文",
    "意大利文",
    "葡萄牙文",
    "俄文",
    "荷兰文",
    "波兰文",
    "土耳其文",
    "丹麦文",
    "希腊文",
    "芬兰文",
    "瑞典文",
    "挪威文",
    "匈牙利文",
    "捷克文",
    "保加利亚文",
];

/// 感情ラベルマッピング
pub const SENTIMENT_LABELS: &[(i32, &str)] = &[
    (0, "非常负面"),
    (1, "负面"),
    (2, "中性"),
    (3, "正面"),
    (4, "非常正面"),
];

// =============================================================================
// Multilingual Sentiment Analyzer
// =============================================================================

/// 多言語感情分析トレイト
#[async_trait]
pub trait MultilingualSentimentAnalyzer: Send + Sync {
    /// モデルを初期化
    fn initialize(&mut self) -> bool;

    /// 単一テキストの感情分析
    async fn analyze_single_text(&self, text: &str) -> SentimentResult;

    /// バッチ感情分析
    async fn analyze_batch(
        &self,
        texts: &[String],
        show_progress: bool,
    ) -> BatchSentimentResult;

    /// クエリ結果の感情分析
    async fn analyze_query_results(
        &self,
        results: &[HashMap<String, Value>],
        text_field: &str,
        min_confidence: f64,
    ) -> Value;

    /// モデルが利用可能か
    fn is_available(&self) -> bool;

    /// モデルを無効化
    fn disable(&mut self, reason: &str);
}

/// スタブ分析器 (実モデルなし)
pub struct StubAnalyzer {
    available: bool,
    disabled: bool,
    disable_reason: Option<String>,
}

impl StubAnalyzer {
    /// 新しいスタブ分析器を作成
    pub fn new() -> Self {
        info!("StubAnalyzer: スタブ感情分析器を作成（モデル推論なし）");
        Self {
            available: true,
            disabled: false,
            disable_reason: None,
        }
    }
}

impl Default for StubAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MultilingualSentimentAnalyzer for StubAnalyzer {
    fn initialize(&mut self) -> bool {
        warn!("StubAnalyzer: スタブ実装のため初期化をスキップ");
        self.available = true;
        true
    }

    async fn analyze_single_text(&self, text: &str) -> SentimentResult {
        let mut distribution = HashMap::new();
        for level in SentimentLevel::all() {
            let prob = if *level == SentimentLevel::Neutral {
                0.5
            } else {
                0.125
            };
            distribution.insert(level.chinese_label().to_string(), prob);
        }

        SentimentResult {
            text: text.to_string(),
            sentiment_label: SentimentLevel::Neutral.chinese_label().to_string(),
            confidence: 0.5,
            probability_distribution: distribution,
            success: true,
            error_message: Some(
                "スタブ実装: 実際の推論なし、Neutral/0.5を返却".to_string(),
            ),
        }
    }

    async fn analyze_batch(
        &self,
        texts: &[String],
        _show_progress: bool,
    ) -> BatchSentimentResult {
        let mut results = Vec::new();
        for text in texts {
            results.push(self.analyze_single_text(text).await);
        }
        BatchSentimentResult::from_results(results)
    }

    async fn analyze_query_results(
        &self,
        results: &[HashMap<String, Value>],
        _text_field: &str,
        _min_confidence: f64,
    ) -> Value {
        serde_json::json!({
            "sentiment_analysis": {
                "available": false,
                "reason": "スタブ実装: ONNX Runtime統合時に有効化",
                "total_analyzed": 0,
                "success_rate": format!("0/{}", results.len()),
                "average_confidence": 0.0,
                "sentiment_distribution": {},
                "high_confidence_results": [],
                "summary": "情感分析未执行：スタブ実装"
            }
        })
    }

    fn is_available(&self) -> bool {
        self.available && !self.disabled
    }

    fn disable(&mut self, reason: &str) {
        self.disabled = true;
        self.disable_reason = Some(reason.to_string());
        warn!("StubAnalyzer: 無効化されました - {}", reason);
    }
}

// =============================================================================
// BERT Topic Detection
// =============================================================================

/// テキスト分類設定
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextClassificationConfig {
    /// モデル名
    pub model_name: String,
    /// 最大シーケンス長
    pub max_length: usize,
    /// バッチサイズ
    pub batch_size: usize,
    /// デバイス ("cpu" or "cuda")
    pub device: String,
    /// ラベルマッピング
    pub label_mapping: HashMap<i32, String>,
}

impl Default for TextClassificationConfig {
    fn default() -> Self {
        Self {
            model_name: "bert-base-chinese".to_string(),
            max_length: 512,
            batch_size: 32,
            device: "cpu".to_string(),
            label_mapping: HashMap::new(),
        }
    }
}

/// BERT トピック検出トレイト
#[async_trait]
pub trait BertTopicDetector: Send + Sync {
    /// テキストから上位 k 個のトピックを予測
    async fn predict_topk(&self, text: &str, top_k: usize) -> Vec<(String, f64)>;
}

/// スタブ BERT 検出器
pub struct StubBertDetector {
    config: TextClassificationConfig,
}

impl StubBertDetector {
    /// 新しいスタブ検出器を作成
    pub fn new() -> Self {
        Self {
            config: TextClassificationConfig::default(),
        }
    }

    /// 設定付きで作成
    pub fn with_config(config: TextClassificationConfig) -> Self {
        Self { config }
    }

    /// 設定を取得
    pub fn config(&self) -> &TextClassificationConfig {
        &self.config
    }
}

impl Default for StubBertDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl BertTopicDetector for StubBertDetector {
    async fn predict_topk(&self, _text: &str, top_k: usize) -> Vec<(String, f64)> {
        // スタブ: 一般的なトピックを返す
        let default_topics = vec![
            ("社会热点".to_string(), 0.3),
            ("科技创新".to_string(), 0.2),
            ("经济金融".to_string(), 0.15),
            ("文化娱乐".to_string(), 0.1),
            ("政治外交".to_string(), 0.08),
            ("教育就业".to_string(), 0.07),
            ("环境健康".to_string(), 0.05),
            ("体育竞技".to_string(), 0.03),
            ("军事安全".to_string(), 0.01),
            ("其他".to_string(), 0.01),
        ];

        default_topics.into_iter().take(top_k).collect()
    }
}

// =============================================================================
// Fine-tuned Model Configs
// =============================================================================

/// モデル設定
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// モデル名
    pub model_name: String,
    /// モデルパス
    pub model_path: String,
    /// トークナイザーパス
    pub tokenizer_path: String,
    /// デバイス ("cpu", "cuda", "mps")
    pub device: String,
    /// 最大シーケンス長
    pub max_length: usize,
    /// バッチサイズ
    pub batch_size: usize,
    /// 追加パラメータ
    pub extra_params: HashMap<String, Value>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_name: String::new(),
            model_path: String::new(),
            tokenizer_path: String::new(),
            device: "cpu".to_string(),
            max_length: 512,
            batch_size: 32,
            extra_params: HashMap::new(),
        }
    }
}

impl ModelConfig {
    /// 新しい ModelConfig を作成
    pub fn new(model_name: &str, model_path: &str) -> Self {
        Self {
            model_name: model_name.to_string(),
            model_path: model_path.to_string(),
            tokenizer_path: model_path.to_string(),
            ..Default::default()
        }
    }
}

/// 微博感情分析設定 (Weibo Sentiment Config)
pub struct WeiboSentimentConfig;

impl WeiboSentimentConfig {
    /// BertChinese-LoRA 設定
    pub fn bert_chinese_lora() -> ModelConfig {
        ModelConfig {
            model_name: "BertChinese-LoRA".to_string(),
            model_path: "models/weibo_sentiment/bert_chinese_lora".to_string(),
            tokenizer_path: "bert-base-chinese".to_string(),
            device: "cpu".to_string(),
            max_length: 512,
            batch_size: 32,
            extra_params: {
                let mut params = HashMap::new();
                params.insert("lora_r".to_string(), serde_json::json!(8));
                params.insert("lora_alpha".to_string(), serde_json::json!(16));
                params.insert(
                    "lora_target_modules".to_string(),
                    serde_json::json!(["query", "key", "value"]),
                );
                params.insert("num_labels".to_string(), serde_json::json!(5));
                params
            },
        }
    }

    /// GPT2-AdapterTuning 設定
    pub fn gpt2_adapter_tuning() -> ModelConfig {
        ModelConfig {
            model_name: "GPT2-AdapterTuning".to_string(),
            model_path: "models/weibo_sentiment/gpt2_adapter".to_string(),
            tokenizer_path: "gpt2".to_string(),
            device: "cpu".to_string(),
            max_length: 512,
            batch_size: 16,
            extra_params: {
                let mut params = HashMap::new();
                params.insert("adapter_size".to_string(), serde_json::json!(64));
                params.insert("num_labels".to_string(), serde_json::json!(5));
                params
            },
        }
    }

    /// GPT2-LoRA 設定
    pub fn gpt2_lora() -> ModelConfig {
        ModelConfig {
            model_name: "GPT2-LoRA".to_string(),
            model_path: "models/weibo_sentiment/gpt2_lora".to_string(),
            tokenizer_path: "gpt2".to_string(),
            device: "cpu".to_string(),
            max_length: 512,
            batch_size: 16,
            extra_params: {
                let mut params = HashMap::new();
                params.insert("lora_r".to_string(), serde_json::json!(4));
                params.insert("lora_alpha".to_string(), serde_json::json!(8));
                params.insert("num_labels".to_string(), serde_json::json!(5));
                params
            },
        }
    }

    /// 全ての利用可能な設定名を返す
    pub fn available_configs() -> Vec<&'static str> {
        vec!["BertChinese-LoRA", "GPT2-AdapterTuning", "GPT2-LoRA"]
    }
}

/// Small Qwen 設定
pub struct SmallQwenConfig;

impl SmallQwenConfig {
    /// デフォルト設定
    pub fn default_config() -> ModelConfig {
        ModelConfig {
            model_name: "SmallQwen-Sentiment".to_string(),
            model_path: "models/small_qwen/sentiment".to_string(),
            tokenizer_path: "Qwen/Qwen2-0.5B".to_string(),
            device: "cpu".to_string(),
            max_length: 1024,
            batch_size: 8,
            extra_params: {
                let mut params = HashMap::new();
                params.insert("num_labels".to_string(), serde_json::json!(5));
                params.insert("use_flash_attention".to_string(), serde_json::json!(false));
                params
            },
        }
    }
}

// =============================================================================
// Training Pipeline Interface
// =============================================================================

/// トレーニング設定
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// エポック数
    pub epochs: usize,
    /// バッチサイズ
    pub batch_size: usize,
    /// 学習率
    pub learning_rate: f64,
    /// ウォームアップステップ数
    pub warmup_steps: usize,
    /// 重み減衰
    pub weight_decay: f64,
    /// LoRA ランク
    pub lora_r: usize,
    /// LoRA アルファ
    pub lora_alpha: usize,
    /// 出力ディレクトリ
    pub output_dir: String,
    /// 評価ステップ間隔
    pub eval_steps: usize,
    /// 保存ステップ間隔
    pub save_steps: usize,
    /// 最大勾配ノルム
    pub max_grad_norm: f64,
    /// FP16 使用フラグ
    pub fp16: bool,
    /// 追加パラメータ
    pub extra_params: HashMap<String, Value>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 3,
            batch_size: 16,
            learning_rate: 2e-5,
            warmup_steps: 500,
            weight_decay: 0.01,
            lora_r: 8,
            lora_alpha: 16,
            output_dir: "training_output".to_string(),
            eval_steps: 100,
            save_steps: 500,
            max_grad_norm: 1.0,
            fp16: false,
            extra_params: HashMap::new(),
        }
    }
}

impl TrainingConfig {
    /// BertChinese-LoRA 用デフォルト設定
    pub fn for_bert_chinese_lora() -> Self {
        Self {
            epochs: 5,
            batch_size: 32,
            learning_rate: 2e-5,
            warmup_steps: 500,
            weight_decay: 0.01,
            lora_r: 8,
            lora_alpha: 16,
            output_dir: "output/bert_chinese_lora".to_string(),
            eval_steps: 200,
            save_steps: 1000,
            max_grad_norm: 1.0,
            fp16: true,
            extra_params: HashMap::new(),
        }
    }

    /// GPT2-LoRA 用デフォルト設定
    pub fn for_gpt2_lora() -> Self {
        Self {
            epochs: 3,
            batch_size: 16,
            learning_rate: 5e-5,
            warmup_steps: 200,
            weight_decay: 0.01,
            lora_r: 4,
            lora_alpha: 8,
            output_dir: "output/gpt2_lora".to_string(),
            eval_steps: 100,
            save_steps: 500,
            max_grad_norm: 1.0,
            fp16: false,
            extra_params: HashMap::new(),
        }
    }

    /// GPT2-AdapterTuning 用デフォルト設定
    pub fn for_gpt2_adapter() -> Self {
        Self {
            epochs: 3,
            batch_size: 16,
            learning_rate: 1e-4,
            warmup_steps: 100,
            weight_decay: 0.0,
            lora_r: 0,
            lora_alpha: 0,
            output_dir: "output/gpt2_adapter".to_string(),
            eval_steps: 100,
            save_steps: 500,
            max_grad_norm: 1.0,
            fp16: false,
            extra_params: {
                let mut params = HashMap::new();
                params.insert("adapter_size".to_string(), serde_json::json!(64));
                params
            },
        }
    }

    /// SmallQwen 用デフォルト設定
    pub fn for_small_qwen() -> Self {
        Self {
            epochs: 2,
            batch_size: 8,
            learning_rate: 1e-5,
            warmup_steps: 100,
            weight_decay: 0.01,
            lora_r: 4,
            lora_alpha: 8,
            output_dir: "output/small_qwen".to_string(),
            eval_steps: 50,
            save_steps: 200,
            max_grad_norm: 1.0,
            fp16: false,
            extra_params: HashMap::new(),
        }
    }
}

/// トレーニング結果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    /// 出力モデルパス
    pub model_path: String,
    /// メトリクス
    pub metrics: HashMap<String, f64>,
    /// トレーニング時間 (秒)
    pub training_time: f64,
    /// 成功フラグ
    pub success: bool,
    /// エラーメッセージ
    pub error_message: Option<String>,
}

impl Default for TrainingResult {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            metrics: HashMap::new(),
            training_time: 0.0,
            success: false,
            error_message: None,
        }
    }
}

/// トレーナートレイト
#[async_trait]
pub trait Trainer: Send + Sync {
    /// モデルをトレーニング
    async fn train(
        &self,
        config: &TrainingConfig,
        dataset_path: &str,
    ) -> anyhow::Result<TrainingResult>;

    /// トレーナー名を取得
    fn name(&self) -> &str;
}

/// スタブトレーナー (実際のトレーニングなし)
pub struct StubTrainer {
    trainer_name: String,
}

impl StubTrainer {
    pub fn new(name: &str) -> Self {
        Self {
            trainer_name: name.to_string(),
        }
    }
}

#[async_trait]
impl Trainer for StubTrainer {
    async fn train(
        &self,
        config: &TrainingConfig,
        dataset_path: &str,
    ) -> anyhow::Result<TrainingResult> {
        warn!(
            "StubTrainer({}): トレーニングは実行されません (dataset: {})",
            self.trainer_name, dataset_path
        );
        Ok(TrainingResult {
            model_path: config.output_dir.clone(),
            metrics: {
                let mut m = HashMap::new();
                m.insert("accuracy".to_string(), 0.0);
                m.insert("f1_score".to_string(), 0.0);
                m.insert("loss".to_string(), f64::NAN);
                m
            },
            training_time: 0.0,
            success: false,
            error_message: Some(format!(
                "スタブ実装: {} トレーナーではトレーニングは実行されません",
                self.trainer_name
            )),
        })
    }

    fn name(&self) -> &str {
        &self.trainer_name
    }
}

// =============================================================================
// Machine Learning Models
// =============================================================================

/// ML モデルタイプ
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MLModelType {
    /// ナイーブベイズ
    Bayes,
    /// サポートベクターマシン
    SVM,
    /// XGBoost
    XGBoost,
    /// LSTM
    LSTM,
    /// BERT
    BERT,
}

impl MLModelType {
    /// モデルタイプ名を取得
    pub fn name(&self) -> &'static str {
        match self {
            Self::Bayes => "NaiveBayes",
            Self::SVM => "SVM",
            Self::XGBoost => "XGBoost",
            Self::LSTM => "LSTM",
            Self::BERT => "BERT",
        }
    }

    /// 全てのモデルタイプを返す
    pub fn all() -> &'static [MLModelType] {
        &[
            Self::Bayes,
            Self::SVM,
            Self::XGBoost,
            Self::LSTM,
            Self::BERT,
        ]
    }
}

impl std::fmt::Display for MLModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// ML モデル設定
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLModelConfig {
    /// モデルタイプ
    pub model_type: MLModelType,
    /// モデルパス
    pub model_path: String,
    /// ベクトライザーパス (Bayes, SVM 用)
    pub vectorizer_path: Option<String>,
    /// モデル固有パラメータ
    pub params: HashMap<String, Value>,
}

impl MLModelConfig {
    /// 新しい設定を作成
    pub fn new(model_type: MLModelType, model_path: &str) -> Self {
        Self {
            model_type,
            model_path: model_path.to_string(),
            vectorizer_path: None,
            params: HashMap::new(),
        }
    }
}

/// ML 予測器トレイト
#[async_trait]
pub trait MLPredictor: Send + Sync {
    /// テキストの感情を予測
    async fn predict(&self, text: &str) -> SentimentResult;

    /// モデルタイプを取得
    fn model_type(&self) -> MLModelType;

    /// モデルがロード済みか
    fn is_loaded(&self) -> bool;
}

/// スタブ ML 予測器
pub struct StubMLPredictor {
    model_type: MLModelType,
}

impl StubMLPredictor {
    /// 指定タイプで作成
    pub fn new(model_type: MLModelType) -> Self {
        info!(
            "StubMLPredictor: {} タイプのスタブ予測器を作成",
            model_type.name()
        );
        Self { model_type }
    }
}

#[async_trait]
impl MLPredictor for StubMLPredictor {
    async fn predict(&self, text: &str) -> SentimentResult {
        let mut distribution = HashMap::new();
        for level in SentimentLevel::all() {
            distribution.insert(level.chinese_label().to_string(), 0.2);
        }

        SentimentResult {
            text: text.to_string(),
            sentiment_label: SentimentLevel::Neutral.chinese_label().to_string(),
            confidence: 0.2,
            probability_distribution: distribution,
            success: false,
            error_message: Some(format!(
                "スタブ実装: {} モデルは利用不可",
                self.model_type.name()
            )),
        }
    }

    fn model_type(&self) -> MLModelType {
        self.model_type
    }

    fn is_loaded(&self) -> bool {
        false
    }
}

// =============================================================================
// テスト
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentiment_level_values() {
        assert_eq!(SentimentLevel::VeryNegative.value(), 0);
        assert_eq!(SentimentLevel::Negative.value(), 1);
        assert_eq!(SentimentLevel::Neutral.value(), 2);
        assert_eq!(SentimentLevel::Positive.value(), 3);
        assert_eq!(SentimentLevel::VeryPositive.value(), 4);
    }

    #[test]
    fn test_sentiment_level_chinese_labels() {
        assert_eq!(SentimentLevel::VeryNegative.chinese_label(), "非常负面");
        assert_eq!(SentimentLevel::Negative.chinese_label(), "负面");
        assert_eq!(SentimentLevel::Neutral.chinese_label(), "中性");
        assert_eq!(SentimentLevel::Positive.chinese_label(), "正面");
        assert_eq!(SentimentLevel::VeryPositive.chinese_label(), "非常正面");
    }

    #[test]
    fn test_sentiment_level_from_value() {
        assert_eq!(SentimentLevel::from_value(0), SentimentLevel::VeryNegative);
        assert_eq!(SentimentLevel::from_value(2), SentimentLevel::Neutral);
        assert_eq!(SentimentLevel::from_value(4), SentimentLevel::VeryPositive);
        assert_eq!(SentimentLevel::from_value(99), SentimentLevel::Neutral);
    }

    #[test]
    fn test_sentiment_result_default() {
        let result = SentimentResult::default();
        assert_eq!(result.sentiment_label, "中性");
        assert!(!result.success);
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_sentiment_result_new_success() {
        let mut dist = HashMap::new();
        dist.insert("正面".to_string(), 0.8);
        dist.insert("中性".to_string(), 0.15);
        dist.insert("负面".to_string(), 0.05);

        let result =
            SentimentResult::new_success("测试文本", SentimentLevel::Positive, 0.8, dist);
        assert!(result.success);
        assert_eq!(result.sentiment_label, "正面");
        assert_eq!(result.confidence, 0.8);
        assert_eq!(result.sentiment_level(), SentimentLevel::Positive);
    }

    #[test]
    fn test_batch_sentiment_result() {
        let results = vec![
            SentimentResult::new_success(
                "good",
                SentimentLevel::Positive,
                0.9,
                HashMap::new(),
            ),
            SentimentResult::new_error("bad", "error"),
            SentimentResult::new_success(
                "ok",
                SentimentLevel::Neutral,
                0.7,
                HashMap::new(),
            ),
        ];

        let batch = BatchSentimentResult::from_results(results);
        assert_eq!(batch.total_processed, 3);
        assert_eq!(batch.success_count, 2);
        assert_eq!(batch.failed_count, 1);
        assert!((batch.average_confidence - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_supported_languages() {
        assert_eq!(SUPPORTED_LANGUAGES.len(), 22);
        assert!(SUPPORTED_LANGUAGES.contains(&"中文"));
        assert!(SUPPORTED_LANGUAGES.contains(&"日文"));
    }

    #[test]
    fn test_model_config_weibo() {
        let bert_config = WeiboSentimentConfig::bert_chinese_lora();
        assert_eq!(bert_config.model_name, "BertChinese-LoRA");
        assert_eq!(bert_config.max_length, 512);

        let gpt2_adapter = WeiboSentimentConfig::gpt2_adapter_tuning();
        assert_eq!(gpt2_adapter.model_name, "GPT2-AdapterTuning");

        let gpt2_lora = WeiboSentimentConfig::gpt2_lora();
        assert_eq!(gpt2_lora.model_name, "GPT2-LoRA");

        assert_eq!(WeiboSentimentConfig::available_configs().len(), 3);
    }

    #[test]
    fn test_model_config_small_qwen() {
        let config = SmallQwenConfig::default_config();
        assert_eq!(config.model_name, "SmallQwen-Sentiment");
        assert_eq!(config.max_length, 1024);
    }

    #[test]
    fn test_training_config_defaults() {
        let default = TrainingConfig::default();
        assert_eq!(default.epochs, 3);
        assert_eq!(default.batch_size, 16);

        let bert = TrainingConfig::for_bert_chinese_lora();
        assert_eq!(bert.epochs, 5);
        assert!(bert.fp16);

        let gpt2_lora = TrainingConfig::for_gpt2_lora();
        assert_eq!(gpt2_lora.lora_r, 4);

        let gpt2_adapter = TrainingConfig::for_gpt2_adapter();
        assert!(gpt2_adapter.extra_params.contains_key("adapter_size"));

        let qwen = TrainingConfig::for_small_qwen();
        assert_eq!(qwen.epochs, 2);
    }

    #[test]
    fn test_ml_model_types() {
        assert_eq!(MLModelType::all().len(), 5);
        assert_eq!(MLModelType::Bayes.name(), "NaiveBayes");
        assert_eq!(MLModelType::SVM.name(), "SVM");
        assert_eq!(MLModelType::XGBoost.name(), "XGBoost");
        assert_eq!(MLModelType::LSTM.name(), "LSTM");
        assert_eq!(MLModelType::BERT.name(), "BERT");
    }

    #[tokio::test]
    async fn test_stub_analyzer() {
        let mut analyzer = StubAnalyzer::new();
        assert!(analyzer.is_available());

        analyzer.initialize();
        assert!(analyzer.is_available());

        let result = analyzer.analyze_single_text("测试文本").await;
        assert!(result.success);
        assert_eq!(result.sentiment_label, "中性");
        assert_eq!(result.confidence, 0.5);

        let batch = analyzer
            .analyze_batch(&["文本1".to_string(), "文本2".to_string()], false)
            .await;
        assert_eq!(batch.total_processed, 2);
        assert_eq!(batch.success_count, 2);

        analyzer.disable("テスト");
        assert!(!analyzer.is_available());
    }

    #[tokio::test]
    async fn test_stub_bert_detector() {
        let detector = StubBertDetector::new();
        let topics = detector.predict_topk("测试文本", 3).await;
        assert_eq!(topics.len(), 3);
        assert_eq!(topics[0].0, "社会热点");
    }

    #[tokio::test]
    async fn test_stub_ml_predictor() {
        let predictor = StubMLPredictor::new(MLModelType::BERT);
        assert!(!predictor.is_loaded());
        assert_eq!(predictor.model_type(), MLModelType::BERT);

        let result = predictor.predict("测试").await;
        assert!(!result.success);
    }
}
