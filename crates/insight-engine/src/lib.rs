//! BettaFish InsightEngine
//!
//! Python の InsightEngine の Rust 実装。
//! ローカルデータベースの世論データを分析する深度検索 AI エージェント。
//! QueryEngine との違い:
//! - MediaCrawlerDB (DB クエリ) を使用 (Tavily の代わり)
//! - KeywordOptimizer (クエリをインターネットスラングに変換)
//! - SentimentAnalyzer (22言語5段階の感情分析)
//! - KMeans クラスタリングによる結果サンプリング
//! - 5つの DB 検索ツール

// =============================================================================
// モジュール構成
// =============================================================================

pub mod state;
pub mod prompts;
pub mod tools;
pub mod nodes;
mod agent;

pub use agent::DeepSearchAgent;
pub use state::State;
pub use tools::db::{MediaCrawlerDB, QueryResult, DBResponse};
pub use tools::keyword_optimizer::{KeywordOptimizer, KeywordOptimizationResponse};
pub use tools::sentiment::{
    SentimentAnalyzer, SentimentResult, BatchSentimentResult, StubSentimentAnalyzer,
};

/// バージョン情報
pub const VERSION: &str = "1.0.0";

/// クラスタリング有効化フラグ
pub const ENABLE_CLUSTERING: bool = true;
/// クラスタリング後最大結果数
pub const MAX_CLUSTERED_RESULTS: usize = 50;
/// 各クラスタの返却結果数
pub const RESULTS_PER_CLUSTER: usize = 5;
