//! BettaFish QueryEngine
//!
//! Python の QueryEngine の Rust 実装。
//! フレームワークレスの深度検索 AI エージェント。

pub mod nodes;
pub mod prompts;
pub mod state;
pub mod tools;

mod agent;

pub use agent::DeepSearchAgent;
pub use state::State;

/// バージョン情報
pub const VERSION: &str = "1.0.0";
