//! BettaFish 設定管理モジュール
//!
//! Pydantic Settings の Rust 実装。環境変数と .env ファイルから設定を自動ロード。

use serde::Deserialize;
use std::env;
use std::path::{Path, PathBuf};

/// グローバル設定構造体
/// Python の config.py と QueryEngine/utils/config.py を統合
#[derive(Debug, Clone, Deserialize)]
pub struct Settings {
    // ==================== Flask サーバー ====================
    pub host: String,
    pub port: u16,

    // ==================== データベース ====================
    pub db_dialect: String,
    pub db_host: String,
    pub db_port: u16,
    pub db_user: String,
    pub db_password: String,
    pub db_name: String,
    pub db_charset: String,

    // ==================== InsightEngine LLM ====================
    pub insight_engine_api_key: String,
    pub insight_engine_base_url: Option<String>,
    pub insight_engine_model_name: String,

    // ==================== MediaEngine LLM ====================
    pub media_engine_api_key: String,
    pub media_engine_base_url: Option<String>,
    pub media_engine_model_name: String,

    // ==================== QueryEngine LLM ====================
    pub query_engine_api_key: String,
    pub query_engine_base_url: Option<String>,
    pub query_engine_model_name: String,

    // ==================== ReportEngine LLM ====================
    pub report_engine_api_key: String,
    pub report_engine_base_url: Option<String>,
    pub report_engine_model_name: String,

    // ==================== MindSpider LLM ====================
    pub mindspider_api_key: String,
    pub mindspider_base_url: Option<String>,
    pub mindspider_model_name: String,

    // ==================== ForumHost LLM ====================
    pub forum_host_api_key: String,
    pub forum_host_base_url: Option<String>,
    pub forum_host_model_name: String,

    // ==================== KeywordOptimizer LLM ====================
    pub keyword_optimizer_api_key: String,
    pub keyword_optimizer_base_url: Option<String>,
    pub keyword_optimizer_model_name: String,

    // ==================== 検索ツール ====================
    pub tavily_api_key: String,
    pub search_tool_type: String,
    pub bocha_base_url: String,
    pub bocha_web_search_api_key: String,
    pub anspire_base_url: String,
    pub anspire_api_key: String,

    // ==================== InsightEngine 検索パラメータ ====================
    pub default_search_hot_content_limit: usize,
    pub default_search_topic_globally_limit_per_table: usize,
    pub default_search_topic_by_date_limit_per_table: usize,
    pub default_get_comments_for_topic_limit: usize,
    pub default_search_topic_on_platform_limit: usize,
    pub max_search_results_for_llm: usize,
    pub max_high_confidence_sentiment_results: usize,

    // ==================== 共通パラメータ ====================
    pub max_reflections: usize,
    pub max_paragraphs: usize,
    pub search_timeout: u64,
    pub max_content_length: usize,
    pub max_search_results: usize,
    pub search_content_max_length: usize,

    // ==================== 出力設定 ====================
    pub output_dir: String,
    pub save_intermediate_states: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 5000,
            db_dialect: "postgresql".to_string(),
            db_host: "localhost".to_string(),
            db_port: 5432,
            db_user: "root".to_string(),
            db_password: "".to_string(),
            db_name: "bettafish".to_string(),
            db_charset: "utf8mb4".to_string(),
            insight_engine_api_key: String::new(),
            insight_engine_base_url: Some("https://api.moonshot.cn/v1".to_string()),
            insight_engine_model_name: "kimi-k2-0711-preview".to_string(),
            media_engine_api_key: String::new(),
            media_engine_base_url: Some("https://aihubmix.com/v1".to_string()),
            media_engine_model_name: "gemini-2.5-pro".to_string(),
            query_engine_api_key: String::new(),
            query_engine_base_url: Some("https://api.deepseek.com".to_string()),
            query_engine_model_name: "deepseek-chat".to_string(),
            report_engine_api_key: String::new(),
            report_engine_base_url: None,
            report_engine_model_name: "gemini-2.5-pro".to_string(),
            mindspider_api_key: String::new(),
            mindspider_base_url: None,
            mindspider_model_name: "deepseek-chat".to_string(),
            forum_host_api_key: String::new(),
            forum_host_base_url: None,
            forum_host_model_name: "Qwen3-235B-A22B".to_string(),
            keyword_optimizer_api_key: String::new(),
            keyword_optimizer_base_url: None,
            keyword_optimizer_model_name: "Qwen3-235B-A22B".to_string(),
            tavily_api_key: String::new(),
            search_tool_type: "AnspireAPI".to_string(),
            bocha_base_url: "https://api.bocha.cn/v1/ai-search".to_string(),
            bocha_web_search_api_key: String::new(),
            anspire_base_url: "https://plugin.anspire.cn/api/ntsearch/search".to_string(),
            anspire_api_key: String::new(),
            default_search_hot_content_limit: 100,
            default_search_topic_globally_limit_per_table: 50,
            default_search_topic_by_date_limit_per_table: 100,
            default_get_comments_for_topic_limit: 500,
            default_search_topic_on_platform_limit: 200,
            max_search_results_for_llm: 0,
            max_high_confidence_sentiment_results: 0,
            max_reflections: 2,
            max_paragraphs: 5,
            search_timeout: 240,
            max_content_length: 500_000,
            max_search_results: 20,
            search_content_max_length: 20_000,
            output_dir: "reports".to_string(),
            save_intermediate_states: true,
        }
    }
}

impl Settings {
    /// .env ファイルと環境変数から設定をロード
    pub fn load() -> Self {
        // .env ファイルを探す（CWD 優先、次にプロジェクトルート）
        let cwd_env = PathBuf::from(".env");
        if cwd_env.exists() {
            let _ = dotenvy::from_path(&cwd_env);
        } else {
            let _ = dotenvy::dotenv();
        }

        let mut settings = Self::default();
        settings.load_from_env();
        settings
    }

    /// 指定パスの .env ファイルから設定をロード
    pub fn load_from_path(path: &Path) -> Self {
        let _ = dotenvy::from_path(path);
        let mut settings = Self::default();
        settings.load_from_env();
        settings
    }

    fn load_from_env(&mut self) {
        macro_rules! env_str {
            ($field:ident, $key:expr) => {
                if let Ok(val) = env::var($key) {
                    if !val.is_empty() {
                        self.$field = val;
                    }
                }
            };
        }
        macro_rules! env_opt_str {
            ($field:ident, $key:expr) => {
                if let Ok(val) = env::var($key) {
                    if !val.is_empty() {
                        self.$field = Some(val);
                    }
                }
            };
        }
        macro_rules! env_parse {
            ($field:ident, $key:expr) => {
                if let Ok(val) = env::var($key) {
                    if let Ok(parsed) = val.parse() {
                        self.$field = parsed;
                    }
                }
            };
        }

        // Flask
        env_str!(host, "HOST");
        env_parse!(port, "PORT");

        // DB
        env_str!(db_dialect, "DB_DIALECT");
        env_str!(db_host, "DB_HOST");
        env_parse!(db_port, "DB_PORT");
        env_str!(db_user, "DB_USER");
        env_str!(db_password, "DB_PASSWORD");
        env_str!(db_name, "DB_NAME");
        env_str!(db_charset, "DB_CHARSET");

        // InsightEngine
        env_str!(insight_engine_api_key, "INSIGHT_ENGINE_API_KEY");
        env_opt_str!(insight_engine_base_url, "INSIGHT_ENGINE_BASE_URL");
        env_str!(insight_engine_model_name, "INSIGHT_ENGINE_MODEL_NAME");

        // MediaEngine
        env_str!(media_engine_api_key, "MEDIA_ENGINE_API_KEY");
        env_opt_str!(media_engine_base_url, "MEDIA_ENGINE_BASE_URL");
        env_str!(media_engine_model_name, "MEDIA_ENGINE_MODEL_NAME");

        // QueryEngine
        env_str!(query_engine_api_key, "QUERY_ENGINE_API_KEY");
        env_opt_str!(query_engine_base_url, "QUERY_ENGINE_BASE_URL");
        env_str!(query_engine_model_name, "QUERY_ENGINE_MODEL_NAME");

        // ReportEngine
        env_str!(report_engine_api_key, "REPORT_ENGINE_API_KEY");
        env_opt_str!(report_engine_base_url, "REPORT_ENGINE_BASE_URL");
        env_str!(report_engine_model_name, "REPORT_ENGINE_MODEL_NAME");

        // MindSpider
        env_str!(mindspider_api_key, "MINDSPIDER_API_KEY");
        env_opt_str!(mindspider_base_url, "MINDSPIDER_BASE_URL");
        env_str!(mindspider_model_name, "MINDSPIDER_MODEL_NAME");

        // ForumHost
        env_str!(forum_host_api_key, "FORUM_HOST_API_KEY");
        env_opt_str!(forum_host_base_url, "FORUM_HOST_BASE_URL");
        env_str!(forum_host_model_name, "FORUM_HOST_MODEL_NAME");

        // KeywordOptimizer
        env_str!(keyword_optimizer_api_key, "KEYWORD_OPTIMIZER_API_KEY");
        env_opt_str!(keyword_optimizer_base_url, "KEYWORD_OPTIMIZER_BASE_URL");
        env_str!(keyword_optimizer_model_name, "KEYWORD_OPTIMIZER_MODEL_NAME");

        // Search
        env_str!(tavily_api_key, "TAVILY_API_KEY");
        env_str!(search_tool_type, "SEARCH_TOOL_TYPE");
        env_str!(bocha_base_url, "BOCHA_BASE_URL");
        env_str!(bocha_web_search_api_key, "BOCHA_WEB_SEARCH_API_KEY");
        env_str!(anspire_base_url, "ANSPIRE_BASE_URL");
        env_str!(anspire_api_key, "ANSPIRE_API_KEY");

        // Search params
        env_parse!(default_search_hot_content_limit, "DEFAULT_SEARCH_HOT_CONTENT_LIMIT");
        env_parse!(default_search_topic_globally_limit_per_table, "DEFAULT_SEARCH_TOPIC_GLOBALLY_LIMIT_PER_TABLE");
        env_parse!(default_search_topic_by_date_limit_per_table, "DEFAULT_SEARCH_TOPIC_BY_DATE_LIMIT_PER_TABLE");
        env_parse!(default_get_comments_for_topic_limit, "DEFAULT_GET_COMMENTS_FOR_TOPIC_LIMIT");
        env_parse!(default_search_topic_on_platform_limit, "DEFAULT_SEARCH_TOPIC_ON_PLATFORM_LIMIT");
        env_parse!(max_search_results_for_llm, "MAX_SEARCH_RESULTS_FOR_LLM");
        env_parse!(max_high_confidence_sentiment_results, "MAX_HIGH_CONFIDENCE_SENTIMENT_RESULTS");
        env_parse!(max_reflections, "MAX_REFLECTIONS");
        env_parse!(max_paragraphs, "MAX_PARAGRAPHS");
        env_parse!(search_timeout, "SEARCH_TIMEOUT");
        env_parse!(max_content_length, "MAX_CONTENT_LENGTH");
        env_parse!(max_search_results, "MAX_SEARCH_RESULTS");
        env_parse!(search_content_max_length, "SEARCH_CONTENT_MAX_LENGTH");

        // Output
        env_str!(output_dir, "OUTPUT_DIR");
        env_parse!(save_intermediate_states, "SAVE_INTERMEDIATE_STATES");
    }

    /// 設定情報を表示用文字列に変換
    pub fn display_query_engine_config(&self) -> String {
        format!(
            "=== Query Engine 設定 ===\n\
             LLM モデル: {}\n\
             LLM Base URL: {}\n\
             Tavily API Key: {}\n\
             検索タイムアウト: {} 秒\n\
             最大コンテンツ長: {}\n\
             最大リフレクション数: {}\n\
             最大段落数: {}\n\
             最大検索結果数: {}\n\
             出力ディレクトリ: {}\n\
             中間状態保存: {}\n\
             ========================",
            self.query_engine_model_name,
            self.query_engine_base_url.as_deref().unwrap_or("(デフォルト)"),
            if self.tavily_api_key.is_empty() { "未設定" } else { "設定済み" },
            self.search_timeout,
            self.search_content_max_length,
            self.max_reflections,
            self.max_paragraphs,
            self.max_search_results,
            self.output_dir,
            self.save_intermediate_states,
        )
    }
}
