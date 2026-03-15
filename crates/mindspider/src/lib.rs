//! BettaFish MindSpider - AI クローラー集群
//!
//! Python の MindSpider パッケージの Rust 実装。
//! データ収集、トピック抽出、プラットフォームクローラー管理を含む。
//!
//! ## モジュール構成
//! - **models**: データベース ORM モデル (SQLAlchemy → Rust structs)
//! - **database**: DatabaseManager (DB 接続・統計・クリーンアップ)
//! - **topic_extractor**: TopicExtractor (DeepSeek API ベースのトピック抽出)
//! - **keyword_manager**: KeywordManager (キーワード管理・配信)
//! - **platform_crawler**: PlatformCrawler (MediaCrawler 統合)
//! - **deep_sentiment**: DeepSentimentCrawling (深度センチメント分析ワークフロー)
//! - **mindspider**: MindSpider メインオーケストレーター

#![allow(dead_code)]

use async_trait::async_trait;
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{error, info, warn};

use bettafish_config::Settings;

// ============================================================================
// Spider トレイト (公開 API)
// ============================================================================

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
    async fn extract_broad_topics(&self) -> anyhow::Result<Vec<DailyTopic>>;

    /// クローリングタスクの状態を取得
    async fn get_task_status(&self, task_id: &str) -> anyhow::Result<serde_json::Value>;
}

// ============================================================================
// サポートされるプラットフォーム
// ============================================================================

/// サポートされるプラットフォーム
pub const SUPPORTED_PLATFORMS: &[&str] = &[
    "xhs",   // 小紅書
    "dy",    // 抖音
    "ks",    // 快手
    "bili",  // ビリビリ
    "wb",    // 微博
    "tieba", // 百度貼吧
    "zhihu", // 知乎
];

/// プラットフォームエイリアスマッピング
pub fn resolve_platform_alias(alias: &str) -> Option<&'static str> {
    match alias.to_lowercase().as_str() {
        "weibo" | "webo" | "微博" => Some("wb"),
        "douyin" | "抖音" => Some("dy"),
        "kuaishou" | "快手" => Some("ks"),
        "bilibili" | "b站" | "bstation" => Some("bili"),
        "xiaohongshu" | "小红书" | "redbook" => Some("xhs"),
        "zhihu" | "知乎" => Some("zhihu"),
        "tieba" | "贴吧" => Some("tieba"),
        _ => None,
    }
}

// ============================================================================
// データベースモデル: MindSpider コアテーブル (models_sa.py)
// ============================================================================

/// 日次ニュースデータ (daily_news テーブル)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyNews {
    pub id: Option<i64>,
    pub news_id: String,
    pub source_platform: String,
    pub title: String,
    pub url: Option<String>,
    pub description: Option<String>,
    pub extra_info: Option<String>,
    pub crawl_date: NaiveDate,
    pub rank_position: Option<i32>,
    pub add_ts: i64,
    pub last_modify_ts: i64,
}

/// 日次トピックデータ (daily_topics テーブル)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyTopic {
    pub id: Option<i64>,
    pub topic_id: String,
    pub topic_name: String,
    pub topic_description: Option<String>,
    /// キーワードリスト (DB では JSON テキストとして保存)
    pub keywords: Vec<String>,
    pub extract_date: NaiveDate,
    pub relevance_score: Option<f64>,
    pub news_count: Option<i32>,
    pub processing_status: Option<String>,
    pub add_ts: i64,
    pub last_modify_ts: i64,
}

impl Default for DailyTopic {
    fn default() -> Self {
        Self {
            id: None,
            topic_id: String::new(),
            topic_name: String::new(),
            topic_description: None,
            keywords: Vec::new(),
            extract_date: chrono::Local::now().date_naive(),
            relevance_score: None,
            news_count: Some(0),
            processing_status: Some("pending".to_string()),
            add_ts: 0,
            last_modify_ts: 0,
        }
    }
}

/// トピック・ニュース関連テーブル (topic_news_relation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicNewsRelation {
    pub id: Option<i64>,
    pub topic_id: String,
    pub news_id: String,
    pub relation_score: Option<f64>,
    pub extract_date: NaiveDate,
    pub add_ts: i64,
}

/// クローリングタスク (crawling_tasks テーブル)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrawlingTask {
    pub id: Option<i64>,
    pub task_id: String,
    pub topic_id: String,
    pub platform: String,
    pub search_keywords: String,
    pub task_status: Option<String>,
    pub start_time: Option<i64>,
    pub end_time: Option<i64>,
    pub total_crawled: Option<i32>,
    pub success_count: Option<i32>,
    pub error_count: Option<i32>,
    pub error_message: Option<String>,
    pub config_params: Option<String>,
    pub scheduled_date: NaiveDate,
    pub add_ts: i64,
    pub last_modify_ts: i64,
}

impl Default for CrawlingTask {
    fn default() -> Self {
        Self {
            id: None,
            task_id: String::new(),
            topic_id: String::new(),
            platform: String::new(),
            search_keywords: String::new(),
            task_status: Some("pending".to_string()),
            start_time: None,
            end_time: None,
            total_crawled: Some(0),
            success_count: Some(0),
            error_count: Some(0),
            error_message: None,
            config_params: None,
            scheduled_date: chrono::Local::now().date_naive(),
            add_ts: 0,
            last_modify_ts: 0,
        }
    }
}

// ============================================================================
// データベースモデル: プラットフォーム固有テーブル (models_bigdata.py)
// ============================================================================

/// Bilibili 動画データ (bilibili_video テーブル)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BilibiliVideo {
    pub id: Option<i64>,
    pub video_id: i64,
    pub video_url: String,
    pub user_id: Option<i64>,
    pub nickname: Option<String>,
    pub avatar: Option<String>,
    pub liked_count: Option<i32>,
    pub add_ts: Option<i64>,
    pub last_modify_ts: Option<i64>,
    pub video_type: Option<String>,
    pub title: Option<String>,
    pub desc: Option<String>,
    pub create_time: Option<i64>,
    pub disliked_count: Option<String>,
    pub video_play_count: Option<String>,
    pub video_favorite_count: Option<String>,
    pub video_share_count: Option<String>,
    pub video_coin_count: Option<String>,
    pub video_danmaku: Option<String>,
    pub video_comment: Option<String>,
    pub video_cover_url: Option<String>,
    pub source_keyword: Option<String>,
    pub topic_id: Option<String>,
    pub crawling_task_id: Option<String>,
}

/// Bilibili 動画コメント (bilibili_video_comment テーブル)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BilibiliVideoComment {
    pub id: Option<i64>,
    pub user_id: Option<String>,
    pub nickname: Option<String>,
    pub sex: Option<String>,
    pub sign: Option<String>,
    pub avatar: Option<String>,
    pub add_ts: Option<i64>,
    pub last_modify_ts: Option<i64>,
    pub comment_id: Option<i64>,
    pub video_id: Option<i64>,
    pub content: Option<String>,
    pub create_time: Option<i64>,
    pub sub_comment_count: Option<String>,
    pub parent_comment_id: Option<String>,
    pub like_count: Option<String>,
}

/// 抖音 Aweme データ (douyin_aweme テーブル)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DouyinAweme {
    pub id: Option<i64>,
    pub user_id: Option<String>,
    pub sec_uid: Option<String>,
    pub short_user_id: Option<String>,
    pub user_unique_id: Option<String>,
    pub nickname: Option<String>,
    pub avatar: Option<String>,
    pub user_signature: Option<String>,
    pub ip_location: Option<String>,
    pub add_ts: Option<i64>,
    pub last_modify_ts: Option<i64>,
    pub aweme_id: Option<String>,
    pub aweme_type: Option<String>,
    pub title: Option<String>,
    pub desc: Option<String>,
    pub create_time: Option<i64>,
    pub liked_count: Option<String>,
    pub comment_count: Option<String>,
    pub share_count: Option<String>,
    pub collected_count: Option<String>,
    pub aweme_url: Option<String>,
    pub cover_url: Option<String>,
    pub video_download_url: Option<String>,
    pub music_download_url: Option<String>,
    pub note_download_url: Option<String>,
    pub source_keyword: Option<String>,
    pub topic_id: Option<String>,
    pub crawling_task_id: Option<String>,
}

/// 抖音コメント (douyin_aweme_comment テーブル)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DouyinAwemeComment {
    pub id: Option<i64>,
    pub user_id: Option<String>,
    pub sec_uid: Option<String>,
    pub short_user_id: Option<String>,
    pub user_unique_id: Option<String>,
    pub nickname: Option<String>,
    pub avatar: Option<String>,
    pub user_signature: Option<String>,
    pub ip_location: Option<String>,
    pub add_ts: Option<i64>,
    pub last_modify_ts: Option<i64>,
    pub comment_id: Option<String>,
    pub aweme_id: Option<String>,
    pub content: Option<String>,
    pub create_time: Option<i64>,
    pub sub_comment_count: Option<String>,
    pub parent_comment_id: Option<String>,
    pub like_count: Option<String>,
    pub pictures: Option<String>,
}

/// 微博ノートデータ (weibo_note テーブル)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeiboNote {
    pub id: Option<i64>,
    pub user_id: Option<String>,
    pub nickname: Option<String>,
    pub avatar: Option<String>,
    pub gender: Option<String>,
    pub profile_url: Option<String>,
    pub ip_location: Option<String>,
    pub add_ts: Option<i64>,
    pub last_modify_ts: Option<i64>,
    pub note_id: Option<i64>,
    pub content: Option<String>,
    pub create_time: Option<i64>,
    pub create_date_time: Option<String>,
    pub liked_count: Option<String>,
    pub comments_count: Option<String>,
    pub shared_count: Option<String>,
    pub note_url: Option<String>,
    pub source_keyword: Option<String>,
    pub topic_id: Option<String>,
    pub crawling_task_id: Option<String>,
}

/// 微博コメント (weibo_note_comment テーブル)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeiboNoteComment {
    pub id: Option<i64>,
    pub user_id: Option<String>,
    pub nickname: Option<String>,
    pub avatar: Option<String>,
    pub gender: Option<String>,
    pub profile_url: Option<String>,
    pub ip_location: Option<String>,
    pub add_ts: Option<i64>,
    pub last_modify_ts: Option<i64>,
    pub comment_id: Option<i64>,
    pub note_id: Option<i64>,
    pub content: Option<String>,
    pub create_time: Option<i64>,
    pub create_date_time: Option<String>,
    pub comment_like_count: Option<String>,
    pub sub_comment_count: Option<String>,
    pub parent_comment_id: Option<String>,
}

/// 小紅書ノートデータ (xhs_note テーブル)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XhsNote {
    pub id: Option<i64>,
    pub user_id: Option<String>,
    pub nickname: Option<String>,
    pub avatar: Option<String>,
    pub ip_location: Option<String>,
    pub add_ts: Option<i64>,
    pub last_modify_ts: Option<i64>,
    pub note_id: Option<String>,
    pub r#type: Option<String>,
    pub title: Option<String>,
    pub desc: Option<String>,
    pub video_url: Option<String>,
    pub time: Option<i64>,
    pub last_update_time: Option<i64>,
    pub liked_count: Option<String>,
    pub collected_count: Option<String>,
    pub comment_count: Option<String>,
    pub share_count: Option<String>,
    pub image_list: Option<String>,
    pub tag_list: Option<String>,
    pub note_url: Option<String>,
    pub source_keyword: Option<String>,
    pub xsec_token: Option<String>,
    pub topic_id: Option<String>,
    pub crawling_task_id: Option<String>,
}

/// 小紅書コメント (xhs_note_comment テーブル)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XhsNoteComment {
    pub id: Option<i64>,
    pub user_id: Option<String>,
    pub nickname: Option<String>,
    pub avatar: Option<String>,
    pub ip_location: Option<String>,
    pub add_ts: Option<i64>,
    pub last_modify_ts: Option<i64>,
    pub comment_id: Option<String>,
    pub create_time: Option<i64>,
    pub note_id: Option<String>,
    pub content: Option<String>,
    pub sub_comment_count: Option<i32>,
    pub pictures: Option<String>,
    pub parent_comment_id: Option<String>,
    pub like_count: Option<String>,
}

/// 快手動画データ (kuaishou_video テーブル)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KuaishouVideo {
    pub id: Option<i64>,
    pub user_id: Option<String>,
    pub nickname: Option<String>,
    pub avatar: Option<String>,
    pub add_ts: Option<i64>,
    pub last_modify_ts: Option<i64>,
    pub video_id: Option<String>,
    pub video_type: Option<String>,
    pub title: Option<String>,
    pub desc: Option<String>,
    pub create_time: Option<i64>,
    pub liked_count: Option<String>,
    pub viewd_count: Option<String>,
    pub video_url: Option<String>,
    pub video_cover_url: Option<String>,
    pub video_play_url: Option<String>,
    pub source_keyword: Option<String>,
    pub topic_id: Option<String>,
    pub crawling_task_id: Option<String>,
}

/// 快手コメント (kuaishou_video_comment テーブル)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KuaishouVideoComment {
    pub id: Option<i64>,
    pub user_id: Option<String>,
    pub nickname: Option<String>,
    pub avatar: Option<String>,
    pub add_ts: Option<i64>,
    pub last_modify_ts: Option<i64>,
    pub comment_id: Option<i64>,
    pub video_id: Option<String>,
    pub content: Option<String>,
    pub create_time: Option<i64>,
    pub sub_comment_count: Option<String>,
}

/// 百度貼吧ノートデータ (tieba_note テーブル)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TiebaNote {
    pub id: Option<i64>,
    pub note_id: Option<String>,
    pub title: Option<String>,
    pub desc: Option<String>,
    pub note_url: Option<String>,
    pub publish_time: Option<String>,
    pub user_link: Option<String>,
    pub user_nickname: Option<String>,
    pub user_avatar: Option<String>,
    pub tieba_id: Option<String>,
    pub tieba_name: Option<String>,
    pub tieba_link: Option<String>,
    pub total_replay_num: Option<i32>,
    pub total_replay_page: Option<i32>,
    pub ip_location: Option<String>,
    pub add_ts: Option<i64>,
    pub last_modify_ts: Option<i64>,
    pub source_keyword: Option<String>,
    pub topic_id: Option<String>,
    pub crawling_task_id: Option<String>,
}

/// 百度貼吧コメント (tieba_comment テーブル)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TiebaComment {
    pub id: Option<i64>,
    pub comment_id: Option<String>,
    pub parent_comment_id: Option<String>,
    pub content: Option<String>,
    pub user_link: Option<String>,
    pub user_nickname: Option<String>,
    pub user_avatar: Option<String>,
    pub tieba_id: Option<String>,
    pub tieba_name: Option<String>,
    pub tieba_link: Option<String>,
    pub publish_time: Option<String>,
    pub ip_location: Option<String>,
    pub sub_comment_count: Option<i32>,
    pub note_id: Option<String>,
    pub note_url: Option<String>,
    pub add_ts: Option<i64>,
    pub last_modify_ts: Option<i64>,
}

/// 知乎コンテンツデータ (zhihu_content テーブル)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZhihuContent {
    pub id: Option<i64>,
    pub content_id: Option<String>,
    pub content_type: Option<String>,
    pub content_text: Option<String>,
    pub content_url: Option<String>,
    pub question_id: Option<String>,
    pub title: Option<String>,
    pub desc: Option<String>,
    pub created_time: Option<String>,
    pub updated_time: Option<String>,
    pub voteup_count: Option<i32>,
    pub comment_count: Option<i32>,
    pub source_keyword: Option<String>,
    pub user_id: Option<String>,
    pub user_link: Option<String>,
    pub user_nickname: Option<String>,
    pub user_avatar: Option<String>,
    pub user_url_token: Option<String>,
    pub add_ts: Option<i64>,
    pub last_modify_ts: Option<i64>,
    pub topic_id: Option<String>,
    pub crawling_task_id: Option<String>,
}

/// 知乎コメント (zhihu_comment テーブル)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZhihuComment {
    pub id: Option<i64>,
    pub comment_id: Option<String>,
    pub parent_comment_id: Option<String>,
    pub content: Option<String>,
    pub publish_time: Option<String>,
    pub ip_location: Option<String>,
    pub sub_comment_count: Option<i32>,
    pub like_count: Option<i32>,
    pub dislike_count: Option<i32>,
    pub content_id: Option<String>,
    pub content_type: Option<String>,
    pub user_id: Option<String>,
    pub user_link: Option<String>,
    pub user_nickname: Option<String>,
    pub user_avatar: Option<String>,
    pub add_ts: Option<i64>,
    pub last_modify_ts: Option<i64>,
}

// ============================================================================
// データベースマネージャートレイト
// ============================================================================

/// データベース操作トレイト (SQLAlchemy Engine の抽象化)
#[async_trait]
pub trait DatabaseBackend: Send + Sync {
    /// データベースに接続
    async fn connect(&mut self) -> anyhow::Result<()>;

    /// データベース接続を閉じる
    async fn close(&mut self) -> anyhow::Result<()>;

    /// テーブル一覧を取得
    async fn get_table_names(&self) -> anyhow::Result<Vec<String>>;

    /// テーブルの行数を取得
    async fn get_table_row_count(&self, table_name: &str) -> anyhow::Result<i64>;

    /// SQL クエリを実行 (行数を返す)
    async fn execute_query(&self, sql: &str) -> anyhow::Result<Vec<HashMap<String, serde_json::Value>>>;
}

/// DatabaseManager: MindSpider データベース管理ツール
///
/// Python の `MindSpider/schema/db_manager.py` に対応。
pub struct DatabaseManager {
    db_dialect: String,
    db_host: String,
    db_port: u16,
    db_user: String,
    db_password: String,
    db_name: String,
    db_charset: String,
    backend: Option<Box<dyn DatabaseBackend>>,
}

impl DatabaseManager {
    /// Settings から DatabaseManager を作成
    pub fn from_settings(settings: &Settings) -> Self {
        Self {
            db_dialect: settings.db_dialect.clone(),
            db_host: settings.db_host.clone(),
            db_port: settings.db_port,
            db_user: settings.db_user.clone(),
            db_password: settings.db_password.clone(),
            db_name: settings.db_name.clone(),
            db_charset: settings.db_charset.clone(),
            backend: None,
        }
    }

    /// MindSpiderConfig から DatabaseManager を作成
    pub fn from_config(config: &MindSpiderConfig) -> Self {
        Self {
            db_dialect: config.db_dialect.clone(),
            db_host: config.db_host.clone(),
            db_port: config.db_port,
            db_user: config.db_user.clone(),
            db_password: config.db_password.clone(),
            db_name: config.db_name.clone(),
            db_charset: "utf8mb4".to_string(),
            backend: None,
        }
    }

    /// カスタムバックエンドを設定
    pub fn set_backend(&mut self, backend: Box<dyn DatabaseBackend>) {
        self.backend = Some(backend);
    }

    /// データベース接続 URL を構築
    pub fn build_connection_url(&self) -> String {
        let dialect = self.db_dialect.to_lowercase();
        if dialect == "postgresql" || dialect == "postgres" {
            format!(
                "postgresql://{}:{}@{}:{}/{}",
                self.db_user, self.db_password, self.db_host, self.db_port, self.db_name
            )
        } else {
            format!(
                "mysql://{}:{}@{}:{}/{}?charset={}",
                self.db_user, self.db_password, self.db_host, self.db_port,
                self.db_name, self.db_charset
            )
        }
    }

    /// データベースに接続
    pub async fn connect(&mut self) -> anyhow::Result<()> {
        info!("データベースに接続中: {}", self.db_name);
        if let Some(backend) = &mut self.backend {
            backend.connect().await?;
            info!("データベース接続成功: {}", self.db_name);
        } else {
            warn!("データベースバックエンドが設定されていません (スタブモード)");
        }
        Ok(())
    }

    /// データベース接続を閉じる
    pub async fn close(&mut self) -> anyhow::Result<()> {
        if let Some(backend) = &mut self.backend {
            backend.close().await?;
        }
        info!("データベース接続を閉じました");
        Ok(())
    }

    /// 全テーブルを表示
    pub async fn show_tables(&self) -> anyhow::Result<String> {
        let mut output = String::new();
        output.push_str(&format!("\n{}\n", "=".repeat(60)));
        output.push_str("データベーステーブル一覧\n");
        output.push_str(&format!("{}\n", "=".repeat(60)));

        if let Some(backend) = &self.backend {
            let tables = backend.get_table_names().await?;
            if tables.is_empty() {
                output.push_str("データベースにテーブルがありません\n");
                return Ok(output);
            }

            let mindspider_tables: Vec<&String> = tables
                .iter()
                .filter(|t| {
                    matches!(
                        t.as_str(),
                        "daily_news" | "daily_topics" | "topic_news_relation" | "crawling_tasks"
                    )
                })
                .collect();
            let mediacrawler_tables: Vec<&String> = tables
                .iter()
                .filter(|t| {
                    !matches!(
                        t.as_str(),
                        "daily_news" | "daily_topics" | "topic_news_relation" | "crawling_tasks"
                    )
                })
                .collect();

            output.push_str("MindSpider コアテーブル:\n");
            for table in &mindspider_tables {
                let count = backend.get_table_row_count(table).await.unwrap_or(0);
                output.push_str(&format!("  - {:<25} ({:>6} レコード)\n", table, count));
            }

            output.push_str("\nMediaCrawler プラットフォームテーブル:\n");
            for table in &mediacrawler_tables {
                match backend.get_table_row_count(table).await {
                    Ok(count) => {
                        output.push_str(&format!("  - {:<25} ({:>6} レコード)\n", table, count));
                    }
                    Err(_) => {
                        output.push_str(&format!("  - {:<25} (クエリ失敗)\n", table));
                    }
                }
            }
        } else {
            output.push_str("  (データベースバックエンド未接続)\n");
        }

        info!("{}", output);
        Ok(output)
    }

    /// データ統計を表示
    pub async fn show_statistics(&self) -> anyhow::Result<String> {
        let mut output = String::new();
        output.push_str(&format!("\n{}\n", "=".repeat(60)));
        output.push_str("データ統計\n");
        output.push_str(&format!("{}\n", "=".repeat(60)));

        if self.backend.is_none() {
            output.push_str("  (データベースバックエンド未接続)\n");
            return Ok(output);
        }

        // プラットフォーム別テーブルマッピング
        let platform_tables = [
            ("xhs_note", "小紅書"),
            ("douyin_aweme", "抖音"),
            ("kuaishou_video", "快手"),
            ("bilibili_video", "B站"),
            ("weibo_note", "微博"),
            ("tieba_note", "貼吧"),
            ("zhihu_content", "知乎"),
        ];

        output.push_str("プラットフォーム別コンテンツ統計:\n");
        if let Some(backend) = &self.backend {
            for (table, platform_name) in &platform_tables {
                match backend.get_table_row_count(table).await {
                    Ok(count) => {
                        output.push_str(&format!("  - {}: {}\n", platform_name, count));
                    }
                    Err(_) => {
                        output.push_str(&format!("  - {}: テーブル不存在\n", platform_name));
                    }
                }
            }
        }

        info!("{}", output);
        Ok(output)
    }

    /// 最近のデータを表示
    pub async fn show_recent_data(&self, _days: i32) -> anyhow::Result<String> {
        let mut output = String::new();
        output.push_str(&format!("\n{}\n", "=".repeat(60)));
        output.push_str(&format!("最近 {} 日間のデータ\n", _days));
        output.push_str(&format!("{}\n", "=".repeat(60)));

        if self.backend.is_none() {
            output.push_str("  (データベースバックエンド未接続)\n");
        }

        info!("{}", output);
        Ok(output)
    }

    /// 古いデータをクリーンアップ
    pub async fn cleanup_old_data(&self, days: i32, dry_run: bool) -> anyhow::Result<String> {
        let mode = if dry_run { "プレビューモード" } else { "実行モード" };
        let mut output = String::new();
        output.push_str(&format!("\n{}\n", "=".repeat(60)));
        output.push_str(&format!("{}日前のデータをクリーンアップ ({})\n", days, mode));
        output.push_str(&format!("{}\n", "=".repeat(60)));

        if self.backend.is_none() {
            output.push_str("  (データベースバックエンド未接続)\n");
        }

        if dry_run {
            output.push_str("\nプレビューモードのため実際の削除は行いません。\n");
        }

        info!("{}", output);
        Ok(output)
    }
}

// ============================================================================
// TopicExtractor: トピック抽出器
// ============================================================================

/// トピック抽出結果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicExtractionResult {
    pub keywords: Vec<String>,
    pub summary: String,
}

/// TopicExtractor: DeepSeek API ベースのトピック抽出器
///
/// Python の `BroadTopicExtraction/topic_extractor.py` に対応。
pub struct TopicExtractor {
    llm_client: bettafish_llm::LLMClient,
}

impl TopicExtractor {
    /// Settings から TopicExtractor を作成
    pub fn new(settings: &Settings) -> anyhow::Result<Self> {
        let client = bettafish_llm::LLMClient::new(
            &settings.mindspider_api_key,
            &settings.mindspider_model_name,
            settings.mindspider_base_url.as_deref(),
        )?;
        Ok(Self { llm_client: client })
    }

    /// MindSpiderConfig から TopicExtractor を作成
    pub fn from_config(config: &MindSpiderConfig) -> anyhow::Result<Self> {
        let client = bettafish_llm::LLMClient::new(
            &config.api_key,
            &config.model_name,
            config.base_url.as_deref(),
        )?;
        Ok(Self { llm_client: client })
    }

    /// ニュースリストからキーワードとサマリーを抽出
    pub async fn extract_keywords_and_summary(
        &self,
        news_list: &[HashMap<String, String>],
        max_keywords: usize,
    ) -> anyhow::Result<TopicExtractionResult> {
        if news_list.is_empty() {
            return Ok(TopicExtractionResult {
                keywords: Vec::new(),
                summary: "本日はホットニュースがありません".to_string(),
            });
        }

        // ニュースサマリーテキストを構築
        let news_text = self.build_news_summary(news_list);

        // 分析プロンプトを構築
        let prompt = self.build_analysis_prompt(&news_text, max_keywords);

        let system_prompt =
            "你是一个专业的新闻分析师，擅长从热点新闻中提取关键词和撰写分析总结。";

        let options = bettafish_llm::InvokeOptions {
            temperature: Some(0.3),
            ..Default::default()
        };

        match self
            .llm_client
            .invoke(system_prompt, &prompt, &options)
            .await
        {
            Ok(result_text) => {
                let (keywords, summary) = self.parse_analysis_result(&result_text);
                let keywords = keywords.into_iter().take(max_keywords).collect();
                info!(
                    "トピック抽出成功: {} キーワード",
                    std::cmp::min(max_keywords, 0) // placeholder count
                );
                Ok(TopicExtractionResult { keywords, summary })
            }
            Err(e) => {
                error!("トピック抽出失敗: {}", e);
                let fallback_keywords = self.extract_simple_keywords(news_list);
                let fallback_summary = format!(
                    "本日は {} 件のホットニュースを収集しました。",
                    news_list.len()
                );
                Ok(TopicExtractionResult {
                    keywords: fallback_keywords.into_iter().take(max_keywords).collect(),
                    summary: fallback_summary,
                })
            }
        }
    }

    /// ニュースサマリーテキストを構築
    fn build_news_summary(&self, news_list: &[HashMap<String, String>]) -> String {
        let mut items = Vec::new();
        for (i, news) in news_list.iter().enumerate() {
            let title = news.get("title").map(|s| s.as_str()).unwrap_or("無題");
            let source = news
                .get("source_platform")
                .or_else(|| news.get("source"))
                .map(|s| s.as_str())
                .unwrap_or("不明");
            // タイトル内の特殊文字をクリーンアップ
            let clean_title = regex::Regex::new(r"[#@]")
                .unwrap()
                .replace_all(title, "")
                .trim()
                .to_string();
            items.push(format!("{}. 【{}】{}", i + 1, source, clean_title));
        }
        items.join("\n")
    }

    /// 分析プロンプトを構築
    fn build_analysis_prompt(&self, news_text: &str, max_keywords: usize) -> String {
        let news_count = news_text.lines().count();
        format!(
            r#"请分析以下{news_count}条今日热点新闻，完成两个任务：

新闻列表：
{news_text}

任务1：提取关键词（最多{max_keywords}个）
- 提取能代表今日热点话题的关键词
- 关键词应该适合用于社交媒体平台搜索
- 优先选择热度高、讨论量大的话题
- 避免过于宽泛或过于具体的词汇

任务2：撰写新闻分析总结（150-300字）
- 简要概括今日热点新闻的主要内容
- 指出当前社会关注的重点话题方向
- 分析这些热点反映的社会现象或趋势
- 语言简洁明了，客观中性

请严格按照以下JSON格式输出：
```json
{{
  "keywords": ["关键词1", "关键词2", "关键词3"],
  "summary": "今日新闻分析总结内容..."
}}
```

请直接输出JSON格式的结果，不要包含其他文字说明。"#
        )
    }

    /// 分析結果をパース
    fn parse_analysis_result(&self, result_text: &str) -> (Vec<String>, String) {
        // JSON コードブロックを抽出
        let json_re = regex::Regex::new(r"```json\s*([\s\S]*?)\s*```").unwrap();
        let json_text = if let Some(captures) = json_re.captures(result_text) {
            captures.get(1).unwrap().as_str().to_string()
        } else {
            result_text.trim().to_string()
        };

        match serde_json::from_str::<serde_json::Value>(&json_text) {
            Ok(data) => {
                let keywords: Vec<String> = data
                    .get("keywords")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str())
                            .map(|s| s.trim().to_string())
                            .filter(|s| s.len() > 1)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();

                // 重複除去
                let mut unique_keywords = Vec::new();
                for kw in keywords {
                    if !unique_keywords.contains(&kw) {
                        unique_keywords.push(kw);
                    }
                }

                let summary = data
                    .get("summary")
                    .and_then(|v| v.as_str())
                    .map(|s| s.trim().to_string())
                    .filter(|s| s.len() >= 10)
                    .unwrap_or_else(|| {
                        "今日のホットニュースは複数の分野にわたり、社会の多様な関心を反映しています。"
                            .to_string()
                    });

                (unique_keywords, summary)
            }
            Err(e) => {
                error!("JSON パース失敗: {}", e);
                (
                    Vec::new(),
                    "分析結果の処理に失敗しました。".to_string(),
                )
            }
        }
    }

    /// シンプルなキーワード抽出 (フォールバック)
    fn extract_simple_keywords(
        &self,
        news_list: &[HashMap<String, String>],
    ) -> Vec<String> {
        let stop_words = [
            "的", "了", "在", "和", "与", "或", "但", "是", "有", "被", "将", "已", "正在",
        ];
        let mut keywords = Vec::new();
        let clean_re = regex::Regex::new(r"[#@【】\[\]()（）]").unwrap();

        for news in news_list {
            if let Some(title) = news.get("title") {
                let clean = clean_re.replace_all(title, " ").to_string();
                for word in clean.split_whitespace() {
                    let word = word.trim();
                    if word.len() > 1
                        && !stop_words.contains(&word)
                        && !keywords.contains(&word.to_string())
                    {
                        keywords.push(word.to_string());
                    }
                }
            }
        }

        keywords.into_iter().take(10).collect()
    }

    /// 検索用キーワードを取得
    pub fn get_search_keywords(&self, keywords: &[String], limit: usize) -> Vec<String> {
        let digit_re = regex::Regex::new(r"^\d+$").unwrap();
        let alpha_re = regex::Regex::new(r"^[a-zA-Z]+$").unwrap();

        let mut search_keywords = Vec::new();
        for keyword in keywords {
            let kw = keyword.trim();
            if kw.len() > 1
                && kw.len() < 20
                && !search_keywords.contains(&kw.to_string())
                && !digit_re.is_match(kw)
                && !alpha_re.is_match(kw)
            {
                search_keywords.push(kw.to_string());
            }
        }

        search_keywords.into_iter().take(limit).collect()
    }
}

// ============================================================================
// KeywordManager: キーワード管理器
// ============================================================================

/// KeywordManager: キーワード管理器
///
/// Python の `DeepSentimentCrawling/keyword_manager.py` に対応。
pub struct KeywordManager {
    db_manager: DatabaseManager,
}

impl KeywordManager {
    /// Settings から KeywordManager を作成
    pub fn new(settings: &Settings) -> Self {
        Self {
            db_manager: DatabaseManager::from_settings(settings),
        }
    }

    /// MindSpiderConfig から KeywordManager を作成
    pub fn from_config(config: &MindSpiderConfig) -> Self {
        Self {
            db_manager: DatabaseManager::from_config(config),
        }
    }

    /// 最新のキーワードリストを取得
    ///
    /// 指定日のキーワードを取得し、なければ最近 7 日分を集約する。
    pub async fn get_latest_keywords(
        &self,
        _target_date: Option<NaiveDate>,
        max_keywords: usize,
    ) -> Vec<String> {
        let target_date = _target_date.unwrap_or_else(|| chrono::Local::now().date_naive());
        info!("{} のキーワードを取得中...", target_date);

        // 当日のトピックを取得
        if let Some(topics_data) = self.get_daily_topics(Some(target_date)).await {
            if !topics_data.keywords.is_empty() {
                let mut keywords = topics_data.keywords;
                info!(
                    "{} の {} キーワードを取得成功",
                    target_date,
                    keywords.len()
                );
                if keywords.len() > max_keywords {
                    keywords.truncate(max_keywords);
                    info!("{} キーワードに制限しました", max_keywords);
                }
                return keywords;
            }
        }

        // 最近のデータがない場合、デフォルトキーワードを返す
        info!("キーワードデータが見つかりません。デフォルトキーワードを使用します");
        self.get_default_keywords()
    }

    /// 日次トピック分析を取得
    pub async fn get_daily_topics(&self, _extract_date: Option<NaiveDate>) -> Option<DailyTopic> {
        // バックエンドが設定されていない場合は None を返す
        if self.db_manager.backend.is_none() {
            return None;
        }
        // 実際の DB クエリはバックエンド実装に依存
        None
    }

    /// 全プラットフォーム用のキーワードリストを取得
    pub async fn get_all_keywords_for_platforms(
        &self,
        platforms: &[String],
        target_date: Option<NaiveDate>,
        max_keywords: usize,
    ) -> Vec<String> {
        let keywords = self.get_latest_keywords(target_date, max_keywords).await;
        if !keywords.is_empty() {
            info!(
                "{} プラットフォーム用に {} キーワードを準備しました",
                platforms.len(),
                keywords.len()
            );
        }
        keywords
    }

    /// デフォルトキーワードリスト
    fn get_default_keywords(&self) -> Vec<String> {
        [
            "科技", "人工智能", "AI", "编程", "互联网",
            "创业", "投资", "理财", "股市", "经济",
            "教育", "学习", "考试", "大学", "就业",
            "健康", "养生", "运动", "美食", "旅游",
            "时尚", "美妆", "购物", "生活", "家居",
            "电影", "音乐", "游戏", "娱乐", "明星",
            "新闻", "热点", "社会", "政策", "环保",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    /// クローリングサマリーを取得
    pub async fn get_crawling_summary(
        &self,
        target_date: Option<NaiveDate>,
    ) -> HashMap<String, serde_json::Value> {
        let target = target_date.unwrap_or_else(|| chrono::Local::now().date_naive());
        let topics_data = self.get_daily_topics(Some(target)).await;

        let mut summary = HashMap::new();
        summary.insert(
            "date".to_string(),
            serde_json::Value::String(target.to_string()),
        );

        if let Some(data) = topics_data {
            summary.insert(
                "keywords_count".to_string(),
                serde_json::json!(data.keywords.len()),
            );
            summary.insert("has_data".to_string(), serde_json::json!(true));
        } else {
            summary.insert("keywords_count".to_string(), serde_json::json!(0));
            summary.insert("has_data".to_string(), serde_json::json!(false));
            summary.insert(
                "summary".to_string(),
                serde_json::json!("データなし"),
            );
        }
        summary
    }
}

// ============================================================================
// PlatformCrawler: プラットフォームクローラー管理器
// ============================================================================

/// クロール結果統計
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CrawlStats {
    pub platform: String,
    pub keywords_count: usize,
    pub duration_seconds: f64,
    pub start_time: String,
    pub end_time: String,
    pub return_code: i32,
    pub success: bool,
    pub notes_count: usize,
    pub comments_count: usize,
    pub errors_count: usize,
}

/// マルチプラットフォームクロール結果
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MultiPlatformCrawlResult {
    pub total_keywords: usize,
    pub total_platforms: usize,
    pub total_tasks: usize,
    pub successful_tasks: usize,
    pub failed_tasks: usize,
    pub total_notes: usize,
    pub total_comments: usize,
    pub platform_summary: HashMap<String, PlatformCrawlSummary>,
}

/// プラットフォーム別クロールサマリー
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PlatformCrawlSummary {
    pub successful_keywords: usize,
    pub failed_keywords: usize,
    pub total_notes: usize,
    pub total_comments: usize,
}

/// PlatformCrawler: MediaCrawler 統合管理器
///
/// Python の `DeepSentimentCrawling/platform_crawler.py` に対応。
pub struct PlatformCrawler {
    supported_platforms: Vec<String>,
    crawl_stats: HashMap<String, CrawlStats>,
}

impl PlatformCrawler {
    /// 新しい PlatformCrawler を作成
    pub fn new() -> Self {
        Self {
            supported_platforms: SUPPORTED_PLATFORMS
                .iter()
                .map(|s| s.to_string())
                .collect(),
            crawl_stats: HashMap::new(),
        }
    }

    /// サポートされるプラットフォーム一覧
    pub fn supported_platforms(&self) -> &[String] {
        &self.supported_platforms
    }

    /// MediaCrawler のデータベース設定を構成
    ///
    /// 実際の実装では MediaCrawler の config/db_config.py を書き換える。
    pub fn configure_mediacrawler_db(&self, _settings: &Settings) -> anyhow::Result<bool> {
        info!("MediaCrawler データベースを設定中...");
        // スタブ: 実際の実装では Python の platform_crawler.py と同様に
        // MediaCrawler の db_config.py を動的に書き換える
        info!("MediaCrawler データベース設定完了 (スタブ)");
        Ok(true)
    }

    /// MediaCrawler のベース設定を作成
    pub fn create_base_config(
        &self,
        platform: &str,
        keywords: &[String],
        crawler_type: &str,
        max_notes: usize,
    ) -> anyhow::Result<bool> {
        if !self.supported_platforms.iter().any(|p| p == platform) {
            anyhow::bail!("サポートされていないプラットフォーム: {}", platform);
        }
        info!(
            "プラットフォーム {} 設定: タイプ={}, キーワード数={}, 最大ノート数={}",
            platform,
            crawler_type,
            keywords.len(),
            max_notes
        );
        // スタブ: 実際の実装では base_config.py を動的に書き換える
        Ok(true)
    }

    /// クローラーを実行
    pub async fn run_crawler(
        &mut self,
        platform: &str,
        keywords: &[String],
        _login_type: &str,
        _max_notes: usize,
    ) -> anyhow::Result<CrawlStats> {
        if !self.supported_platforms.iter().any(|p| p == platform) {
            anyhow::bail!("サポートされていないプラットフォーム: {}", platform);
        }
        if keywords.is_empty() {
            anyhow::bail!("キーワードリストが空です");
        }

        let start_time = chrono::Local::now();
        info!(
            "クローラー開始: platform={}, keywords={} ({}件)",
            platform,
            keywords.iter().take(5).cloned().collect::<Vec<_>>().join(", "),
            keywords.len()
        );

        // スタブ: 実際の実装では subprocess で MediaCrawler の main.py を起動する
        let end_time = chrono::Local::now();
        let duration = (end_time - start_time).num_milliseconds() as f64 / 1000.0;

        let stats = CrawlStats {
            platform: platform.to_string(),
            keywords_count: keywords.len(),
            duration_seconds: duration,
            start_time: start_time.format("%Y-%m-%dT%H:%M:%S").to_string(),
            end_time: end_time.format("%Y-%m-%dT%H:%M:%S").to_string(),
            return_code: 0,
            success: true,
            notes_count: 0,
            comments_count: 0,
            errors_count: 0,
        };

        self.crawl_stats.insert(platform.to_string(), stats.clone());
        info!("クローラー完了: {} (スタブ), 耗時: {:.1}s", platform, duration);
        Ok(stats)
    }

    /// キーワードベースのマルチプラットフォームクロール
    pub async fn run_multi_platform_crawl_by_keywords(
        &mut self,
        keywords: &[String],
        platforms: &[String],
        login_type: &str,
        max_notes_per_keyword: usize,
    ) -> anyhow::Result<MultiPlatformCrawlResult> {
        info!(
            "マルチプラットフォームクロール開始: {}キーワード x {}プラットフォーム = {}タスク",
            keywords.len(),
            platforms.len(),
            keywords.len() * platforms.len()
        );

        let mut result = MultiPlatformCrawlResult {
            total_keywords: keywords.len(),
            total_platforms: platforms.len(),
            total_tasks: keywords.len() * platforms.len(),
            ..Default::default()
        };

        for platform in platforms {
            let mut summary = PlatformCrawlSummary::default();

            match self
                .run_crawler(platform, keywords, login_type, max_notes_per_keyword)
                .await
            {
                Ok(stats) => {
                    if stats.success {
                        result.successful_tasks += keywords.len();
                        summary.successful_keywords = keywords.len();
                        summary.total_notes = stats.notes_count;
                        summary.total_comments = stats.comments_count;
                        result.total_notes += stats.notes_count;
                        result.total_comments += stats.comments_count;
                    } else {
                        result.failed_tasks += keywords.len();
                        summary.failed_keywords = keywords.len();
                    }
                }
                Err(e) => {
                    error!("{} プラットフォームクロール失敗: {}", platform, e);
                    result.failed_tasks += keywords.len();
                    summary.failed_keywords = keywords.len();
                }
            }

            result
                .platform_summary
                .insert(platform.clone(), summary);
        }

        info!(
            "マルチプラットフォームクロール完了: 成功={}, 失敗={}",
            result.successful_tasks, result.failed_tasks
        );
        Ok(result)
    }

    /// クロール統計を取得
    pub fn get_crawl_statistics(&self) -> serde_json::Value {
        serde_json::json!({
            "platforms_crawled": self.crawl_stats.keys().collect::<Vec<_>>(),
            "total_platforms": self.crawl_stats.len(),
            "detailed_stats": self.crawl_stats,
        })
    }
}

impl Default for PlatformCrawler {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// DeepSentimentCrawling: 深度センチメント分析ワークフロー
// ============================================================================

/// DeepSentimentCrawling: 深度センチメント分析ワークフロー
///
/// Python の `DeepSentimentCrawling/main.py` に対応。
pub struct DeepSentimentCrawling {
    keyword_manager: KeywordManager,
    platform_crawler: PlatformCrawler,
}

impl DeepSentimentCrawling {
    /// Settings から作成
    pub fn new(settings: &Settings) -> Self {
        Self {
            keyword_manager: KeywordManager::new(settings),
            platform_crawler: PlatformCrawler::new(),
        }
    }

    /// 日次クローリングを実行
    pub async fn run_daily_crawling(
        &mut self,
        target_date: Option<NaiveDate>,
        platforms: Option<Vec<String>>,
        max_keywords: usize,
        max_notes: usize,
        login_type: &str,
    ) -> anyhow::Result<serde_json::Value> {
        let target =
            target_date.unwrap_or_else(|| chrono::Local::now().date_naive());
        let platforms = platforms.unwrap_or_else(|| {
            SUPPORTED_PLATFORMS.iter().map(|s| s.to_string()).collect()
        });

        info!(
            "深度センチメントクローリング開始: 日付={}, プラットフォーム={:?}",
            target, platforms
        );

        // キーワードサマリーを取得
        let summary = self
            .keyword_manager
            .get_crawling_summary(Some(target))
            .await;

        let has_data = summary
            .get("has_data")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        if !has_data {
            warn!("トピックデータが見つかりません。先にブロードトピック抽出を実行してください。");
        }

        // キーワードを取得
        let keywords = self
            .keyword_manager
            .get_latest_keywords(Some(target), max_keywords)
            .await;

        if keywords.is_empty() {
            warn!("キーワードが見つかりません");
            return Ok(serde_json::json!({
                "success": false,
                "error": "キーワードなし"
            }));
        }

        info!(
            "取得キーワード: {} 件, プラットフォーム: {} 件",
            keywords.len(),
            platforms.len()
        );

        // マルチプラットフォームクロールを実行
        let crawl_results = self
            .platform_crawler
            .run_multi_platform_crawl_by_keywords(
                &keywords,
                &platforms,
                login_type,
                max_notes,
            )
            .await?;

        let final_report = serde_json::json!({
            "date": target.to_string(),
            "summary": summary,
            "crawl_results": crawl_results,
            "success": crawl_results.successful_tasks > 0,
        });

        info!("深度センチメントクローリング完了");
        Ok(final_report)
    }

    /// 単一プラットフォームのクローリングを実行
    pub async fn run_platform_crawling(
        &mut self,
        platform: &str,
        target_date: Option<NaiveDate>,
        max_keywords: usize,
        max_notes: usize,
        login_type: &str,
    ) -> anyhow::Result<CrawlStats> {
        if !SUPPORTED_PLATFORMS.contains(&platform) {
            anyhow::bail!("サポートされていないプラットフォーム: {}", platform);
        }

        let target =
            target_date.unwrap_or_else(|| chrono::Local::now().date_naive());

        info!(
            "プラットフォームクローリング開始: {} ({})",
            platform, target
        );

        let keywords = self
            .keyword_manager
            .get_latest_keywords(Some(target), max_keywords)
            .await;

        if keywords.is_empty() {
            anyhow::bail!("{} プラットフォームのキーワードが見つかりません", platform);
        }

        info!("{} キーワードでクロール開始", keywords.len());

        self.platform_crawler
            .run_crawler(platform, &keywords, login_type, max_notes)
            .await
    }
}

// ============================================================================
// MindSpider 設定
// ============================================================================

/// MindSpider 設定
#[derive(Debug, Clone, Deserialize)]
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

impl MindSpiderConfig {
    /// Settings から MindSpiderConfig を構築
    pub fn from_settings(settings: &Settings) -> Self {
        Self {
            api_key: settings.mindspider_api_key.clone(),
            base_url: settings.mindspider_base_url.clone(),
            model_name: settings.mindspider_model_name.clone(),
            db_dialect: settings.db_dialect.clone(),
            db_host: settings.db_host.clone(),
            db_port: settings.db_port,
            db_user: settings.db_user.clone(),
            db_password: settings.db_password.clone(),
            db_name: settings.db_name.clone(),
        }
    }
}

// ============================================================================
// MindSpider: メインオーケストレーター
// ============================================================================

/// MindSpider メインオーケストレーター
///
/// Python の `MindSpider/main.py::MindSpider` クラスに対応。
/// BroadTopicExtraction と DeepSentimentCrawling を統合管理する。
pub struct MindSpider {
    config: MindSpiderConfig,
    settings: Settings,
    db_manager: DatabaseManager,
}

impl MindSpider {
    /// Settings から MindSpider を作成
    pub fn new(settings: Settings) -> Self {
        let config = MindSpiderConfig::from_settings(&settings);
        let db_manager = DatabaseManager::from_settings(&settings);
        info!("MindSpider AI クローラープロジェクト");
        Self {
            config,
            settings,
            db_manager,
        }
    }

    /// 基本設定をチェック
    pub fn check_config(&self) -> bool {
        info!("基本設定をチェック中...");
        let required_fields = [
            ("DB_HOST", &self.settings.db_host),
            ("DB_USER", &self.settings.db_user),
            ("DB_PASSWORD", &self.settings.db_password),
            ("DB_NAME", &self.settings.db_name),
            ("MINDSPIDER_API_KEY", &self.settings.mindspider_api_key),
            ("MINDSPIDER_MODEL_NAME", &self.settings.mindspider_model_name),
        ];

        let mut missing = Vec::new();
        for (name, value) in &required_fields {
            if value.is_empty() {
                missing.push(*name);
            }
        }

        if !missing.is_empty() {
            error!("設定が不足しています: {}", missing.join(", "));
            error!(".env ファイルの環境変数設定を確認してください");
            return false;
        }

        info!("基本設定チェック完了");
        true
    }

    /// データベース接続をチェック
    pub async fn check_database_connection(&mut self) -> bool {
        info!("データベース接続をチェック中...");
        match self.db_manager.connect().await {
            Ok(()) => {
                info!("データベース接続正常");
                true
            }
            Err(e) => {
                error!("データベース接続失敗: {}", e);
                false
            }
        }
    }

    /// データベースを初期化
    pub async fn initialize_database(&self) -> bool {
        info!("データベースを初期化中...");
        // スタブ: 実際の実装では init_database.py 相当のスキーママイグレーションを行う
        info!("データベース初期化完了 (スタブ)");
        true
    }

    /// BroadTopicExtraction を実行
    pub async fn run_broad_topic_extraction(
        &self,
        _extract_date: Option<NaiveDate>,
        _keywords_count: usize,
    ) -> bool {
        info!("BroadTopicExtraction モジュール実行中...");

        // 設定チェック
        if !self.check_config() {
            return false;
        }

        info!("キーワード数: {}", _keywords_count);

        // スタブ: 実際の実装では BroadTopicExtraction/main.py を起動する
        info!("BroadTopicExtraction モジュール実行完了 (スタブ)");
        true
    }

    /// DeepSentimentCrawling を実行
    pub async fn run_deep_sentiment_crawling(
        &self,
        target_date: Option<NaiveDate>,
        platforms: Option<Vec<String>>,
        max_keywords: usize,
        max_notes: usize,
        _test_mode: bool,
    ) -> bool {
        info!("DeepSentimentCrawling モジュール実行中...");

        let mut crawler = DeepSentimentCrawling::new(&self.settings);

        match crawler
            .run_daily_crawling(
                target_date,
                platforms,
                max_keywords,
                max_notes,
                "qrcode",
            )
            .await
        {
            Ok(result) => {
                let success = result
                    .get("success")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                if success {
                    info!("DeepSentimentCrawling モジュール実行完了");
                } else {
                    error!(
                        "DeepSentimentCrawling モジュール実行失敗: {:?}",
                        result.get("error")
                    );
                }
                success
            }
            Err(e) => {
                error!("DeepSentimentCrawling モジュール実行エラー: {}", e);
                false
            }
        }
    }

    /// 完全ワークフローを実行
    pub async fn run_complete_workflow(
        &self,
        target_date: Option<NaiveDate>,
        platforms: Option<Vec<String>>,
        keywords_count: usize,
        max_keywords: usize,
        max_notes: usize,
        test_mode: bool,
    ) -> bool {
        let target =
            target_date.unwrap_or_else(|| chrono::Local::now().date_naive());

        info!("完全な MindSpider ワークフローを開始");
        info!("対象日: {}", target);
        info!(
            "プラットフォーム: {}",
            platforms
                .as_ref()
                .map(|p| format!("{:?}", p))
                .unwrap_or_else(|| "全サポートプラットフォーム".to_string())
        );
        info!("テストモード: {}", if test_mode { "はい" } else { "いいえ" });

        // ステップ 1: トピック抽出
        info!("=== ステップ 1: トピック抽出 ===");
        if !self
            .run_broad_topic_extraction(Some(target), keywords_count)
            .await
        {
            error!("トピック抽出に失敗しました。プロセスを中断します。");
            return false;
        }

        // ステップ 2: センチメントクローリング
        info!("=== ステップ 2: センチメントクローリング ===");
        if !self
            .run_deep_sentiment_crawling(
                Some(target),
                platforms,
                max_keywords,
                max_notes,
                test_mode,
            )
            .await
        {
            error!(
                "センチメントクローリングに失敗しました。トピック抽出は完了しています。"
            );
            return false;
        }

        info!("完全ワークフロー実行成功！");
        true
    }

    /// プロジェクトステータスを表示
    pub fn show_status(&self) {
        info!("MindSpider プロジェクトステータス:");
        let config_ok = self.check_config();
        info!("設定状態: {}", if config_ok { "正常" } else { "異常" });
    }
}

// ============================================================================
// Spider トレイト実装
// ============================================================================

#[async_trait]
impl Spider for MindSpider {
    async fn start_deep_sentiment_crawling(
        &mut self,
        keywords: &[String],
        platforms: &[String],
    ) -> anyhow::Result<()> {
        let mut crawler = DeepSentimentCrawling::new(&self.settings);
        crawler
            .run_daily_crawling(
                None,
                Some(platforms.to_vec()),
                keywords.len(),
                50,
                "qrcode",
            )
            .await?;
        Ok(())
    }

    async fn extract_broad_topics(&self) -> anyhow::Result<Vec<DailyTopic>> {
        // スタブ: 実際にはニュースを取得してトピック抽出を行う
        Ok(Vec::new())
    }

    async fn get_task_status(&self, _task_id: &str) -> anyhow::Result<serde_json::Value> {
        Ok(serde_json::json!({
            "task_id": _task_id,
            "status": "unknown",
            "message": "タスクステータス取得はまだ実装されていません"
        }))
    }
}
