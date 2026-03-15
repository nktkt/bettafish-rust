//! MediaCrawlerDB - ローカル世論データベースクエリツール集
//!
//! Python の tools/search.py の Rust 実装。
//! AI Agent が呼び出す5種類の DB 検索ツールを提供。
//!
//! ツール:
//! - search_hot_content: 指定期間内の総合熱度最高コンテンツ
//! - search_topic_globally: DB 全体でトピックをグローバル検索
//! - search_topic_by_date: 日付範囲でトピック検索
//! - get_comments_for_topic: トピックのコメントデータ取得
//! - search_topic_on_platform: 特定プラットフォームでトピック検索

use chrono::{DateTime, NaiveDate, NaiveDateTime};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{info, warn};

// ---------------------------------------------------------------------------
// データ構造定義
// ---------------------------------------------------------------------------

/// 統一 DB クエリ結果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    /// プラットフォーム名 (bilibili, weibo, douyin, kuaishou, xhs, zhihu, tieba)
    pub platform: String,
    /// コンテンツタイプ (video, note, comment, content, news)
    pub content_type: String,
    /// タイトルまたは本文
    pub title_or_content: String,
    /// 投稿者ニックネーム
    pub author_nickname: Option<String>,
    /// URL
    pub url: Option<String>,
    /// 公開日時
    pub publish_time: Option<String>,
    /// エンゲージメント指標 (likes, comments, shares, views, favorites, coins, danmaku)
    pub engagement: HashMap<String, i64>,
    /// ソースキーワード
    pub source_keyword: Option<String>,
    /// 熱度スコア
    pub hotness_score: f64,
    /// ソーステーブル名
    pub source_table: String,
}

/// DB ツールの完全レスポンス
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DBResponse {
    /// ツール名
    pub tool_name: String,
    /// パラメータ
    pub parameters: HashMap<String, serde_json::Value>,
    /// 結果リスト
    pub results: Vec<QueryResult>,
    /// 結果件数
    pub results_count: usize,
    /// エラーメッセージ
    pub error_message: Option<String>,
    /// メタデータ (感情分析結果等)
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Default for DBResponse {
    fn default() -> Self {
        Self {
            tool_name: String::new(),
            parameters: HashMap::new(),
            results: Vec::new(),
            results_count: 0,
            error_message: None,
            metadata: HashMap::new(),
        }
    }
}

impl DBResponse {
    /// 新しい DBResponse を作成
    pub fn new(tool_name: &str, parameters: HashMap<String, serde_json::Value>) -> Self {
        Self {
            tool_name: tool_name.to_string(),
            parameters,
            ..Default::default()
        }
    }

    /// エラーレスポンスを作成
    pub fn error(tool_name: &str, msg: &str) -> Self {
        Self {
            tool_name: tool_name.to_string(),
            error_message: Some(msg.to_string()),
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// プラットフォーム定義
// ---------------------------------------------------------------------------

/// サポートされているプラットフォーム
#[allow(dead_code)]
pub const SUPPORTED_PLATFORMS: &[&str] = &[
    "bilibili", "weibo", "douyin", "kuaishou", "xhs", "zhihu", "tieba",
];

/// 時間期間
#[allow(dead_code)]
pub const TIME_PERIODS: &[&str] = &["24h", "week", "year"];

// ---------------------------------------------------------------------------
// MediaCrawlerDB
// ---------------------------------------------------------------------------

/// ローカル世論 DB クエリツールクライアント
///
/// 内部で SQL を構築して DB に問い合わせる。
/// 本 Rust 実装では実際の DB 接続はスタブ化し、
/// 将来の DB ドライバ統合で置換する。
pub struct MediaCrawlerDB {
    /// DB ダイアレクト ("mysql" or "postgresql")
    #[allow(dead_code)]
    db_dialect: String,
}

/// 熱度計算の重み定数
impl MediaCrawlerDB {
    /// いいね重み
    pub const W_LIKE: f64 = 1.0;
    /// コメント重み
    pub const W_COMMENT: f64 = 5.0;
    /// シェア/転送/お気に入り/コイン等の高価値インタラクション重み
    pub const W_SHARE: f64 = 10.0;
    /// 閲覧数重み
    pub const W_VIEW: f64 = 0.1;
    /// 弾幕重み
    pub const W_DANMAKU: f64 = 0.5;

    /// 新しい MediaCrawlerDB クライアントを作成
    pub fn new() -> Self {
        Self {
            db_dialect: std::env::var("DB_DIALECT").unwrap_or_else(|_| "mysql".to_string()),
        }
    }

    /// DB ダイアレクトに応じたフィールド名ラッパー
    #[allow(dead_code)]
    fn wrap_field(&self, field: &str) -> String {
        if self.db_dialect == "postgresql" {
            format!("\"{}\"", field)
        } else {
            format!("`{}`", field)
        }
    }

    /// タイムスタンプを ISO 8601 文字列に変換するヘルパー
    #[allow(dead_code)]
    fn to_datetime_string(ts: &serde_json::Value) -> Option<String> {
        match ts {
            serde_json::Value::String(s) => {
                if s.is_empty() {
                    return None;
                }
                // ISO 8601 形式をそのまま返す
                if let Ok(_dt) = NaiveDateTime::parse_from_str(
                    s.split('+').next().unwrap_or(s).trim(),
                    "%Y-%m-%dT%H:%M:%S",
                ) {
                    return Some(s.clone());
                }
                if let Ok(_dt) =
                    NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S")
                {
                    return Some(s.clone());
                }
                Some(s.clone())
            }
            serde_json::Value::Number(n) => {
                if let Some(val) = n.as_f64() {
                    let ts_secs = if val > 1_000_000_000_000.0 {
                        val / 1000.0
                    } else {
                        val
                    };
                    if let Some(dt) = DateTime::from_timestamp(ts_secs as i64, 0) {
                        return Some(dt.format("%Y-%m-%dT%H:%M:%S").to_string());
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// エンゲージメント指標を抽出
    #[allow(dead_code)]
    fn extract_engagement(row: &HashMap<String, serde_json::Value>) -> HashMap<String, i64> {
        let mut engagement = HashMap::new();

        let mapping: &[(&str, &[&str])] = &[
            (
                "likes",
                &[
                    "liked_count",
                    "like_count",
                    "voteup_count",
                    "comment_like_count",
                ],
            ),
            (
                "comments",
                &[
                    "video_comment",
                    "comments_count",
                    "comment_count",
                    "total_replay_num",
                    "sub_comment_count",
                ],
            ),
            (
                "shares",
                &[
                    "video_share_count",
                    "shared_count",
                    "share_count",
                    "total_forwards",
                ],
            ),
            ("views", &["video_play_count", "viewd_count"]),
            ("favorites", &["video_favorite_count", "collected_count"]),
            ("coins", &["video_coin_count"]),
            ("danmaku", &["video_danmaku"]),
        ];

        for (key, potential_cols) in mapping {
            for col in *potential_cols {
                if let Some(val) = row.get(*col) {
                    let num = match val {
                        serde_json::Value::Number(n) => n.as_i64().unwrap_or(0),
                        serde_json::Value::String(s) => s.parse::<i64>().unwrap_or(0),
                        _ => 0,
                    };
                    if num != 0 {
                        engagement.insert(key.to_string(), num);
                        break;
                    }
                }
            }
        }

        engagement
    }

    /// 熱度スコアを計算
    #[allow(dead_code)]
    fn calculate_hotness(engagement: &HashMap<String, i64>) -> f64 {
        let likes = *engagement.get("likes").unwrap_or(&0) as f64;
        let comments = *engagement.get("comments").unwrap_or(&0) as f64;
        let shares = *engagement.get("shares").unwrap_or(&0) as f64;
        let views = *engagement.get("views").unwrap_or(&0) as f64;
        let favorites = *engagement.get("favorites").unwrap_or(&0) as f64;
        let coins = *engagement.get("coins").unwrap_or(&0) as f64;
        let danmaku = *engagement.get("danmaku").unwrap_or(&0) as f64;

        likes * Self::W_LIKE
            + comments * Self::W_COMMENT
            + (shares + favorites + coins) * Self::W_SHARE
            + views * Self::W_VIEW
            + danmaku * Self::W_DANMAKU
    }

    /// DB クエリを実行 (スタブ実装)
    ///
    /// 将来は実際の DB ドライバ (sqlx 等) を使用する。
    /// 現在はログ出力のみで空の結果を返す。
    #[allow(dead_code)]
    async fn execute_query(
        &self,
        query: &str,
        _params: &[&str],
    ) -> Vec<HashMap<String, serde_json::Value>> {
        info!("DB クエリ実行 (スタブ): {}", &query[..query.len().min(200)]);
        Vec::new()
    }

    // =========================================================================
    // 5つの検索ツール
    // =========================================================================

    /// 【ツール1】ホットコンテンツ検索
    ///
    /// 指定期間内の総合熱度最高コンテンツを取得。
    /// 各プラットフォームの点赞・コメント・シェア等から加重熱度を計算。
    ///
    /// # Arguments
    /// * `time_period` - 時間範囲 ("24h", "week", "year")
    /// * `limit` - 最大結果数
    pub async fn search_hot_content(
        &self,
        time_period: &str,
        limit: usize,
    ) -> DBResponse {
        let mut params = HashMap::new();
        params.insert(
            "time_period".to_string(),
            serde_json::Value::String(time_period.to_string()),
        );
        params.insert(
            "limit".to_string(),
            serde_json::json!(limit),
        );
        info!(
            "--- TOOL: 查找热点内容 (time_period: {}, limit: {}) ---",
            time_period, limit
        );

        let _days = match time_period {
            "24h" => 1,
            "week" => 7,
            _ => 365,
        };

        // スタブ: 実際のクエリは DB ドライバ統合時に実装
        // 各プラットフォームテーブル (bilibili_video, douyin_aweme, weibo_note,
        // xhs_note, kuaishou_video, zhihu_content) に対して
        // UNION ALL で熱度スコア付きクエリを構築し、ORDER BY hotness_score DESC
        let _tables = [
            "bilibili_video",
            "douyin_aweme",
            "weibo_note",
            "xhs_note",
            "kuaishou_video",
            "zhihu_content",
        ];

        info!(
            "  スタブ: search_hot_content クエリを構築 (time_period={}, limit={})",
            time_period, limit
        );

        DBResponse {
            tool_name: "search_hot_content".to_string(),
            parameters: params,
            results: Vec::new(),
            results_count: 0,
            error_message: None,
            metadata: HashMap::new(),
        }
    }

    /// 【ツール2】グローバルトピック検索
    ///
    /// DB 全体（コンテンツ、コメント、タグ、ソースキーワード）でトピックを全面検索。
    ///
    /// # Arguments
    /// * `topic` - 検索するトピックキーワード
    /// * `limit_per_table` - 各テーブルからの最大レコード数
    pub async fn search_topic_globally(
        &self,
        topic: &str,
        limit_per_table: usize,
    ) -> DBResponse {
        let mut params = HashMap::new();
        params.insert(
            "topic".to_string(),
            serde_json::Value::String(topic.to_string()),
        );
        params.insert(
            "limit_per_table".to_string(),
            serde_json::json!(limit_per_table),
        );
        info!(
            "--- TOOL: 全局话题搜索 (topic: {}, limit_per_table: {}) ---",
            topic, limit_per_table
        );

        let _search_term = format!("%{}%", topic);

        // 検索対象テーブル設定
        let _search_configs: &[(&str, &[&str], &str)] = &[
            ("bilibili_video", &["title", "desc", "source_keyword"], "video"),
            ("bilibili_video_comment", &["content"], "comment"),
            ("douyin_aweme", &["title", "desc", "source_keyword"], "video"),
            ("douyin_aweme_comment", &["content"], "comment"),
            ("kuaishou_video", &["title", "desc", "source_keyword"], "video"),
            ("kuaishou_video_comment", &["content"], "comment"),
            ("weibo_note", &["content", "source_keyword"], "note"),
            ("weibo_note_comment", &["content"], "comment"),
            ("xhs_note", &["title", "desc", "tag_list", "source_keyword"], "note"),
            ("xhs_note_comment", &["content"], "comment"),
            (
                "zhihu_content",
                &["title", "desc", "content_text", "source_keyword"],
                "content",
            ),
            ("zhihu_comment", &["content"], "comment"),
            ("tieba_note", &["title", "desc", "source_keyword"], "note"),
            ("tieba_comment", &["content"], "comment"),
            ("daily_news", &["title"], "news"),
        ];

        info!(
            "  スタブ: search_topic_globally クエリを構築 (topic={}, limit_per_table={})",
            topic, limit_per_table
        );

        DBResponse {
            tool_name: "search_topic_globally".to_string(),
            parameters: params,
            results: Vec::new(),
            results_count: 0,
            error_message: None,
            metadata: HashMap::new(),
        }
    }

    /// 【ツール3】日付範囲トピック検索
    ///
    /// 指定の日付範囲内でトピック関連コンテンツを検索。
    ///
    /// # Arguments
    /// * `topic` - 検索するトピックキーワード
    /// * `start_date` - 開始日 (YYYY-MM-DD)
    /// * `end_date` - 終了日 (YYYY-MM-DD)
    /// * `limit_per_table` - 各テーブルからの最大レコード数
    pub async fn search_topic_by_date(
        &self,
        topic: &str,
        start_date: &str,
        end_date: &str,
        limit_per_table: usize,
    ) -> DBResponse {
        let mut params = HashMap::new();
        params.insert(
            "topic".to_string(),
            serde_json::Value::String(topic.to_string()),
        );
        params.insert(
            "start_date".to_string(),
            serde_json::Value::String(start_date.to_string()),
        );
        params.insert(
            "end_date".to_string(),
            serde_json::Value::String(end_date.to_string()),
        );
        params.insert(
            "limit_per_table".to_string(),
            serde_json::json!(limit_per_table),
        );
        info!(
            "--- TOOL: 按日期搜索话题 (topic: {}, {} ~ {}, limit_per_table: {}) ---",
            topic, start_date, end_date, limit_per_table
        );

        // 日付パース検証
        if NaiveDate::parse_from_str(start_date, "%Y-%m-%d").is_err()
            || NaiveDate::parse_from_str(end_date, "%Y-%m-%d").is_err()
        {
            return DBResponse {
                tool_name: "search_topic_by_date".to_string(),
                parameters: params,
                results: Vec::new(),
                results_count: 0,
                error_message: Some(
                    "日期格式错误，请使用 'YYYY-MM-DD' 格式。".to_string(),
                ),
                metadata: HashMap::new(),
            };
        }

        info!(
            "  スタブ: search_topic_by_date クエリを構築 (topic={}, {} ~ {})",
            topic, start_date, end_date
        );

        DBResponse {
            tool_name: "search_topic_by_date".to_string(),
            parameters: params,
            results: Vec::new(),
            results_count: 0,
            error_message: None,
            metadata: HashMap::new(),
        }
    }

    /// 【ツール4】トピックコメント取得
    ///
    /// 全プラットフォームからトピック関連の公衆コメントデータを専門的に取得。
    ///
    /// # Arguments
    /// * `topic` - 検索するトピックキーワード
    /// * `limit` - 返却コメント総数の上限
    pub async fn get_comments_for_topic(
        &self,
        topic: &str,
        limit: usize,
    ) -> DBResponse {
        let mut params = HashMap::new();
        params.insert(
            "topic".to_string(),
            serde_json::Value::String(topic.to_string()),
        );
        params.insert("limit".to_string(), serde_json::json!(limit));
        info!(
            "--- TOOL: 获取话题评论 (topic: {}, limit: {}) ---",
            topic, limit
        );

        let _comment_tables = [
            "bilibili_video_comment",
            "douyin_aweme_comment",
            "kuaishou_video_comment",
            "weibo_note_comment",
            "xhs_note_comment",
            "zhihu_comment",
            "tieba_comment",
        ];

        info!(
            "  スタブ: get_comments_for_topic クエリを構築 (topic={}, limit={})",
            topic, limit
        );

        DBResponse {
            tool_name: "get_comments_for_topic".to_string(),
            parameters: params,
            results: Vec::new(),
            results_count: 0,
            error_message: None,
            metadata: HashMap::new(),
        }
    }

    /// 【ツール5】プラットフォーム定向検索
    ///
    /// 指定プラットフォーム上でトピックを精確に検索。
    ///
    /// # Arguments
    /// * `platform` - プラットフォーム名 (bilibili, weibo, douyin, kuaishou, xhs, zhihu, tieba)
    /// * `topic` - 検索するトピックキーワード
    /// * `start_date` - 開始日 (YYYY-MM-DD, Optional)
    /// * `end_date` - 終了日 (YYYY-MM-DD, Optional)
    /// * `limit` - 最大結果数
    pub async fn search_topic_on_platform(
        &self,
        platform: &str,
        topic: &str,
        start_date: Option<&str>,
        end_date: Option<&str>,
        limit: usize,
    ) -> DBResponse {
        let mut params = HashMap::new();
        params.insert(
            "platform".to_string(),
            serde_json::Value::String(platform.to_string()),
        );
        params.insert(
            "topic".to_string(),
            serde_json::Value::String(topic.to_string()),
        );
        if let Some(sd) = start_date {
            params.insert(
                "start_date".to_string(),
                serde_json::Value::String(sd.to_string()),
            );
        }
        if let Some(ed) = end_date {
            params.insert(
                "end_date".to_string(),
                serde_json::Value::String(ed.to_string()),
            );
        }
        params.insert("limit".to_string(), serde_json::json!(limit));
        info!(
            "--- TOOL: 平台定向搜索 (platform: {}, topic: {}, limit: {}) ---",
            platform, topic, limit
        );

        if !SUPPORTED_PLATFORMS.contains(&platform) {
            return DBResponse {
                tool_name: "search_topic_on_platform".to_string(),
                parameters: params,
                error_message: Some(format!("不支持的平台: {}", platform)),
                ..Default::default()
            };
        }

        info!(
            "  スタブ: search_topic_on_platform クエリを構築 (platform={}, topic={}, limit={})",
            platform, topic, limit
        );

        DBResponse {
            tool_name: "search_topic_on_platform".to_string(),
            parameters: params,
            results: Vec::new(),
            results_count: 0,
            error_message: None,
            metadata: HashMap::new(),
        }
    }
}

impl Default for MediaCrawlerDB {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ヘルパー関数
// ---------------------------------------------------------------------------

/// DBResponse のサマリーを表示
#[allow(dead_code)]
pub fn print_response_summary(response: &DBResponse) {
    if let Some(ref err) = response.error_message {
        warn!("ツール '{}' 実行エラー: {}", response.tool_name, err);
        return;
    }

    let params_str: String = response
        .parameters
        .iter()
        .map(|(k, v)| format!("{}={}", k, v))
        .collect::<Vec<_>>()
        .join(", ");
    info!(
        "クエリ: ツール='{}', パラメータ=[{}]",
        response.tool_name, params_str
    );
    info!("{} 件の関連レコードを発見。", response.results_count);

    if !response.results.is_empty() {
        let preview_count = response.results.len().min(5);
        for (idx, res) in response.results[..preview_count].iter().enumerate() {
            let content_preview: String = res
                .title_or_content
                .replace('\n', " ")
                .chars()
                .take(70)
                .collect();
            let author = res.author_nickname.as_deref().unwrap_or("N/A");
            let publish_time = res.publish_time.as_deref().unwrap_or("N/A");
            let hotness = if res.hotness_score > 0.0 {
                format!(", hotness: {:.2}", res.hotness_score)
            } else {
                String::new()
            };
            info!(
                "{}. [{}/{}] {}... 作者: {} | 時間: {}{}",
                idx + 1,
                res.platform.to_uppercase(),
                res.content_type,
                content_preview,
                author,
                publish_time,
                hotness,
            );
        }
    }
}
