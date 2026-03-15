//! BettaFish MediaEngine
//!
//! Python の MediaEngine の Rust 実装。
//! Bocha/Anspire API を使ったマルチモーダル検索 AI エージェント。

#![allow(dead_code)]

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{Local, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Duration;
use tracing::{error, info, warn};

use bettafish_common::forum_reader::{format_host_speech_for_prompt, get_latest_host_speech};
use bettafish_common::retry::{search_api_retry_config, with_graceful_retry};
use bettafish_common::text_processing::{
    clean_json_tags, clean_markdown_tags, extract_clean_response, fix_incomplete_json,
    format_search_results_for_prompt, remove_reasoning_from_output,
};
use bettafish_config::Settings;
use bettafish_llm::{InvokeOptions, LLMClient};

/// バージョン情報
pub const VERSION: &str = "1.0.0";

// =============================================================================
// データ構造定義
// =============================================================================

/// ウェブページ検索結果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebpageResult {
    /// ページ名
    pub name: String,
    /// URL
    pub url: String,
    /// スニペット
    pub snippet: String,
    /// 表示用 URL
    pub display_url: Option<String>,
    /// 最終クロール日
    pub date_last_crawled: Option<String>,
}

/// 画像検索結果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageResult {
    /// 画像名
    pub name: String,
    /// コンテンツ URL
    pub content_url: String,
    /// ホストページ URL
    pub host_page_url: Option<String>,
    /// サムネイル URL
    pub thumbnail_url: Option<String>,
    /// 幅
    pub width: Option<u32>,
    /// 高さ
    pub height: Option<u32>,
}

/// モーダルカード構造化データ結果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModalCardResult {
    /// カードタイプ (例: weather_china, stock, baike_pro, medical_common)
    pub card_type: String,
    /// 解析済みの JSON コンテンツ
    pub content: Value,
}

/// Bocha API の完全レスポンス
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BochaResponse {
    /// 検索クエリ
    pub query: String,
    /// 会話 ID
    pub conversation_id: Option<String>,
    /// AI 生成の総括回答
    pub answer: Option<String>,
    /// AI 生成の追加質問
    pub follow_ups: Vec<String>,
    /// ウェブページ結果
    pub webpages: Vec<WebpageResult>,
    /// 画像結果
    pub images: Vec<ImageResult>,
    /// モーダルカード結果
    pub modal_cards: Vec<ModalCardResult>,
}

impl Default for BochaResponse {
    fn default() -> Self {
        Self {
            query: "検索失敗".to_string(),
            conversation_id: None,
            answer: None,
            follow_ups: Vec::new(),
            webpages: Vec::new(),
            images: Vec::new(),
            modal_cards: Vec::new(),
        }
    }
}

/// Anspire API の完全レスポンス
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnspireResponse {
    /// 検索クエリ
    pub query: String,
    /// 会話 ID
    pub conversation_id: Option<String>,
    /// スコア
    pub score: Option<f64>,
    /// ウェブページ結果
    pub webpages: Vec<WebpageResult>,
}

impl Default for AnspireResponse {
    fn default() -> Self {
        Self {
            query: "検索失敗".to_string(),
            conversation_id: None,
            score: None,
            webpages: Vec::new(),
        }
    }
}

// =============================================================================
// Bocha API 内部レスポンス構造
// =============================================================================

#[derive(Debug, Deserialize)]
struct BochaApiResponse {
    code: Option<i32>,
    msg: Option<String>,
    conversation_id: Option<String>,
    messages: Option<Vec<BochaApiMessage>>,
}

#[derive(Debug, Deserialize)]
struct BochaApiMessage {
    role: Option<String>,
    #[serde(rename = "type")]
    msg_type: Option<String>,
    content_type: Option<String>,
    content: Option<String>,
}

// =============================================================================
// Anspire API 内部レスポンス構造
// =============================================================================

#[derive(Debug, Deserialize)]
struct AnspireApiResponse {
    #[serde(rename = "Uuid")]
    uuid: Option<String>,
    results: Option<Vec<AnspireApiResult>>,
}

#[derive(Debug, Deserialize)]
struct AnspireApiResult {
    title: Option<String>,
    url: Option<String>,
    content: Option<String>,
    date: Option<String>,
    score: Option<f64>,
}

// =============================================================================
// BochaMultimodalSearch クライアント
// =============================================================================

/// Bocha マルチモーダル検索クライアント
///
/// 5 種類の検索ツールを提供:
/// 1. comprehensive_search - 全面的な総合検索
/// 2. web_search_only - 純粋なウェブ検索
/// 3. search_for_structured_data - 構造化データ検索
/// 4. search_last_24_hours - 24 時間以内の情報検索
/// 5. search_last_week - 今週の情報検索
pub struct BochaMultimodalSearch {
    base_url: String,
    api_key: String,
    client: Client,
}

impl BochaMultimodalSearch {
    /// クライアントを初期化
    pub fn new(api_key: &str, base_url: Option<&str>) -> Result<Self> {
        if api_key.is_empty() {
            anyhow::bail!("Bocha API Key が見つかりません！BOCHA_WEB_SEARCH_API_KEY 環境変数を設定するか、初期化時に提供してください");
        }

        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .context("HTTP クライアントの作成に失敗")?;

        let base_url = base_url
            .filter(|u| !u.is_empty())
            .unwrap_or("https://api.bocha.cn/v1/ai-search")
            .to_string();

        Ok(Self {
            base_url,
            api_key: api_key.to_string(),
            client,
        })
    }

    /// API レスポンスを解析
    fn parse_search_response(&self, response_dict: &Value, query: &str) -> BochaResponse {
        let mut final_response = BochaResponse {
            query: query.to_string(),
            ..Default::default()
        };

        final_response.conversation_id = response_dict
            .get("conversation_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let messages = response_dict
            .get("messages")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        for msg in &messages {
            let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("");
            if role != "assistant" {
                continue;
            }

            let msg_type = msg.get("type").and_then(|v| v.as_str()).unwrap_or("");
            let content_type = msg.get("content_type").and_then(|v| v.as_str()).unwrap_or("");
            let content_str = msg.get("content").and_then(|v| v.as_str()).unwrap_or("{}");

            // コンテンツを解析（JSON またはプレーンテキスト）
            let content_data: Value = serde_json::from_str(content_str)
                .unwrap_or_else(|_| Value::String(content_str.to_string()));

            match (msg_type, content_type) {
                ("answer", "text") => {
                    final_response.answer = match &content_data {
                        Value::String(s) => Some(s.clone()),
                        _ => Some(content_str.to_string()),
                    };
                }
                ("follow_up", "text") => {
                    match &content_data {
                        Value::String(s) => final_response.follow_ups.push(s.clone()),
                        _ => final_response.follow_ups.push(content_str.to_string()),
                    }
                }
                ("source", "webpage") => {
                    let web_results = content_data
                        .get("value")
                        .and_then(|v| v.as_array())
                        .cloned()
                        .unwrap_or_default();
                    for item in &web_results {
                        final_response.webpages.push(WebpageResult {
                            name: item.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                            url: item.get("url").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                            snippet: item.get("snippet").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                            display_url: item.get("displayUrl").and_then(|v| v.as_str()).map(|s| s.to_string()),
                            date_last_crawled: item.get("dateLastCrawled").and_then(|v| v.as_str()).map(|s| s.to_string()),
                        });
                    }
                }
                ("source", "image") => {
                    final_response.images.push(ImageResult {
                        name: content_data.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                        content_url: content_data.get("contentUrl").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                        host_page_url: content_data.get("hostPageUrl").and_then(|v| v.as_str()).map(|s| s.to_string()),
                        thumbnail_url: content_data.get("thumbnailUrl").and_then(|v| v.as_str()).map(|s| s.to_string()),
                        width: content_data.get("width").and_then(|v| v.as_u64()).map(|v| v as u32),
                        height: content_data.get("height").and_then(|v| v.as_u64()).map(|v| v as u32),
                    });
                }
                ("source", _) => {
                    // その他の content_type はモーダルカードとして扱う
                    final_response.modal_cards.push(ModalCardResult {
                        card_type: content_type.to_string(),
                        content: content_data,
                    });
                }
                _ => {}
            }
        }

        final_response
    }

    /// 内部検索実行
    async fn search_internal(&self, params: Value) -> Result<BochaResponse> {
        let query = params
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown Query")
            .to_string();

        let mut payload = serde_json::json!({ "stream": false });
        if let Some(obj) = params.as_object() {
            for (k, v) in obj {
                payload[k] = v.clone();
            }
        }

        let response = self
            .client
            .post(&self.base_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .header("Accept", "*/*")
            .json(&payload)
            .send()
            .await
            .context("Bocha API リクエスト送信に失敗")?;

        let status = response.status();
        if !status.is_success() {
            let err_body = response.text().await.unwrap_or_default();
            anyhow::bail!("Bocha API エラー ({}): {}", status, err_body);
        }

        let response_dict: Value = response
            .json()
            .await
            .context("Bocha レスポンスの解析に失敗")?;

        let code = response_dict.get("code").and_then(|v| v.as_i64()).unwrap_or(0);
        if code != 200 {
            let msg = response_dict.get("msg").and_then(|v| v.as_str()).unwrap_or("未知エラー");
            error!("API がエラーを返しました: {}", msg);
            return Ok(BochaResponse {
                query,
                ..Default::default()
            });
        }

        Ok(self.parse_search_response(&response_dict, &query))
    }

    // ===== Agent が使用できるツールメソッド =====

    /// 【ツール】全面総合検索: 網頁、画像、AI 総括、追問提案、モーダルカードを返す
    pub async fn comprehensive_search(&self, query: &str, max_results: usize) -> BochaResponse {
        info!("--- TOOL: 全面総合検索 (query: {}) ---", query);
        let config = search_api_retry_config();
        let default = BochaResponse::default();

        with_graceful_retry(&config, "comprehensive_search", default, || {
            let params = serde_json::json!({
                "query": query,
                "count": max_results,
                "answer": true,
            });
            async move { self.search_internal(params).await }
        })
        .await
    }

    /// 【ツール】純ウェブ検索: AI 総括なし、高速・低コスト
    pub async fn web_search_only(&self, query: &str, max_results: usize) -> BochaResponse {
        info!("--- TOOL: 純ウェブ検索 (query: {}) ---", query);
        let config = search_api_retry_config();
        let default = BochaResponse::default();

        with_graceful_retry(&config, "web_search_only", default, || {
            let params = serde_json::json!({
                "query": query,
                "count": max_results,
                "answer": false,
            });
            async move { self.search_internal(params).await }
        })
        .await
    }

    /// 【ツール】構造化データ検索: 天気、株価、為替などのモーダルカード取得
    pub async fn search_for_structured_data(&self, query: &str) -> BochaResponse {
        info!("--- TOOL: 構造化データ検索 (query: {}) ---", query);
        let config = search_api_retry_config();
        let default = BochaResponse::default();

        with_graceful_retry(&config, "search_for_structured_data", default, || {
            let params = serde_json::json!({
                "query": query,
                "count": 5,
                "answer": true,
            });
            async move { self.search_internal(params).await }
        })
        .await
    }

    /// 【ツール】24 時間以内情報検索: 最新動態追跡
    pub async fn search_last_24_hours(&self, query: &str) -> BochaResponse {
        info!("--- TOOL: 24 時間以内情報検索 (query: {}) ---", query);
        let config = search_api_retry_config();
        let default = BochaResponse::default();

        with_graceful_retry(&config, "search_last_24_hours", default, || {
            let params = serde_json::json!({
                "query": query,
                "freshness": "oneDay",
                "answer": true,
            });
            async move { self.search_internal(params).await }
        })
        .await
    }

    /// 【ツール】今週情報検索: 過去一週間の報道
    pub async fn search_last_week(&self, query: &str) -> BochaResponse {
        info!("--- TOOL: 今週情報検索 (query: {}) ---", query);
        let config = search_api_retry_config();
        let default = BochaResponse::default();

        with_graceful_retry(&config, "search_last_week", default, || {
            let params = serde_json::json!({
                "query": query,
                "freshness": "oneWeek",
                "answer": true,
            });
            async move { self.search_internal(params).await }
        })
        .await
    }
}

// =============================================================================
// AnspireAISearch クライアント
// =============================================================================

/// Anspire AI 検索クライアント
///
/// 3 種類の検索ツールを提供:
/// 1. comprehensive_search - 総合検索
/// 2. search_last_24_hours - 24 時間以内の情報検索
/// 3. search_last_week - 今週の情報検索
pub struct AnspireAISearch {
    base_url: String,
    api_key: String,
    client: Client,
}

impl AnspireAISearch {
    /// クライアントを初期化
    pub fn new(api_key: &str, base_url: Option<&str>) -> Result<Self> {
        if api_key.is_empty() {
            anyhow::bail!("Anspire API Key が見つかりません！ANSPIRE_API_KEY 環境変数を設定するか、初期化時に提供してください");
        }

        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .context("HTTP クライアントの作成に失敗")?;

        let base_url = base_url
            .filter(|u| !u.is_empty())
            .unwrap_or("https://plugin.anspire.cn/api/ntsearch/search")
            .to_string();

        Ok(Self {
            base_url,
            api_key: api_key.to_string(),
            client,
        })
    }

    /// API レスポンスを解析
    fn parse_search_response(&self, response_dict: &Value, query: &str) -> AnspireResponse {
        let mut final_response = AnspireResponse {
            query: query.to_string(),
            ..Default::default()
        };

        final_response.conversation_id = response_dict
            .get("Uuid")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let results = response_dict
            .get("results")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        for msg in &results {
            final_response.score = msg.get("score").and_then(|v| v.as_f64());
            final_response.webpages.push(WebpageResult {
                name: msg.get("title").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                url: msg.get("url").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                snippet: msg.get("content").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                display_url: None,
                date_last_crawled: msg.get("date").and_then(|v| v.as_str()).map(|s| s.to_string()),
            });
        }

        final_response
    }

    /// 内部検索実行 (GET メソッド)
    async fn search_internal(
        &self,
        query: &str,
        top_k: usize,
        from_time: &str,
        to_time: &str,
    ) -> Result<AnspireResponse> {
        let params = vec![
            ("query", query.to_string()),
            ("top_k", top_k.to_string()),
            ("Insite", String::new()),
            ("FromTime", from_time.to_string()),
            ("ToTime", to_time.to_string()),
        ];

        let response = self
            .client
            .get(&self.base_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .header("Connection", "keep-alive")
            .header("Accept", "*/*")
            .query(&params)
            .send()
            .await
            .context("Anspire API リクエスト送信に失敗")?;

        let status = response.status();
        if !status.is_success() {
            let err_body = response.text().await.unwrap_or_default();
            anyhow::bail!("Anspire API エラー ({}): {}", status, err_body);
        }

        let response_dict: Value = response
            .json()
            .await
            .context("Anspire レスポンスの解析に失敗")?;

        Ok(self.parse_search_response(&response_dict, query))
    }

    // ===== Agent が使用できるツールメソッド =====

    /// 【ツール】総合検索: 全面的な情報取得
    pub async fn comprehensive_search(&self, query: &str, max_results: usize) -> AnspireResponse {
        info!("--- TOOL: 総合検索 (query: {}) ---", query);
        let config = search_api_retry_config();
        let default = AnspireResponse::default();

        with_graceful_retry(&config, "comprehensive_search", default, || {
            async move { self.search_internal(query, max_results, "", "").await }
        })
        .await
    }

    /// 【ツール】24 時間以内情報検索: 最新動態追跡
    pub async fn search_last_24_hours(&self, query: &str, max_results: usize) -> AnspireResponse {
        info!("--- TOOL: 24 時間以内情報検索 (query: {}) ---", query);
        let config = search_api_retry_config();
        let default = AnspireResponse::default();

        let now = Utc::now();
        let from_time = (now - chrono::Duration::days(1)).format("%Y-%m-%d %H:%M:%S").to_string();
        let to_time = now.format("%Y-%m-%d %H:%M:%S").to_string();

        with_graceful_retry(&config, "search_last_24_hours", default, || {
            let ft = from_time.clone();
            let tt = to_time.clone();
            async move { self.search_internal(query, max_results, &ft, &tt).await }
        })
        .await
    }

    /// 【ツール】今週情報検索: 過去一週間の報道
    pub async fn search_last_week(&self, query: &str, max_results: usize) -> AnspireResponse {
        info!("--- TOOL: 今週情報検索 (query: {}) ---", query);
        let config = search_api_retry_config();
        let default = AnspireResponse::default();

        let now = Utc::now();
        let from_time = (now - chrono::Duration::weeks(1)).format("%Y-%m-%d %H:%M:%S").to_string();
        let to_time = now.format("%Y-%m-%d %H:%M:%S").to_string();

        with_graceful_retry(&config, "search_last_week", default, || {
            let ft = from_time.clone();
            let tt = to_time.clone();
            async move { self.search_internal(query, max_results, &ft, &tt).await }
        })
        .await
    }
}

// =============================================================================
// 検索レスポンスの統一変換トレイト
// =============================================================================

/// 検索結果を統一 Value 形式に変換するトレイト
trait IntoSearchResults {
    fn webpages(&self) -> &[WebpageResult];
}

impl IntoSearchResults for BochaResponse {
    fn webpages(&self) -> &[WebpageResult] {
        &self.webpages
    }
}

impl IntoSearchResults for AnspireResponse {
    fn webpages(&self) -> &[WebpageResult] {
        &self.webpages
    }
}

/// ウェブページ結果リストを Value ベクトルに変換
fn convert_webpages_to_values(webpages: &[WebpageResult], max_results: usize) -> Vec<Value> {
    let limit = webpages.len().min(max_results);
    webpages[..limit]
        .iter()
        .map(|wp| {
            serde_json::json!({
                "title": wp.name,
                "url": wp.url,
                "content": wp.snippet,
                "score": Value::Null,
                "raw_content": wp.snippet,
                "published_date": wp.date_last_crawled,
            })
        })
        .collect()
}

// =============================================================================
// 状態管理 (QueryEngine と同構造)
// =============================================================================

/// 単一検索結果の状態
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Search {
    pub query: String,
    pub url: String,
    pub title: String,
    pub content: String,
    pub score: Option<f64>,
    pub timestamp: String,
}

impl Default for Search {
    fn default() -> Self {
        Self {
            query: String::new(),
            url: String::new(),
            title: String::new(),
            content: String::new(),
            score: None,
            timestamp: Local::now().to_rfc3339(),
        }
    }
}

/// 段落研究プロセスの状態
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Research {
    pub search_history: Vec<Search>,
    pub latest_summary: String,
    pub reflection_iteration: usize,
    pub is_completed: bool,
}

impl Default for Research {
    fn default() -> Self {
        Self {
            search_history: Vec::new(),
            latest_summary: String::new(),
            reflection_iteration: 0,
            is_completed: false,
        }
    }
}

impl Research {
    pub fn add_search(&mut self, search: Search) {
        self.search_history.push(search);
    }

    pub fn add_search_results(&mut self, query: &str, results: &[Value]) {
        for result in results {
            let search = Search {
                query: query.to_string(),
                url: result.get("url").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                title: result.get("title").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                content: result.get("content").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                score: result.get("score").and_then(|v| v.as_f64()),
                timestamp: Local::now().to_rfc3339(),
            };
            self.add_search(search);
        }
    }

    pub fn increment_reflection(&mut self) {
        self.reflection_iteration += 1;
    }

    pub fn mark_completed(&mut self) {
        self.is_completed = true;
    }
}

/// レポート中の単一段落の状態
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Paragraph {
    pub title: String,
    pub content: String,
    pub research: Research,
    pub order: usize,
}

impl Default for Paragraph {
    fn default() -> Self {
        Self {
            title: String::new(),
            content: String::new(),
            research: Research::default(),
            order: 0,
        }
    }
}

impl Paragraph {
    pub fn is_completed(&self) -> bool {
        self.research.is_completed && !self.research.latest_summary.is_empty()
    }

    pub fn get_final_content(&self) -> &str {
        if self.research.latest_summary.is_empty() {
            &self.content
        } else {
            &self.research.latest_summary
        }
    }
}

/// レポート全体の状態
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct State {
    pub query: String,
    pub report_title: String,
    pub paragraphs: Vec<Paragraph>,
    pub final_report: String,
    pub is_completed: bool,
    pub created_at: String,
    pub updated_at: String,
}

impl Default for State {
    fn default() -> Self {
        let now = Local::now().to_rfc3339();
        Self {
            query: String::new(),
            report_title: String::new(),
            paragraphs: Vec::new(),
            final_report: String::new(),
            is_completed: false,
            created_at: now.clone(),
            updated_at: now,
        }
    }
}

impl State {
    pub fn add_paragraph(&mut self, title: &str, content: &str) -> usize {
        let order = self.paragraphs.len();
        self.paragraphs.push(Paragraph {
            title: title.to_string(),
            content: content.to_string(),
            research: Research::default(),
            order,
        });
        self.update_timestamp();
        order
    }

    pub fn update_timestamp(&mut self) {
        self.updated_at = Local::now().to_rfc3339();
    }

    pub fn mark_completed(&mut self) {
        self.is_completed = true;
        self.update_timestamp();
    }

    pub fn get_progress_summary(&self) -> Value {
        let completed = self.paragraphs.iter().filter(|p| p.is_completed()).count();
        let total = self.paragraphs.len();
        let percentage = if total > 0 {
            completed as f64 / total as f64 * 100.0
        } else {
            0.0
        };

        serde_json::json!({
            "total_paragraphs": total,
            "completed_paragraphs": completed,
            "progress_percentage": percentage,
            "is_completed": self.is_completed,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        })
    }

    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    pub fn from_json(json_str: &str) -> serde_json::Result<Self> {
        serde_json::from_str(json_str)
    }

    pub fn save_to_file(&self, filepath: &str) -> Result<()> {
        let json = self.to_json()?;
        fs::write(filepath, json)?;
        Ok(())
    }

    pub fn load_from_file(filepath: &str) -> Result<Self> {
        let content = fs::read_to_string(filepath)?;
        let state: Self = serde_json::from_str(&content)?;
        Ok(state)
    }
}

// =============================================================================
// ノード基底トレイト
// =============================================================================

/// ノード基底トレイト
#[async_trait]
trait Node: Send + Sync {
    fn name(&self) -> &str;
    async fn run(&self, input_data: &Value) -> Result<Value>;
}

/// 状態変更機能付きノードトレイト
#[async_trait]
trait StateMutationNode: Node {
    async fn mutate_state(
        &self,
        input_data: &Value,
        state: &mut State,
        paragraph_index: usize,
    ) -> Result<()>;
}

// =============================================================================
// プロンプト定義 (MediaEngine 固有 - 多モーダル分析重視)
// =============================================================================

/// レポート構造出力スキーマ
const OUTPUT_SCHEMA_REPORT_STRUCTURE: &str = r#"{
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "title": {"type": "string"},
      "content": {"type": "string"}
    }
  }
}"#;

/// 初回検索入力スキーマ
const INPUT_SCHEMA_FIRST_SEARCH: &str = r#"{
  "type": "object",
  "properties": {
    "title": {"type": "string"},
    "content": {"type": "string"}
  }
}"#;

/// 初回検索出力スキーマ
const OUTPUT_SCHEMA_FIRST_SEARCH: &str = r#"{
  "type": "object",
  "properties": {
    "search_query": {"type": "string"},
    "search_tool": {"type": "string"},
    "reasoning": {"type": "string"}
  },
  "required": ["search_query", "search_tool", "reasoning"]
}"#;

/// 初回サマリー入力スキーマ
const INPUT_SCHEMA_FIRST_SUMMARY: &str = r#"{
  "type": "object",
  "properties": {
    "title": {"type": "string"},
    "content": {"type": "string"},
    "search_query": {"type": "string"},
    "search_results": {
      "type": "array",
      "items": {"type": "string"}
    }
  }
}"#;

/// 初回サマリー出力スキーマ
const OUTPUT_SCHEMA_FIRST_SUMMARY: &str = r#"{
  "type": "object",
  "properties": {
    "paragraph_latest_state": {"type": "string"}
  }
}"#;

/// リフレクション入力スキーマ
const INPUT_SCHEMA_REFLECTION: &str = r#"{
  "type": "object",
  "properties": {
    "title": {"type": "string"},
    "content": {"type": "string"},
    "paragraph_latest_state": {"type": "string"}
  }
}"#;

/// リフレクション出力スキーマ
const OUTPUT_SCHEMA_REFLECTION: &str = r#"{
  "type": "object",
  "properties": {
    "search_query": {"type": "string"},
    "search_tool": {"type": "string"},
    "reasoning": {"type": "string"}
  },
  "required": ["search_query", "search_tool", "reasoning"]
}"#;

/// リフレクションサマリー入力スキーマ
const INPUT_SCHEMA_REFLECTION_SUMMARY: &str = r#"{
  "type": "object",
  "properties": {
    "title": {"type": "string"},
    "content": {"type": "string"},
    "search_query": {"type": "string"},
    "search_results": {
      "type": "array",
      "items": {"type": "string"}
    },
    "paragraph_latest_state": {"type": "string"}
  }
}"#;

/// リフレクションサマリー出力スキーマ
const OUTPUT_SCHEMA_REFLECTION_SUMMARY: &str = r#"{
  "type": "object",
  "properties": {
    "updated_paragraph_latest_state": {"type": "string"}
  }
}"#;

/// レポートフォーマット入力スキーマ
const INPUT_SCHEMA_REPORT_FORMATTING: &str = r#"{
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "title": {"type": "string"},
      "paragraph_latest_state": {"type": "string"}
    }
  }
}"#;

fn system_prompt_report_structure() -> String {
    format!(
        r#"你是一位深度研究助手。给定一个查询，你需要规划一个报告的结构和其中包含的段落。最多5个段落。
确保段落的排序合理有序。
一旦大纲创建完成，你将获得工具来分别为每个部分搜索网络并进行反思。
请按照以下JSON模式定义格式化输出：

<OUTPUT JSON SCHEMA>
{schema}
</OUTPUT JSON SCHEMA>

标题和内容属性将用于更深入的研究。
确保输出是一个符合上述输出JSON模式定义的JSON对象。
只返回JSON对象，不要有解释或额外文本。"#,
        schema = OUTPUT_SCHEMA_REPORT_STRUCTURE
    )
}

fn system_prompt_first_search() -> String {
    format!(
        r#"你是一位深度研究助手。你将获得报告中的一个段落，其标题和预期内容将按照以下JSON模式定义提供：

<INPUT JSON SCHEMA>
{input_schema}
</INPUT JSON SCHEMA>

你可以使用以下5种专业的多模态搜索工具：

1. **comprehensive_search** - 全面综合搜索工具
   - 适用于：一般性的研究需求，需要完整信息时
   - 特点：返回网页、图片、AI总结、追问建议和可能的结构化数据，是最常用的基础工具

2. **web_search_only** - 纯网页搜索工具
   - 适用于：只需要网页链接和摘要，不需要AI分析时
   - 特点：速度更快，成本更低，只返回网页结果

3. **search_for_structured_data** - 结构化数据查询工具
   - 适用于：查询天气、股票、汇率、百科定义等结构化信息时
   - 特点：专门用于触发"模态卡"的查询，返回结构化数据

4. **search_last_24_hours** - 24小时内信息搜索工具
   - 适用于：需要了解最新动态、突发事件时
   - 特点：只搜索过去24小时内发布的内容

5. **search_last_week** - 本周信息搜索工具
   - 适用于：需要了解近期发展趋势时
   - 特点：搜索过去一周内的主要报道

你的任务是：
1. 根据段落主题选择最合适的搜索工具
2. 制定最佳的搜索查询
3. 解释你的选择理由

注意：所有工具都不需要额外参数，选择工具主要基于搜索意图和需要的信息类型。
请按照以下JSON模式定义格式化输出（文字请使用中文）：

<OUTPUT JSON SCHEMA>
{output_schema}
</OUTPUT JSON SCHEMA>

确保输出是一个符合上述输出JSON模式定义的JSON对象。
只返回JSON对象，不要有解释或额外文本。"#,
        input_schema = INPUT_SCHEMA_FIRST_SEARCH,
        output_schema = OUTPUT_SCHEMA_FIRST_SEARCH
    )
}

fn system_prompt_first_summary() -> String {
    format!(
        r#"你是一位专业的多媒体内容分析师和深度报告撰写专家。你将获得搜索查询、多模态搜索结果以及你正在研究的报告段落，数据将按照以下JSON模式定义提供：

<INPUT JSON SCHEMA>
{input_schema}
</INPUT JSON SCHEMA>

**你的核心任务：创建信息丰富、多维度的综合分析段落（每段不少于800-1200字）**

**撰写标准和多模态内容整合要求：**

1. **开篇概述**：
   - 用2-3句话明确本段的分析焦点和核心问题
   - 突出多模态信息的整合价值

2. **多源信息整合层次**：
   - **网页内容分析**：详细分析网页搜索结果中的文字信息、数据、观点
   - **图片信息解读**：深入分析相关图片所传达的信息、情感、视觉元素
   - **AI总结整合**：利用AI总结信息，提炼关键观点和趋势
   - **结构化数据应用**：充分利用天气、股票、百科等结构化信息（如适用）

3. **内容结构化组织**：
   ## 综合信息概览
   [多种信息源的核心发现]

   ## 文本内容深度分析
   [网页、文章内容的详细分析]

   ## 视觉信息解读
   [图片、多媒体内容的分析]

   ## 数据综合分析
   [各类数据的整合分析]

   ## 多维度洞察
   [基于多种信息源的深度洞察]

4. **具体内容要求**：
   - **文本引用**：大量引用搜索结果中的具体文字内容
   - **图片描述**：详细描述相关图片的内容、风格、传达的信息
   - **数据提取**：准确提取和分析各种数据信息
   - **趋势识别**：基于多源信息识别发展趋势和模式

5. **信息密度标准**：
   - 每100字至少包含2-3个来自不同信息源的具体信息点
   - 充分利用搜索结果的多样性和丰富性
   - 避免信息冗余，确保每个信息点都有价值
   - 实现文字、图像、数据的有机结合

6. **分析深度要求**：
   - **关联分析**：分析不同信息源之间的关联性和一致性
   - **对比分析**：比较不同来源信息的差异和互补性
   - **趋势分析**：基于多源信息判断发展趋势
   - **影响评估**：评估事件或话题的影响范围和程度

7. **多模态特色体现**：
   - **视觉化描述**：用文字生动描述图片内容和视觉冲击
   - **数据可视**：将数字信息转化为易理解的描述
   - **立体化分析**：从多个感官和维度理解分析对象
   - **综合判断**：基于文字、图像、数据的综合判断

8. **语言表达要求**：
   - 准确、客观、具有分析深度
   - 既要专业又要生动有趣
   - 充分体现多模态信息的丰富性
   - 逻辑清晰，条理分明

请按照以下JSON模式定义格式化输出：

<OUTPUT JSON SCHEMA>
{output_schema}
</OUTPUT JSON SCHEMA>

确保输出是一个符合上述输出JSON模式定义的JSON对象。
只返回JSON对象，不要有解释或额外文本。"#,
        input_schema = INPUT_SCHEMA_FIRST_SUMMARY,
        output_schema = OUTPUT_SCHEMA_FIRST_SUMMARY
    )
}

fn system_prompt_reflection() -> String {
    format!(
        r#"你是一位深度研究助手。你负责为研究报告构建全面的段落。你将获得段落标题、计划内容摘要，以及你已经创建的段落最新状态，所有这些都将按照以下JSON模式定义提供：

<INPUT JSON SCHEMA>
{input_schema}
</INPUT JSON SCHEMA>

你可以使用以下5种专业的多模态搜索工具：

1. **comprehensive_search** - 全面综合搜索工具
2. **web_search_only** - 纯网页搜索工具
3. **search_for_structured_data** - 结构化数据查询工具
4. **search_last_24_hours** - 24小时内信息搜索工具
5. **search_last_week** - 本周信息搜索工具

你的任务是：
1. 反思段落文本的当前状态，思考是否遗漏了主题的某些关键方面
2. 选择最合适的搜索工具来补充缺失信息
3. 制定精确的搜索查询
4. 解释你的选择和推理

注意：所有工具都不需要额外参数，选择工具主要基于搜索意图和需要的信息类型。
请按照以下JSON模式定义格式化输出：

<OUTPUT JSON SCHEMA>
{output_schema}
</OUTPUT JSON SCHEMA>

确保输出是一个符合上述输出JSON模式定义的JSON对象。
只返回JSON对象，不要有解释或额外文本。"#,
        input_schema = INPUT_SCHEMA_REFLECTION,
        output_schema = OUTPUT_SCHEMA_REFLECTION
    )
}

fn system_prompt_reflection_summary() -> String {
    format!(
        r#"你是一位深度研究助手。
你将获得搜索查询、搜索结果、段落标题以及你正在研究的报告段落的预期内容。
你正在迭代完善这个段落，并且段落的最新状态也会提供给你。
数据将按照以下JSON模式定义提供：

<INPUT JSON SCHEMA>
{input_schema}
</INPUT JSON SCHEMA>

你的任务是根据搜索结果和预期内容丰富段落的当前最新状态。
不要删除最新状态中的关键信息，尽量丰富它，只添加缺失的信息。
适当地组织段落结构以便纳入报告中。
请按照以下JSON模式定义格式化输出：

<OUTPUT JSON SCHEMA>
{output_schema}
</OUTPUT JSON SCHEMA>

确保输出是一个符合上述输出JSON模式定义的JSON对象。
只返回JSON对象，不要有解释或额外文本。"#,
        input_schema = INPUT_SCHEMA_REFLECTION_SUMMARY,
        output_schema = OUTPUT_SCHEMA_REFLECTION_SUMMARY
    )
}

fn system_prompt_report_formatting() -> String {
    format!(
        r#"你是一位资深的多媒体内容分析专家和融合报告编辑。你专精于将文字、图像、数据等多维信息整合为全景式的综合分析报告。
你将获得以下JSON格式的数据：

<INPUT JSON SCHEMA>
{input_schema}
</INPUT JSON SCHEMA>

**你的核心使命：创建一份立体化、多维度的全景式多媒体分析报告，不少于一万字**

**多媒体分析报告的创新架构：**

```markdown
# 【全景解析】[主题]多维度融合分析报告

## 全景概览
### 多维信息摘要
- 文字信息核心发现
- 视觉内容关键洞察
- 数据趋势重要指标
- 跨媒体关联分析

### 信息源分布图
- 网页文字内容：XX%
- 图片视觉信息：XX%
- 结构化数据：XX%
- AI分析洞察：XX%

## 一、[段落1标题]
### 1.1 多模态信息画像
| 信息类型 | 数量 | 主要内容 | 情感倾向 | 传播效果 | 影响力指数 |
|----------|------|----------|----------|----------|------------|

### 1.2 视觉内容深度解析
### 1.3 文字与视觉的融合分析
### 1.4 数据与内容的交叉验证

## 跨媒体综合分析
### 信息一致性评估
### 多维度影响力对比
### 融合效应分析

## 多维洞察与预测
## 多媒体数据附录
```

**多媒体报告特色格式化要求：**

1. **多维信息整合**：
   - 创建跨媒体对比表格
   - 用综合评分体系量化分析
   - 展现不同信息源的互补性

2. **立体化叙述**：
   - 从多个感官维度描述内容
   - 结合文字、图像、数据讲述完整故事

3. **创新分析视角**：
   - 信息传播效果的跨媒体对比
   - 视觉与文字的情感一致性分析
   - 多媒体组合的协同效应评估

4. **专业多媒体术语**：
   - 使用视觉传播、多媒体融合等专业词汇
   - 体现对不同媒体形式特点的深度理解

**质量控制标准：**
- **信息覆盖度**：充分利用文字、图像、数据等各类信息
- **分析立体度**：从多个维度和角度进行综合分析
- **融合深度**：实现不同信息类型的深度融合
- **创新价值**：提供传统单一媒体分析无法实现的洞察

**最终输出**：一份融合多种媒体形式、具有立体化视角、创新分析方法的全景式多媒体分析报告，不少于一万字，为读者提供前所未有的全方位信息体验。"#,
        input_schema = INPUT_SCHEMA_REPORT_FORMATTING
    )
}

// =============================================================================
// ノード実装
// =============================================================================

/// LLM 出力から検索クエリと推論を抽出する共通処理
fn process_search_output(output: &str, default_query: &str, default_reasoning: &str) -> Value {
    let cleaned = remove_reasoning_from_output(output);
    let cleaned = clean_json_tags(&cleaned);

    let result: Value = match serde_json::from_str(&cleaned) {
        Ok(v) => v,
        Err(e) => {
            error!("JSON 解析失敗: {}", e);
            let extracted = extract_clean_response(&cleaned);
            if extracted.get("error").is_some() {
                if let Some(fixed) = fix_incomplete_json(&cleaned) {
                    match serde_json::from_str(&fixed) {
                        Ok(v) => v,
                        Err(_) => {
                            return serde_json::json!({
                                "search_query": default_query,
                                "search_tool": "comprehensive_search",
                                "reasoning": default_reasoning,
                            });
                        }
                    }
                } else {
                    return serde_json::json!({
                        "search_query": default_query,
                        "search_tool": "comprehensive_search",
                        "reasoning": default_reasoning,
                    });
                }
            } else {
                extracted
            }
        }
    };

    let search_query = result
        .get("search_query")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    if search_query.is_empty() {
        return serde_json::json!({
            "search_query": default_query,
            "search_tool": "comprehensive_search",
            "reasoning": default_reasoning,
        });
    }

    let mut output_obj = serde_json::json!({
        "search_query": search_query,
        "reasoning": result.get("reasoning").and_then(|v| v.as_str()).unwrap_or(""),
    });

    if let Some(tool) = result.get("search_tool").and_then(|v| v.as_str()) {
        output_obj["search_tool"] = Value::String(tool.to_string());
    }

    output_obj
}

/// LLM 出力からサマリーコンテンツを抽出する共通処理
fn extract_summary_content(output: &str, key: &str, fallback_msg: &str) -> String {
    let cleaned = remove_reasoning_from_output(output);
    let cleaned = clean_json_tags(&cleaned);

    match serde_json::from_str::<Value>(&cleaned) {
        Ok(result) => {
            if let Some(content) = result.get(key).and_then(|v| v.as_str()) {
                if !content.is_empty() {
                    return content.to_string();
                }
            }
            cleaned
        }
        Err(e) => {
            error!("JSON 解析失敗: {}", e);
            if let Some(fixed) = fix_incomplete_json(&cleaned) {
                match serde_json::from_str::<Value>(&fixed) {
                    Ok(result) => {
                        if let Some(content) = result.get(key).and_then(|v| v.as_str()) {
                            if !content.is_empty() {
                                return content.to_string();
                            }
                        }
                        cleaned
                    }
                    Err(_) => {
                        if cleaned.is_empty() {
                            fallback_msg.to_string()
                        } else {
                            cleaned
                        }
                    }
                }
            } else if cleaned.is_empty() {
                fallback_msg.to_string()
            } else {
                cleaned
            }
        }
    }
}

/// HOST 発言を読み取り、メッセージに追加するヘルパー
fn prepare_message_with_host_speech(data: &Value) -> String {
    let mut message = serde_json::to_string(data).unwrap_or_default();

    match get_latest_host_speech("logs") {
        Some(speech) if !speech.is_empty() => {
            info!("HOST 発言を読み取り済み、長さ: {} 文字", speech.len());
            let formatted = format_host_speech_for_prompt(&speech);
            message = format!("{}\n{}", formatted, message);
        }
        _ => {}
    }

    message
}

// --- レポート構造ノード ---

struct ReportStructureNode {
    llm_client: LLMClient,
    query: String,
}

impl ReportStructureNode {
    fn new(llm_client: LLMClient, query: &str) -> Self {
        Self {
            llm_client,
            query: query.to_string(),
        }
    }

    fn generate_default_structure() -> Vec<Value> {
        vec![
            serde_json::json!({ "title": "研究概述", "content": "对查询主题进行总体概述和分析" }),
            serde_json::json!({ "title": "深度分析", "content": "深入分析查询主题的各个方面" }),
        ]
    }
}

#[async_trait]
impl Node for ReportStructureNode {
    fn name(&self) -> &str {
        "ReportStructureNode"
    }

    async fn run(&self, _input_data: &Value) -> Result<Value> {
        let system_prompt = system_prompt_report_structure();
        let response = self
            .llm_client
            .stream_invoke_to_string(&system_prompt, &self.query, &InvokeOptions::default())
            .await?;

        let cleaned = remove_reasoning_from_output(&response);
        let cleaned = clean_json_tags(&cleaned);

        let report_structure: Vec<Value> = match serde_json::from_str(&cleaned) {
            Ok(v) => v,
            Err(e) => {
                error!("JSON 解析失敗: {}", e);
                let extracted = extract_clean_response(&cleaned);
                if extracted.get("error").is_some() {
                    if let Some(fixed) = fix_incomplete_json(&cleaned) {
                        match serde_json::from_str(&fixed) {
                            Ok(v) => v,
                            Err(_) => return Ok(Value::Array(Self::generate_default_structure())),
                        }
                    } else {
                        return Ok(Value::Array(Self::generate_default_structure()));
                    }
                } else if let Some(arr) = extracted.as_array() {
                    arr.clone()
                } else {
                    vec![extracted]
                }
            }
        };

        let mut validated: Vec<Value> = Vec::new();
        for (i, paragraph) in report_structure.iter().enumerate() {
            if let Some(obj) = paragraph.as_object() {
                let title = obj.get("title").and_then(|v| v.as_str()).unwrap_or("");
                let content = obj.get("content").and_then(|v| v.as_str()).unwrap_or("");
                if title.is_empty() || content.is_empty() {
                    warn!("段落 {} にタイトルまたはコンテンツがありません、スキップ", i + 1);
                    continue;
                }
                validated.push(serde_json::json!({ "title": title, "content": content }));
            }
        }

        if validated.is_empty() {
            return Ok(Value::Array(Self::generate_default_structure()));
        }

        Ok(Value::Array(validated))
    }
}

// --- 初回検索ノード ---

struct FirstSearchNode {
    llm_client: LLMClient,
}

impl FirstSearchNode {
    fn new(llm_client: LLMClient) -> Self {
        Self { llm_client }
    }
}

#[async_trait]
impl Node for FirstSearchNode {
    fn name(&self) -> &str {
        "FirstSearchNode"
    }

    async fn run(&self, input_data: &Value) -> Result<Value> {
        let message = serde_json::to_string(input_data)?;
        let system_prompt = system_prompt_first_search();
        let response = self
            .llm_client
            .stream_invoke_to_string(&system_prompt, &message, &InvokeOptions::default())
            .await?;

        Ok(process_search_output(
            &response,
            "相关主题研究",
            "由于解析失败，使用默认搜索查询",
        ))
    }
}

// --- リフレクションノード ---

struct ReflectionNode {
    llm_client: LLMClient,
}

impl ReflectionNode {
    fn new(llm_client: LLMClient) -> Self {
        Self { llm_client }
    }
}

#[async_trait]
impl Node for ReflectionNode {
    fn name(&self) -> &str {
        "ReflectionNode"
    }

    async fn run(&self, input_data: &Value) -> Result<Value> {
        let message = serde_json::to_string(input_data)?;
        let system_prompt = system_prompt_reflection();
        let response = self
            .llm_client
            .stream_invoke_to_string(&system_prompt, &message, &InvokeOptions::default())
            .await?;

        Ok(process_search_output(
            &response,
            "深度研究补充信息",
            "由于解析失败，使用默认反思搜索查询",
        ))
    }
}

// --- 初回サマリーノード ---

struct FirstSummaryNode {
    llm_client: LLMClient,
}

impl FirstSummaryNode {
    fn new(llm_client: LLMClient) -> Self {
        Self { llm_client }
    }
}

#[async_trait]
impl Node for FirstSummaryNode {
    fn name(&self) -> &str {
        "FirstSummaryNode"
    }

    async fn run(&self, input_data: &Value) -> Result<Value> {
        let message = prepare_message_with_host_speech(input_data);
        let system_prompt = system_prompt_first_summary();
        let response = self
            .llm_client
            .stream_invoke_to_string(&system_prompt, &message, &InvokeOptions::default())
            .await?;

        let content = extract_summary_content(&response, "paragraph_latest_state", "段落サマリー生成失敗");
        Ok(Value::String(content))
    }
}

// --- リフレクションサマリーノード ---

struct ReflectionSummaryNode {
    llm_client: LLMClient,
}

impl ReflectionSummaryNode {
    fn new(llm_client: LLMClient) -> Self {
        Self { llm_client }
    }
}

#[async_trait]
impl Node for ReflectionSummaryNode {
    fn name(&self) -> &str {
        "ReflectionSummaryNode"
    }

    async fn run(&self, input_data: &Value) -> Result<Value> {
        let message = prepare_message_with_host_speech(input_data);
        let system_prompt = system_prompt_reflection_summary();
        let response = self
            .llm_client
            .stream_invoke_to_string(&system_prompt, &message, &InvokeOptions::default())
            .await?;

        let content = extract_summary_content(
            &response,
            "updated_paragraph_latest_state",
            "リフレクションサマリー生成失敗",
        );
        Ok(Value::String(content))
    }
}

// --- レポートフォーマットノード ---

struct ReportFormattingNode {
    llm_client: LLMClient,
}

impl ReportFormattingNode {
    fn new(llm_client: LLMClient) -> Self {
        Self { llm_client }
    }

    fn format_report_manually(&self, paragraphs_data: &[Value], report_title: &str) -> String {
        let title = if report_title.is_empty() {
            "深度研究报告"
        } else {
            report_title
        };

        let mut lines = vec![
            format!("# {}", title),
            String::new(),
            "---".to_string(),
            String::new(),
        ];

        for (i, paragraph) in paragraphs_data.iter().enumerate() {
            let default_title = format!("段落 {}", i + 1);
            let p_title = paragraph
                .get("title")
                .and_then(|v| v.as_str())
                .unwrap_or(&default_title);
            let content = paragraph
                .get("paragraph_latest_state")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            if !content.is_empty() {
                lines.push(format!("## {}", p_title));
                lines.push(String::new());
                lines.push(content.to_string());
                lines.push(String::new());
                lines.push("---".to_string());
                lines.push(String::new());
            }
        }

        if paragraphs_data.len() > 1 {
            lines.push("## 结论".to_string());
            lines.push(String::new());
            lines.push(
                "本报告通过深度搜索和研究，对相关主题进行了全面分析。\
                 以上各个方面的内容为理解该主题提供了重要参考。"
                    .to_string(),
            );
            lines.push(String::new());
        }

        lines.join("\n")
    }
}

#[async_trait]
impl Node for ReportFormattingNode {
    fn name(&self) -> &str {
        "ReportFormattingNode"
    }

    async fn run(&self, input_data: &Value) -> Result<Value> {
        let message = serde_json::to_string(input_data)?;
        let system_prompt = system_prompt_report_formatting();
        let response = self
            .llm_client
            .stream_invoke_to_string(&system_prompt, &message, &InvokeOptions::default())
            .await?;

        let cleaned = remove_reasoning_from_output(&response);
        let cleaned = clean_markdown_tags(&cleaned);

        let result = if cleaned.trim().is_empty() {
            "# 报告生成失败\n\n无法生成有效的报告内容。".to_string()
        } else if !cleaned.trim().starts_with('#') {
            format!("# 深度研究报告\n\n{}", cleaned.trim())
        } else {
            cleaned.trim().to_string()
        };

        Ok(Value::String(result))
    }
}

// =============================================================================
// 検索クライアントの抽象化
// =============================================================================

/// 検索クライアントの種類
enum SearchClient {
    Bocha(BochaMultimodalSearch),
    Anspire(AnspireAISearch),
}

impl SearchClient {
    /// 指定の検索ツールを実行し、ウェブページ結果を返す
    async fn execute_search_tool(&self, tool_name: &str, query: &str) -> Vec<WebpageResult> {
        match self {
            SearchClient::Bocha(client) => {
                let response = match tool_name {
                    "comprehensive_search" => client.comprehensive_search(query, 10).await,
                    "web_search_only" => client.web_search_only(query, 15).await,
                    "search_for_structured_data" => client.search_for_structured_data(query).await,
                    "search_last_24_hours" => client.search_last_24_hours(query).await,
                    "search_last_week" => client.search_last_week(query).await,
                    _ => {
                        warn!("未知の検索ツール: {}、デフォルトの総合検索を使用", tool_name);
                        client.comprehensive_search(query, 10).await
                    }
                };
                response.webpages
            }
            SearchClient::Anspire(client) => {
                let response = match tool_name {
                    "comprehensive_search" => client.comprehensive_search(query, 10).await,
                    "search_last_24_hours" => client.search_last_24_hours(query, 10).await,
                    "search_last_week" => client.search_last_week(query, 10).await,
                    _ => {
                        warn!("未知の検索ツール: {}、デフォルトの総合検索を使用", tool_name);
                        client.comprehensive_search(query, 10).await
                    }
                };
                response.webpages
            }
        }
    }
}

// =============================================================================
// DeepSearchAgent メインクラス
// =============================================================================

/// Deep Search Agent メインクラス (MediaEngine)
pub struct DeepSearchAgent {
    config: Settings,
    llm_client: LLMClient,
    search_client: SearchClient,
    first_search_node: FirstSearchNode,
    reflection_node: ReflectionNode,
    first_summary_node: FirstSummaryNode,
    reflection_summary_node: ReflectionSummaryNode,
    report_formatting_node: ReportFormattingNode,
    pub state: State,
}

impl DeepSearchAgent {
    /// Deep Search Agent を初期化 (Bocha 使用)
    pub fn new(config: Option<Settings>) -> Result<Self> {
        let config = config.unwrap_or_else(Settings::load);

        let llm_client = LLMClient::new(
            &config.media_engine_api_key,
            &config.media_engine_model_name,
            config.media_engine_base_url.as_deref(),
        )?;

        let search_client = SearchClient::Bocha(BochaMultimodalSearch::new(
            &config.bocha_web_search_api_key,
            Some(&config.bocha_base_url),
        )?);

        let first_search_node = FirstSearchNode::new(llm_client.clone());
        let reflection_node = ReflectionNode::new(llm_client.clone());
        let first_summary_node = FirstSummaryNode::new(llm_client.clone());
        let reflection_summary_node = ReflectionSummaryNode::new(llm_client.clone());
        let report_formatting_node = ReportFormattingNode::new(llm_client.clone());

        fs::create_dir_all(&config.output_dir).ok();

        info!("Media Agent を初期化しました");
        info!("使用 LLM: {:?}", llm_client.get_model_info());
        info!("検索ツールセット: BochaMultimodalSearch (5 種類の多モーダル検索ツールをサポート)");

        Ok(Self {
            config,
            llm_client,
            search_client,
            first_search_node,
            reflection_node,
            first_summary_node,
            reflection_summary_node,
            report_formatting_node,
            state: State::default(),
        })
    }

    /// 深度研究を実行
    pub async fn research(&mut self, query: &str, save_report: bool) -> Result<String> {
        info!("\n{}", "=".repeat(60));
        info!("深度研究を開始: {}", query);
        info!("{}", "=".repeat(60));

        // Step 1: レポート構造を生成
        self.generate_report_structure(query).await?;

        // Step 2: 各段落を処理
        self.process_paragraphs().await?;

        // Step 3: 最終レポートを生成
        let final_report = self.generate_final_report().await?;

        // Step 4: レポートを保存
        if save_report {
            self.save_report(&final_report)?;
        }

        info!("\n{}", "=".repeat(60));
        info!("深度研究が完了しました！");
        info!("{}", "=".repeat(60));

        Ok(final_report)
    }

    async fn generate_report_structure(&mut self, query: &str) -> Result<()> {
        info!("\n[ステップ 1] レポート構造を生成中...");

        let report_structure_node = ReportStructureNode::new(self.llm_client.clone(), query);
        let report_structure = report_structure_node.run(&Value::Null).await?;

        self.state.query = query.to_string();
        if self.state.report_title.is_empty() {
            self.state.report_title = format!("关于'{}'的深度研究报告", query);
        }

        if let Some(paragraphs) = report_structure.as_array() {
            for p in paragraphs {
                let title = p.get("title").and_then(|v| v.as_str()).unwrap_or("");
                let content = p.get("content").and_then(|v| v.as_str()).unwrap_or("");
                self.state.add_paragraph(title, content);
            }
        }

        let mut message = format!(
            "レポート構造が生成されました、計 {} 個の段落:",
            self.state.paragraphs.len()
        );
        for (i, paragraph) in self.state.paragraphs.iter().enumerate() {
            message.push_str(&format!("\n  {}. {}", i + 1, paragraph.title));
        }
        info!("{}", message);

        Ok(())
    }

    async fn process_paragraphs(&mut self) -> Result<()> {
        let total_paragraphs = self.state.paragraphs.len();

        for i in 0..total_paragraphs {
            info!(
                "\n[ステップ 2.{}] 段落を処理中: {}",
                i + 1,
                self.state.paragraphs[i].title
            );
            info!("{}", "-".repeat(50));

            self.initial_search_and_summary(i).await?;
            self.reflection_loop(i).await?;
            self.state.paragraphs[i].research.mark_completed();

            let progress = (i + 1) as f64 / total_paragraphs as f64 * 100.0;
            info!("段落処理完了 ({:.1}%)", progress);
        }

        Ok(())
    }

    async fn initial_search_and_summary(&mut self, paragraph_index: usize) -> Result<()> {
        let title = self.state.paragraphs[paragraph_index].title.clone();
        let content = self.state.paragraphs[paragraph_index].content.clone();

        let search_input = serde_json::json!({
            "title": title,
            "content": content,
        });

        info!("  - 検索クエリを生成中...");
        let search_output = self.first_search_node.run(&search_input).await?;
        let search_query = search_output
            .get("search_query")
            .and_then(|v| v.as_str())
            .unwrap_or("相关主题研究")
            .to_string();
        let search_tool = search_output
            .get("search_tool")
            .and_then(|v| v.as_str())
            .unwrap_or("comprehensive_search")
            .to_string();
        let reasoning = search_output
            .get("reasoning")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        info!("  - 検索クエリ: {}", search_query);
        info!("  - 選択されたツール: {}", search_tool);
        info!("  - 推論: {}", reasoning);

        info!("  - ウェブ検索を実行中...");
        let webpages = self
            .search_client
            .execute_search_tool(&search_tool, &search_query)
            .await;

        let search_results = convert_webpages_to_values(&webpages, 10);

        if !search_results.is_empty() {
            info!("  - {} 件の検索結果を発見", search_results.len());
        } else {
            info!("  - 検索結果が見つかりません");
        }

        self.state.paragraphs[paragraph_index]
            .research
            .add_search_results(&search_query, &search_results);

        info!("  - 初回サマリーを生成中...");
        let formatted_results = format_search_results_for_prompt(
            &search_results
                .iter()
                .map(|v| {
                    let mut map = HashMap::new();
                    if let Some(obj) = v.as_object() {
                        for (k, v) in obj {
                            map.insert(k.clone(), v.clone());
                        }
                    }
                    map
                })
                .collect::<Vec<_>>(),
            self.config.search_content_max_length,
        );

        let summary_input = serde_json::json!({
            "title": title,
            "content": content,
            "search_query": search_query,
            "search_results": formatted_results,
        });

        let summary = self.first_summary_node.run(&summary_input).await?;
        let summary_str = summary.as_str().unwrap_or("");
        if paragraph_index < self.state.paragraphs.len() {
            self.state.paragraphs[paragraph_index].research.latest_summary = summary_str.to_string();
        }
        self.state.update_timestamp();

        info!("  - 初回サマリー完了");
        Ok(())
    }

    async fn reflection_loop(&mut self, paragraph_index: usize) -> Result<()> {
        for reflection_i in 0..self.config.max_reflections {
            info!(
                "  - リフレクション {}/{}...",
                reflection_i + 1,
                self.config.max_reflections
            );

            let title = self.state.paragraphs[paragraph_index].title.clone();
            let content = self.state.paragraphs[paragraph_index].content.clone();
            let latest_summary = self.state.paragraphs[paragraph_index]
                .research
                .latest_summary
                .clone();

            let reflection_input = serde_json::json!({
                "title": title,
                "content": content,
                "paragraph_latest_state": latest_summary,
            });

            let reflection_output = self.reflection_node.run(&reflection_input).await?;
            let search_query = reflection_output
                .get("search_query")
                .and_then(|v| v.as_str())
                .unwrap_or("深度研究补充信息")
                .to_string();
            let search_tool = reflection_output
                .get("search_tool")
                .and_then(|v| v.as_str())
                .unwrap_or("comprehensive_search")
                .to_string();
            let reasoning = reflection_output
                .get("reasoning")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            info!("    リフレクションクエリ: {}", search_query);
            info!("    選択されたツール: {}", search_tool);
            info!("    リフレクション推論: {}", reasoning);

            let webpages = self
                .search_client
                .execute_search_tool(&search_tool, &search_query)
                .await;

            let search_results = convert_webpages_to_values(&webpages, 10);

            if !search_results.is_empty() {
                info!("    {} 件のリフレクション検索結果を発見", search_results.len());
            } else {
                info!("    リフレクション検索結果が見つかりません");
            }

            self.state.paragraphs[paragraph_index]
                .research
                .add_search_results(&search_query, &search_results);

            let formatted_results = format_search_results_for_prompt(
                &search_results
                    .iter()
                    .map(|v| {
                        let mut map = HashMap::new();
                        if let Some(obj) = v.as_object() {
                            for (k, v) in obj {
                                map.insert(k.clone(), v.clone());
                            }
                        }
                        map
                    })
                    .collect::<Vec<_>>(),
                self.config.search_content_max_length,
            );

            let latest = self.state.paragraphs[paragraph_index]
                .research
                .latest_summary
                .clone();

            let reflection_summary_input = serde_json::json!({
                "title": title,
                "content": content,
                "search_query": search_query,
                "search_results": formatted_results,
                "paragraph_latest_state": latest,
            });

            let updated_summary = self
                .reflection_summary_node
                .run(&reflection_summary_input)
                .await?;
            let summary_str = updated_summary.as_str().unwrap_or("");
            if paragraph_index < self.state.paragraphs.len() {
                self.state.paragraphs[paragraph_index].research.latest_summary =
                    summary_str.to_string();
                self.state.paragraphs[paragraph_index]
                    .research
                    .increment_reflection();
            }
            self.state.update_timestamp();

            info!("    リフレクション {} 完了", reflection_i + 1);
        }

        Ok(())
    }

    async fn generate_final_report(&mut self) -> Result<String> {
        info!("\n[ステップ 3] 最終レポートを生成中...");

        let report_data: Vec<Value> = self
            .state
            .paragraphs
            .iter()
            .map(|p| {
                serde_json::json!({
                    "title": p.title,
                    "paragraph_latest_state": p.research.latest_summary,
                })
            })
            .collect();

        let report_data_val = Value::Array(report_data.clone());

        let final_report = match self.report_formatting_node.run(&report_data_val).await {
            Ok(val) => val.as_str().unwrap_or("").to_string(),
            Err(e) => {
                error!("LLM フォーマット失敗、バックアップメソッドを使用: {}", e);
                self.report_formatting_node
                    .format_report_manually(&report_data, &self.state.report_title)
            }
        };

        self.state.final_report = final_report.clone();
        self.state.mark_completed();

        info!("最終レポートの生成が完了しました");
        Ok(final_report)
    }

    fn save_report(&self, report_content: &str) -> Result<()> {
        let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();
        let query_safe: String = self
            .state
            .query
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == ' ' || *c == '-' || *c == '_')
            .collect::<String>()
            .trim()
            .replace(' ', "_");
        let query_safe: String = query_safe.chars().take(30).collect();

        let filename = format!("deep_search_report_{}_{}.md", query_safe, timestamp);
        let filepath = Path::new(&self.config.output_dir).join(&filename);

        fs::write(&filepath, report_content).context("レポートの保存に失敗")?;
        info!("レポートを保存しました: {}", filepath.display());

        if self.config.save_intermediate_states {
            let state_filename = format!("state_{}_{}.json", query_safe, timestamp);
            let state_filepath = Path::new(&self.config.output_dir).join(&state_filename);
            self.state
                .save_to_file(state_filepath.to_str().unwrap_or("state.json"))?;
            info!("状態を保存しました: {}", state_filepath.display());
        }

        Ok(())
    }

    /// 進捗サマリーを取得
    pub fn get_progress_summary(&self) -> Value {
        self.state.get_progress_summary()
    }

    /// ファイルから状態を読み込み
    pub fn load_state(&mut self, filepath: &str) -> Result<()> {
        self.state = State::load_from_file(filepath)?;
        info!("状態を {} から読み込みました", filepath);
        Ok(())
    }

    /// 状態をファイルに保存
    pub fn save_state(&self, filepath: &str) -> Result<()> {
        self.state.save_to_file(filepath)?;
        info!("状態を {} に保存しました", filepath);
        Ok(())
    }
}

// =============================================================================
// AnspireSearchAgent (DeepSearchAgent を拡張)
// =============================================================================

/// Anspire 検索エンジンを使用する Deep Search Agent
pub struct AnspireSearchAgent {
    inner: DeepSearchAgent,
}

impl AnspireSearchAgent {
    /// Anspire 検索 Agent を初期化
    pub fn new(config: Option<Settings>) -> Result<Self> {
        let config = config.unwrap_or_else(Settings::load);

        let llm_client = LLMClient::new(
            &config.media_engine_api_key,
            &config.media_engine_model_name,
            config.media_engine_base_url.as_deref(),
        )?;

        let search_client = SearchClient::Anspire(AnspireAISearch::new(
            &config.anspire_api_key,
            Some(&config.anspire_base_url),
        )?);

        let first_search_node = FirstSearchNode::new(llm_client.clone());
        let reflection_node = ReflectionNode::new(llm_client.clone());
        let first_summary_node = FirstSummaryNode::new(llm_client.clone());
        let reflection_summary_node = ReflectionSummaryNode::new(llm_client.clone());
        let report_formatting_node = ReportFormattingNode::new(llm_client.clone());

        fs::create_dir_all(&config.output_dir).ok();

        info!("Media Agent を初期化しました");
        info!("使用 LLM: {:?}", llm_client.get_model_info());
        info!("検索ツールセット: AnspireSearch");

        Ok(Self {
            inner: DeepSearchAgent {
                config,
                llm_client,
                search_client,
                first_search_node,
                reflection_node,
                first_summary_node,
                reflection_summary_node,
                report_formatting_node,
                state: State::default(),
            },
        })
    }

    /// 深度研究を実行
    pub async fn research(&mut self, query: &str, save_report: bool) -> Result<String> {
        self.inner.research(query, save_report).await
    }

    /// 進捗サマリーを取得
    pub fn get_progress_summary(&self) -> Value {
        self.inner.get_progress_summary()
    }
}

// =============================================================================
// ファクトリ関数
// =============================================================================

/// 設定に基づいて適切な Agent インスタンスを作成
pub fn create_agent() -> Result<DeepSearchAgent> {
    let settings = Settings::load();
    if settings.search_tool_type == "AnspireAPI" {
        // Anspire の場合も DeepSearchAgent を返すが内部的に AnspireAISearch を使用
        let llm_client = LLMClient::new(
            &settings.media_engine_api_key,
            &settings.media_engine_model_name,
            settings.media_engine_base_url.as_deref(),
        )?;

        let search_client = SearchClient::Anspire(AnspireAISearch::new(
            &settings.anspire_api_key,
            Some(&settings.anspire_base_url),
        )?);

        let first_search_node = FirstSearchNode::new(llm_client.clone());
        let reflection_node = ReflectionNode::new(llm_client.clone());
        let first_summary_node = FirstSummaryNode::new(llm_client.clone());
        let reflection_summary_node = ReflectionSummaryNode::new(llm_client.clone());
        let report_formatting_node = ReportFormattingNode::new(llm_client.clone());

        fs::create_dir_all(&settings.output_dir).ok();

        info!("Media Agent を初期化しました (Anspire)");

        Ok(DeepSearchAgent {
            config: settings,
            llm_client,
            search_client,
            first_search_node,
            reflection_node,
            first_summary_node,
            reflection_summary_node,
            report_formatting_node,
            state: State::default(),
        })
    } else {
        DeepSearchAgent::new(Some(settings))
    }
}
