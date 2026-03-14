//! QueryEngine 検索ツール
//!
//! Python の tools/search.py の Rust 実装。
//! Tavily API を使った 6 種類の専用ニュース検索ツール。

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::info;

use bettafish_common::retry::{search_api_retry_config, with_graceful_retry};

/// 検索結果データ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub content: String,
    pub score: Option<f64>,
    pub raw_content: Option<String>,
    pub published_date: Option<String>,
}

/// 画像検索結果データ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageResult {
    pub url: String,
    pub description: Option<String>,
}

/// Tavily API の完全レスポンス
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TavilyResponse {
    pub query: String,
    pub answer: Option<String>,
    pub results: Vec<SearchResult>,
    pub images: Vec<ImageResult>,
    pub response_time: Option<f64>,
}

impl Default for TavilyResponse {
    fn default() -> Self {
        Self {
            query: "検索失敗".to_string(),
            answer: None,
            results: Vec::new(),
            images: Vec::new(),
            response_time: None,
        }
    }
}

/// Tavily API レスポンス（生）
#[derive(Debug, Deserialize)]
struct TavilyApiResponse {
    query: Option<String>,
    answer: Option<String>,
    results: Option<Vec<TavilyApiResult>>,
    images: Option<Vec<TavilyApiImage>>,
    response_time: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct TavilyApiResult {
    title: Option<String>,
    url: Option<String>,
    content: Option<String>,
    score: Option<f64>,
    raw_content: Option<String>,
    published_date: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TavilyApiImage {
    url: Option<String>,
    description: Option<String>,
}

/// ニュース検索ツールを含むクライアント
///
/// 各パブリックメソッドは AI Agent が個別に呼び出すツールとして設計。
pub struct TavilyNewsAgency {
    api_key: String,
    client: Client,
}

impl TavilyNewsAgency {
    /// クライアントを初期化
    pub fn new(api_key: &str) -> Result<Self> {
        if api_key.is_empty() {
            anyhow::bail!("Tavily API Key が見つかりません！TAVILY_API_KEY 環境変数を設定するか、初期化時に提供してください");
        }

        let client = Client::builder()
            .timeout(Duration::from_secs(240))
            .build()
            .context("HTTP クライアントの作成に失敗")?;

        Ok(Self {
            api_key: api_key.to_string(),
            client,
        })
    }

    /// 内部汎用検索実行
    async fn search_internal(
        &self,
        params: serde_json::Value,
    ) -> Result<TavilyResponse> {
        let mut body = params;
        body["api_key"] = serde_json::Value::String(self.api_key.clone());
        body["topic"] = serde_json::Value::String("general".to_string());

        let response = self
            .client
            .post("https://api.tavily.com/search")
            .json(&body)
            .send()
            .await
            .context("Tavily API リクエスト送信に失敗")?;

        let status = response.status();
        if !status.is_success() {
            let err_body = response.text().await.unwrap_or_default();
            anyhow::bail!("Tavily API エラー ({}): {}", status, err_body);
        }

        let api_resp: TavilyApiResponse = response
            .json()
            .await
            .context("Tavily レスポンスの解析に失敗")?;

        let results = api_resp
            .results
            .unwrap_or_default()
            .into_iter()
            .map(|r| SearchResult {
                title: r.title.unwrap_or_default(),
                url: r.url.unwrap_or_default(),
                content: r.content.unwrap_or_default(),
                score: r.score,
                raw_content: r.raw_content,
                published_date: r.published_date,
            })
            .collect();

        let images = api_resp
            .images
            .unwrap_or_default()
            .into_iter()
            .map(|i| ImageResult {
                url: i.url.unwrap_or_default(),
                description: i.description,
            })
            .collect();

        Ok(TavilyResponse {
            query: api_resp.query.unwrap_or_default(),
            answer: api_resp.answer,
            results,
            images,
            response_time: api_resp.response_time,
        })
    }

    // ===== Agent が使用できるツールメソッド =====

    /// 【ツール】基礎ニュース検索: 標準的で高速な汎用ニュース検索
    pub async fn basic_search_news(
        &self,
        query: &str,
        max_results: usize,
    ) -> TavilyResponse {
        info!("--- TOOL: 基礎ニュース検索 (query: {}) ---", query);
        let config = search_api_retry_config();
        let default = TavilyResponse::default();

        with_graceful_retry(&config, "basic_search_news", default, || {
            let params = serde_json::json!({
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": false,
            });
            async move { self.search_internal(params).await }
        })
        .await
    }

    /// 【ツール】深度ニュース分析: 最も包括的で深い検索
    pub async fn deep_search_news(&self, query: &str) -> TavilyResponse {
        info!("--- TOOL: 深度ニュース分析 (query: {}) ---", query);
        let config = search_api_retry_config();
        let default = TavilyResponse::default();

        with_graceful_retry(&config, "deep_search_news", default, || {
            let params = serde_json::json!({
                "query": query,
                "search_depth": "advanced",
                "max_results": 20,
                "include_answer": "advanced",
            });
            async move { self.search_internal(params).await }
        })
        .await
    }

    /// 【ツール】24時間内ニュース検索: 最新動態の追跡
    pub async fn search_news_last_24_hours(&self, query: &str) -> TavilyResponse {
        info!("--- TOOL: 24時間内ニュース検索 (query: {}) ---", query);
        let config = search_api_retry_config();
        let default = TavilyResponse::default();

        with_graceful_retry(&config, "search_news_last_24_hours", default, || {
            let params = serde_json::json!({
                "query": query,
                "time_range": "d",
                "max_results": 10,
            });
            async move { self.search_internal(params).await }
        })
        .await
    }

    /// 【ツール】今週ニュース検索: 過去一週間のニュース
    pub async fn search_news_last_week(&self, query: &str) -> TavilyResponse {
        info!("--- TOOL: 今週ニュース検索 (query: {}) ---", query);
        let config = search_api_retry_config();
        let default = TavilyResponse::default();

        with_graceful_retry(&config, "search_news_last_week", default, || {
            let params = serde_json::json!({
                "query": query,
                "time_range": "w",
                "max_results": 10,
            });
            async move { self.search_internal(params).await }
        })
        .await
    }

    /// 【ツール】ニュース画像検索: 関連画像の検索
    pub async fn search_images_for_news(&self, query: &str) -> TavilyResponse {
        info!("--- TOOL: ニュース画像検索 (query: {}) ---", query);
        let config = search_api_retry_config();
        let default = TavilyResponse::default();

        with_graceful_retry(&config, "search_images_for_news", default, || {
            let params = serde_json::json!({
                "query": query,
                "include_images": true,
                "include_image_descriptions": true,
                "max_results": 5,
            });
            async move { self.search_internal(params).await }
        })
        .await
    }

    /// 【ツール】日付範囲指定ニュース検索: 特定期間の歴史的検索
    pub async fn search_news_by_date(
        &self,
        query: &str,
        start_date: &str,
        end_date: &str,
    ) -> TavilyResponse {
        info!(
            "--- TOOL: 日付範囲指定ニュース検索 (query: {}, from: {}, to: {}) ---",
            query, start_date, end_date
        );
        let config = search_api_retry_config();
        let default = TavilyResponse::default();

        with_graceful_retry(&config, "search_news_by_date", default, || {
            let params = serde_json::json!({
                "query": query,
                "start_date": start_date,
                "end_date": end_date,
                "max_results": 15,
            });
            async move { self.search_internal(params).await }
        })
        .await
    }
}
