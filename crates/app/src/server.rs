//! BettaFish HTTP サーバー
//!
//! Python の app.py に対応する HTTP ルートハンドラー定義。
//! 純 Rust 実装で、axum/actix なしでルート構造とハンドラーを定義。
//! 将来的に Web フレームワーク統合時にリクエスト/レスポンス型を差し替える。

#![allow(dead_code)]

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tracing::{error, info, warn};

use bettafish_config::Settings;

// =============================================================================
// サーバー設定
// =============================================================================

/// サーバー設定
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// ホストアドレス
    pub host: String,
    /// ポート番号
    pub port: u16,
    /// デバッグモード
    pub debug: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 5000,
            debug: false,
        }
    }
}

impl ServerConfig {
    /// Settings から生成
    pub fn from_settings(settings: &Settings) -> Self {
        Self {
            host: settings.host.clone(),
            port: settings.port,
            debug: false,
        }
    }
}

// =============================================================================
// HTTP メソッド & ルート定義
// =============================================================================

/// HTTP メソッド
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Delete,
}

impl std::fmt::Display for HttpMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Get => write!(f, "GET"),
            Self::Post => write!(f, "POST"),
            Self::Put => write!(f, "PUT"),
            Self::Delete => write!(f, "DELETE"),
        }
    }
}

/// ルート定義
#[derive(Debug, Clone)]
pub struct Route {
    /// HTTP メソッド
    pub method: HttpMethod,
    /// パスパターン
    pub path: String,
    /// ハンドラー名
    pub handler_name: String,
    /// 説明
    pub description: String,
}

/// 全ルート定義を取得
pub fn get_routes() -> Vec<Route> {
    vec![
        Route {
            method: HttpMethod::Get,
            path: "/".to_string(),
            handler_name: "index".to_string(),
            description: "インデックスページ情報".to_string(),
        },
        Route {
            method: HttpMethod::Post,
            path: "/api/research".to_string(),
            handler_name: "start_research".to_string(),
            description: "研究タスクを開始".to_string(),
        },
        Route {
            method: HttpMethod::Get,
            path: "/api/research/status/:task_id".to_string(),
            handler_name: "get_research_status".to_string(),
            description: "研究タスクのステータスを取得".to_string(),
        },
        Route {
            method: HttpMethod::Post,
            path: "/api/config".to_string(),
            handler_name: "update_config".to_string(),
            description: "設定を更新".to_string(),
        },
        Route {
            method: HttpMethod::Get,
            path: "/api/config".to_string(),
            handler_name: "read_config".to_string(),
            description: "設定を読み取り".to_string(),
        },
        Route {
            method: HttpMethod::Post,
            path: "/api/report/generate".to_string(),
            handler_name: "generate_report".to_string(),
            description: "レポートを生成".to_string(),
        },
        Route {
            method: HttpMethod::Get,
            path: "/api/report/status/:task_id".to_string(),
            handler_name: "get_report_status".to_string(),
            description: "レポート生成ステータスを取得".to_string(),
        },
        Route {
            method: HttpMethod::Post,
            path: "/api/forum/start".to_string(),
            handler_name: "start_forum".to_string(),
            description: "フォーラム監視を開始".to_string(),
        },
        Route {
            method: HttpMethod::Post,
            path: "/api/forum/stop".to_string(),
            handler_name: "stop_forum".to_string(),
            description: "フォーラム監視を停止".to_string(),
        },
        Route {
            method: HttpMethod::Get,
            path: "/api/forum/log".to_string(),
            handler_name: "get_forum_log".to_string(),
            description: "フォーラムログを取得".to_string(),
        },
    ]
}

// =============================================================================
// リクエスト/レスポンス型
// =============================================================================

/// JSON リクエストボディ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRequest {
    pub body: Value,
}

/// JSON レスポンス
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonResponse {
    pub status: u16,
    pub body: Value,
}

impl JsonResponse {
    /// 成功レスポンス
    pub fn ok(body: Value) -> Self {
        Self { status: 200, body }
    }

    /// エラーレスポンス
    pub fn error(status: u16, message: &str) -> Self {
        Self {
            status,
            body: serde_json::json!({
                "error": message,
            }),
        }
    }

    /// 作成レスポンス
    pub fn created(body: Value) -> Self {
        Self { status: 201, body }
    }
}

// =============================================================================
// タスク管理
// =============================================================================

/// タスクステータス
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskStatus {
    Pending,
    Running,
    Completed,
    Failed,
}

impl std::fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::Running => write!(f, "running"),
            Self::Completed => write!(f, "completed"),
            Self::Failed => write!(f, "failed"),
        }
    }
}

/// タスク情報
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskInfo {
    pub task_id: String,
    pub status: TaskStatus,
    pub created_at: String,
    pub updated_at: String,
    pub result: Option<Value>,
    pub error: Option<String>,
}

/// タスクストア
pub struct TaskStore {
    tasks: Arc<Mutex<HashMap<String, TaskInfo>>>,
    next_id: Arc<Mutex<u64>>,
}

impl TaskStore {
    /// 新しいタスクストアを作成
    pub fn new() -> Self {
        Self {
            tasks: Arc::new(Mutex::new(HashMap::new())),
            next_id: Arc::new(Mutex::new(1)),
        }
    }

    /// 新しいタスクを作成
    pub fn create_task(&self) -> String {
        let mut id_guard = self.next_id.lock().unwrap();
        let task_id = format!("task_{}", *id_guard);
        *id_guard += 1;

        let now = chrono::Local::now().to_rfc3339();
        let task = TaskInfo {
            task_id: task_id.clone(),
            status: TaskStatus::Pending,
            created_at: now.clone(),
            updated_at: now,
            result: None,
            error: None,
        };

        let mut tasks = self.tasks.lock().unwrap();
        tasks.insert(task_id.clone(), task);

        task_id
    }

    /// タスクステータスを更新
    pub fn update_task(&self, task_id: &str, status: TaskStatus, result: Option<Value>) {
        let mut tasks = self.tasks.lock().unwrap();
        if let Some(task) = tasks.get_mut(task_id) {
            task.status = status;
            task.updated_at = chrono::Local::now().to_rfc3339();
            task.result = result;
        }
    }

    /// タスクをエラー状態にする
    pub fn fail_task(&self, task_id: &str, error: &str) {
        let mut tasks = self.tasks.lock().unwrap();
        if let Some(task) = tasks.get_mut(task_id) {
            task.status = TaskStatus::Failed;
            task.updated_at = chrono::Local::now().to_rfc3339();
            task.error = Some(error.to_string());
        }
    }

    /// タスク情報を取得
    pub fn get_task(&self, task_id: &str) -> Option<TaskInfo> {
        let tasks = self.tasks.lock().unwrap();
        tasks.get(task_id).cloned()
    }
}

impl Default for TaskStore {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// RequestHandler トレイト
// =============================================================================

/// リクエストハンドラートレイト
#[async_trait::async_trait]
pub trait RequestHandler: Send + Sync {
    /// リクエストを処理
    async fn handle(&self, request: &JsonRequest) -> Result<JsonResponse>;
}

// =============================================================================
// ハンドラー関数
// =============================================================================

/// GET / - インデックスページ情報
pub async fn handle_index(settings: &Settings) -> Result<JsonResponse> {
    info!("GET / - インデックスページ");
    Ok(JsonResponse::ok(serde_json::json!({
        "name": "BettaFish",
        "version": "1.0.0",
        "description": "世論分析プラットフォーム (Rust)",
        "endpoints": {
            "research": "/api/research",
            "config": "/api/config",
            "report": "/api/report",
            "forum": "/api/forum",
        },
        "config": {
            "host": settings.host,
            "port": settings.port,
            "query_engine_model": settings.query_engine_model_name,
            "insight_engine_model": settings.insight_engine_model_name,
            "media_engine_model": settings.media_engine_model_name,
        },
    })))
}

/// POST /api/research - 研究タスクを開始
pub async fn handle_start_research(
    body: &Value,
    task_store: &TaskStore,
    _settings: &Settings,
) -> Result<JsonResponse> {
    let query = body
        .get("query")
        .and_then(|v| v.as_str())
        .unwrap_or("人工智能最新发展趋势分析");

    let engine = body
        .get("engine")
        .and_then(|v| v.as_str())
        .unwrap_or("query");

    info!(
        "POST /api/research - クエリ: '{}', エンジン: '{}'",
        query, engine
    );

    let task_id = task_store.create_task();
    task_store.update_task(&task_id, TaskStatus::Pending, None);

    info!("研究タスクを作成しました: {}", task_id);

    Ok(JsonResponse::created(serde_json::json!({
        "task_id": task_id,
        "status": "pending",
        "query": query,
        "engine": engine,
        "message": "研究タスクがキューに追加されました",
    })))
}

/// GET /api/research/status/:task_id - 研究タスクのステータスを取得
pub async fn handle_get_research_status(
    task_id: &str,
    task_store: &TaskStore,
) -> Result<JsonResponse> {
    info!("GET /api/research/status/{}", task_id);

    match task_store.get_task(task_id) {
        Some(task) => Ok(JsonResponse::ok(serde_json::json!({
            "task_id": task.task_id,
            "status": task.status.to_string(),
            "created_at": task.created_at,
            "updated_at": task.updated_at,
            "result": task.result,
            "error": task.error,
        }))),
        None => Ok(JsonResponse::error(404, &format!("タスク '{}' が見つかりません", task_id))),
    }
}

/// POST /api/config - 設定を更新
pub async fn handle_update_config(
    body: &Value,
    _settings: &Settings,
) -> Result<JsonResponse> {
    info!("POST /api/config - 設定更新リクエスト");

    // 現時点ではログのみ (実際の .env 書き換えは危険なため)
    let updated_keys: Vec<String> = body
        .as_object()
        .map(|obj| obj.keys().cloned().collect())
        .unwrap_or_default();

    warn!("設定更新は現在スタブ実装です。更新リクエストキー: {:?}", updated_keys);

    Ok(JsonResponse::ok(serde_json::json!({
        "message": "設定更新リクエストを受け付けました (スタブ)",
        "updated_keys": updated_keys,
    })))
}

/// GET /api/config - 設定を読み取り
pub async fn handle_read_config(settings: &Settings) -> Result<JsonResponse> {
    info!("GET /api/config - 設定読み取り");

    Ok(JsonResponse::ok(serde_json::json!({
        "host": settings.host,
        "port": settings.port,
        "db_dialect": settings.db_dialect,
        "db_host": settings.db_host,
        "db_port": settings.db_port,
        "db_name": settings.db_name,
        "query_engine_model": settings.query_engine_model_name,
        "insight_engine_model": settings.insight_engine_model_name,
        "media_engine_model": settings.media_engine_model_name,
        "report_engine_model": settings.report_engine_model_name,
        "mindspider_model": settings.mindspider_model_name,
        "search_tool_type": settings.search_tool_type,
        "max_reflections": settings.max_reflections,
        "max_paragraphs": settings.max_paragraphs,
        "output_dir": settings.output_dir,
    })))
}

/// POST /api/report/generate - レポートを生成
pub async fn handle_generate_report(
    body: &Value,
    task_store: &TaskStore,
    _settings: &Settings,
) -> Result<JsonResponse> {
    let report_type = body
        .get("report_type")
        .and_then(|v| v.as_str())
        .unwrap_or("full");
    let source_dir = body
        .get("source_dir")
        .and_then(|v| v.as_str())
        .unwrap_or("reports");

    info!(
        "POST /api/report/generate - type: '{}', source: '{}'",
        report_type, source_dir
    );

    let task_id = task_store.create_task();
    task_store.update_task(&task_id, TaskStatus::Pending, None);

    Ok(JsonResponse::created(serde_json::json!({
        "task_id": task_id,
        "status": "pending",
        "report_type": report_type,
        "source_dir": source_dir,
        "message": "レポート生成タスクがキューに追加されました",
    })))
}

/// GET /api/report/status/:task_id - レポート生成ステータスを取得
pub async fn handle_get_report_status(
    task_id: &str,
    task_store: &TaskStore,
) -> Result<JsonResponse> {
    info!("GET /api/report/status/{}", task_id);

    match task_store.get_task(task_id) {
        Some(task) => Ok(JsonResponse::ok(serde_json::json!({
            "task_id": task.task_id,
            "status": task.status.to_string(),
            "created_at": task.created_at,
            "updated_at": task.updated_at,
            "result": task.result,
            "error": task.error,
        }))),
        None => Ok(JsonResponse::error(404, &format!("タスク '{}' が見つかりません", task_id))),
    }
}

/// POST /api/forum/start - フォーラム監視を開始
pub async fn handle_start_forum(_settings: &Settings) -> Result<JsonResponse> {
    info!("POST /api/forum/start - フォーラム監視を開始");

    // ForumEngine の起動はバックグラウンドタスクとして実行
    Ok(JsonResponse::ok(serde_json::json!({
        "message": "フォーラム監視を開始しました",
        "status": "running",
    })))
}

/// POST /api/forum/stop - フォーラム監視を停止
pub async fn handle_stop_forum(_settings: &Settings) -> Result<JsonResponse> {
    info!("POST /api/forum/stop - フォーラム監視を停止");

    Ok(JsonResponse::ok(serde_json::json!({
        "message": "フォーラム監視を停止しました",
        "status": "stopped",
    })))
}

/// GET /api/forum/log - フォーラムログを取得
pub async fn handle_get_forum_log() -> Result<JsonResponse> {
    info!("GET /api/forum/log - フォーラムログを取得");

    let log_content = bettafish_common::forum_reader::get_all_host_speeches("logs");

    let log_entries: Vec<Value> = log_content
        .iter()
        .map(|speech| {
            serde_json::json!({
                "timestamp": speech.timestamp,
                "content": speech.content,
            })
        })
        .collect();

    Ok(JsonResponse::ok(serde_json::json!({
        "log_entries": log_entries,
        "total_entries": log_entries.len(),
    })))
}

// =============================================================================
// アプリケーションサーバー
// =============================================================================

/// アプリケーションサーバー
pub struct AppServer {
    config: ServerConfig,
    settings: Settings,
    task_store: TaskStore,
}

impl AppServer {
    /// 新しいサーバーを作成
    pub fn new(settings: Settings) -> Self {
        let config = ServerConfig::from_settings(&settings);
        Self {
            config,
            settings,
            task_store: TaskStore::new(),
        }
    }

    /// サーバーを起動 (メインループ)
    pub async fn start(&self) -> Result<()> {
        info!(
            "BettaFish サーバーを起動中: {}:{}",
            self.config.host, self.config.port
        );

        // ルート情報を表示
        let routes = get_routes();
        info!("登録済みルート:");
        for route in &routes {
            info!(
                "  {} {} -> {} ({})",
                route.method, route.path, route.handler_name, route.description
            );
        }

        info!(
            "サーバーが起動しました: http://{}:{}",
            self.config.host, self.config.port
        );
        info!("注意: HTTP リスナーは未実装です。将来 axum/actix-web で実装予定。");
        info!("Ctrl+C で停止してください。");

        // シグナルを待機 (実際の HTTP サーバーの代わり)
        tokio::signal::ctrl_c().await?;

        info!("サーバーを停止しました");
        Ok(())
    }

    /// ルーティングを処理 (テスト用)
    pub async fn route(&self, method: HttpMethod, path: &str, body: Option<Value>) -> JsonResponse {
        let body = body.unwrap_or(Value::Null);

        match (method, path) {
            (HttpMethod::Get, "/") => handle_index(&self.settings)
                .await
                .unwrap_or_else(|e| JsonResponse::error(500, &e.to_string())),

            (HttpMethod::Post, "/api/research") => {
                handle_start_research(&body, &self.task_store, &self.settings)
                    .await
                    .unwrap_or_else(|e| JsonResponse::error(500, &e.to_string()))
            }

            (HttpMethod::Get, p) if p.starts_with("/api/research/status/") => {
                let task_id = p.strip_prefix("/api/research/status/").unwrap_or("");
                handle_get_research_status(task_id, &self.task_store)
                    .await
                    .unwrap_or_else(|e| JsonResponse::error(500, &e.to_string()))
            }

            (HttpMethod::Post, "/api/config") => handle_update_config(&body, &self.settings)
                .await
                .unwrap_or_else(|e| JsonResponse::error(500, &e.to_string())),

            (HttpMethod::Get, "/api/config") => handle_read_config(&self.settings)
                .await
                .unwrap_or_else(|e| JsonResponse::error(500, &e.to_string())),

            (HttpMethod::Post, "/api/report/generate") => {
                handle_generate_report(&body, &self.task_store, &self.settings)
                    .await
                    .unwrap_or_else(|e| JsonResponse::error(500, &e.to_string()))
            }

            (HttpMethod::Get, p) if p.starts_with("/api/report/status/") => {
                let task_id = p.strip_prefix("/api/report/status/").unwrap_or("");
                handle_get_report_status(task_id, &self.task_store)
                    .await
                    .unwrap_or_else(|e| JsonResponse::error(500, &e.to_string()))
            }

            (HttpMethod::Post, "/api/forum/start") => handle_start_forum(&self.settings)
                .await
                .unwrap_or_else(|e| JsonResponse::error(500, &e.to_string())),

            (HttpMethod::Post, "/api/forum/stop") => handle_stop_forum(&self.settings)
                .await
                .unwrap_or_else(|e| JsonResponse::error(500, &e.to_string())),

            (HttpMethod::Get, "/api/forum/log") => handle_get_forum_log()
                .await
                .unwrap_or_else(|e| JsonResponse::error(500, &e.to_string())),

            _ => {
                error!("未知のルート: {} {}", method, path);
                JsonResponse::error(404, &format!("ルートが見つかりません: {} {}", method, path))
            }
        }
    }
}

// =============================================================================
// テスト
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_config_default() {
        let config = ServerConfig::default();
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 5000);
        assert!(!config.debug);
    }

    #[test]
    fn test_get_routes() {
        let routes = get_routes();
        assert_eq!(routes.len(), 10);

        let get_routes: Vec<_> = routes.iter().filter(|r| r.method == HttpMethod::Get).collect();
        let post_routes: Vec<_> = routes.iter().filter(|r| r.method == HttpMethod::Post).collect();

        assert_eq!(get_routes.len(), 5);
        assert_eq!(post_routes.len(), 5);
    }

    #[test]
    fn test_json_response_ok() {
        let resp = JsonResponse::ok(serde_json::json!({"test": true}));
        assert_eq!(resp.status, 200);
    }

    #[test]
    fn test_json_response_error() {
        let resp = JsonResponse::error(404, "Not found");
        assert_eq!(resp.status, 404);
    }

    #[test]
    fn test_task_store() {
        let store = TaskStore::new();
        let id = store.create_task();
        assert!(id.starts_with("task_"));

        let task = store.get_task(&id).unwrap();
        assert!(matches!(task.status, TaskStatus::Pending));

        store.update_task(
            &id,
            TaskStatus::Completed,
            Some(serde_json::json!({"result": "ok"})),
        );
        let task = store.get_task(&id).unwrap();
        assert!(matches!(task.status, TaskStatus::Completed));
        assert!(task.result.is_some());

        let id2 = store.create_task();
        store.fail_task(&id2, "something went wrong");
        let task2 = store.get_task(&id2).unwrap();
        assert!(matches!(task2.status, TaskStatus::Failed));
        assert_eq!(task2.error.as_deref(), Some("something went wrong"));

        assert!(store.get_task("nonexistent").is_none());
    }

    #[tokio::test]
    async fn test_handle_index() {
        let settings = Settings::default();
        let resp = handle_index(&settings).await.unwrap();
        assert_eq!(resp.status, 200);
        assert!(resp.body.get("name").is_some());
    }

    #[tokio::test]
    async fn test_handle_read_config() {
        let settings = Settings::default();
        let resp = handle_read_config(&settings).await.unwrap();
        assert_eq!(resp.status, 200);
        assert!(resp.body.get("host").is_some());
    }

    #[tokio::test]
    async fn test_handle_start_research() {
        let settings = Settings::default();
        let task_store = TaskStore::new();
        let body = serde_json::json!({"query": "测试查询", "engine": "query"});
        let resp = handle_start_research(&body, &task_store, &settings)
            .await
            .unwrap();
        assert_eq!(resp.status, 201);
        assert!(resp.body.get("task_id").is_some());
    }

    #[tokio::test]
    async fn test_app_server_routing() {
        let settings = Settings::default();
        let server = AppServer::new(settings);

        let resp = server.route(HttpMethod::Get, "/", None).await;
        assert_eq!(resp.status, 200);

        let resp = server.route(HttpMethod::Get, "/api/config", None).await;
        assert_eq!(resp.status, 200);

        let resp = server
            .route(HttpMethod::Get, "/nonexistent", None)
            .await;
        assert_eq!(resp.status, 404);
    }
}
