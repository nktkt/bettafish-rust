//! SingleEngineApp 相当 - 個別エンジンのスタンドアロン実行
//!
//! Python の SingleEngineApp に対応。
//! InsightEngine, MediaEngine, QueryEngine を個別に実行する機能を提供。

#![allow(dead_code)]

use anyhow::Result;
use tracing::{error, info};

use bettafish_config::Settings;

/// InsightEngine を単独実行
///
/// ローカル DB の世論データを分析する深度検索を実行。
pub async fn run_insight_engine(query: &str, settings: Settings) -> Result<String> {
    info!("{}", "=".repeat(60));
    info!("InsightEngine 単独実行モード");
    info!("クエリ: {}", query);
    info!("{}", "=".repeat(60));

    match bettafish_insight_engine::DeepSearchAgent::new(Some(settings)) {
        Ok(mut agent) => match agent.research(query, true).await {
            Ok(report) => {
                info!("InsightEngine 分析完了: {} 文字", report.len());
                let progress = agent.get_progress_summary();
                info!(
                    "進捗サマリー: {}",
                    serde_json::to_string_pretty(&progress)?
                );
                Ok(report)
            }
            Err(e) => {
                error!("InsightEngine 分析中にエラーが発生: {}", e);
                Err(e)
            }
        },
        Err(e) => {
            error!("InsightEngine 初期化失敗: {}", e);
            error!("ヒント: .env ファイルに INSIGHT_ENGINE_API_KEY を設定してください");
            Err(e)
        }
    }
}

/// MediaEngine を単独実行
///
/// Bocha/Anspire API を使ったマルチモーダル検索を実行。
pub async fn run_media_engine(query: &str, settings: Settings) -> Result<String> {
    info!("{}", "=".repeat(60));
    info!("MediaEngine 単独実行モード");
    info!("クエリ: {}", query);
    info!("{}", "=".repeat(60));

    match bettafish_media_engine::DeepSearchAgent::new(Some(settings)) {
        Ok(mut agent) => match agent.research(query, true).await {
            Ok(report) => {
                info!("MediaEngine 分析完了: {} 文字", report.len());
                let progress = agent.get_progress_summary();
                info!(
                    "進捗サマリー: {}",
                    serde_json::to_string_pretty(&progress)?
                );
                Ok(report)
            }
            Err(e) => {
                error!("MediaEngine 分析中にエラーが発生: {}", e);
                Err(e)
            }
        },
        Err(e) => {
            error!("MediaEngine 初期化失敗: {}", e);
            error!("ヒント: .env ファイルに MEDIA_ENGINE_API_KEY を設定してください");
            Err(e)
        }
    }
}

/// QueryEngine を単独実行
///
/// Tavily API を使った深度検索を実行。
pub async fn run_query_engine(query: &str, settings: Settings) -> Result<String> {
    info!("{}", "=".repeat(60));
    info!("QueryEngine 単独実行モード");
    info!("クエリ: {}", query);
    info!("{}", "=".repeat(60));

    match bettafish_query_engine::DeepSearchAgent::new(Some(settings)) {
        Ok(mut agent) => match agent.research(query, true).await {
            Ok(report) => {
                info!("QueryEngine 分析完了: {} 文字", report.len());
                let progress = agent.get_progress_summary();
                info!(
                    "進捗サマリー: {}",
                    serde_json::to_string_pretty(&progress)?
                );
                Ok(report)
            }
            Err(e) => {
                error!("QueryEngine 分析中にエラーが発生: {}", e);
                Err(e)
            }
        },
        Err(e) => {
            error!("QueryEngine 初期化失敗: {}", e);
            error!("ヒント: .env ファイルに QUERY_ENGINE_API_KEY を設定してください");
            Err(e)
        }
    }
}
