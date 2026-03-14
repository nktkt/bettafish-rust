//! BettaFish Rust メインアプリケーション
//!
//! Python の app.py に対応するエントリーポイント。
//! QueryEngine の完全実行をサポート。

use anyhow::Result;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

use bettafish_config::Settings;
use bettafish_query_engine::DeepSearchAgent;

#[tokio::main]
async fn main() -> Result<()> {
    // ログ初期化
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_target(false)
        .with_thread_ids(false)
        .init();

    info!("BettaFish Rust v{}", bettafish_query_engine::VERSION);
    info!("{}", "=".repeat(60));

    // 設定をロード
    let settings = Settings::load();
    info!("{}", settings.display_query_engine_config());

    // コマンドライン引数からクエリを取得
    let args: Vec<String> = std::env::args().collect();
    let query = if args.len() > 1 {
        args[1..].join(" ")
    } else {
        // デフォルトのデモクエリ
        info!("使用法: bettafish <クエリ>");
        info!("例: bettafish \"人工知能の最新動向\"");
        info!("");
        info!("デモモードで実行中...");
        "人工智能最新发展趋势分析".to_string()
    };

    info!("研究クエリ: {}", query);
    info!("{}", "=".repeat(60));

    // エージェントを作成して研究を実行
    match DeepSearchAgent::new(Some(settings)) {
        Ok(mut agent) => {
            match agent.research(&query, true).await {
                Ok(report) => {
                    info!("\n{}", "=".repeat(60));
                    info!("研究が正常に完了しました！");
                    info!("レポートの長さ: {} 文字", report.len());
                    info!("{}", "=".repeat(60));

                    // 進捗サマリーを表示
                    let progress = agent.get_progress_summary();
                    info!("進捗サマリー: {}", serde_json::to_string_pretty(&progress)?);
                }
                Err(e) => {
                    error!("研究中にエラーが発生: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Err(e) => {
            error!("エージェントの初期化に失敗: {}", e);
            error!("ヒント: .env ファイルに QUERY_ENGINE_API_KEY と TAVILY_API_KEY を設定してください");
            std::process::exit(1);
        }
    }

    Ok(())
}
