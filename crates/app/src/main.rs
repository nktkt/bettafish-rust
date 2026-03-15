//! BettaFish Rust メインアプリケーション
//!
//! Python の app.py に対応する包括的 CLI エントリーポイント。
//! QueryEngine, InsightEngine, MediaEngine, ForumEngine, ReportEngine,
//! MindSpider の各コンポーネントを統合管理する。

#![allow(dead_code)]

use anyhow::Result;
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

use bettafish_config::Settings;
use bettafish_mindspider::{MindSpider, SUPPORTED_PLATFORMS};
use bettafish_query_engine::DeepSearchAgent;

mod server;
mod single_engine;

#[cfg(test)]
mod tests;

/// バージョン情報
const VERSION: &str = "1.0.0";

/// CLI コマンド
#[derive(Debug)]
enum Command {
    /// 設定を表示
    ShowConfig,
    /// QueryEngine でクエリを実行
    Query(String),
    /// 完全パイプラインを実行 (MindSpider → engines → Report)
    Pipeline(PipelineArgs),
    /// MindSpider のみ実行
    MindSpiderCmd(MindSpiderArgs),
    /// プロジェクトステータス表示
    Status,
    /// ヘルプを表示
    Help,
    /// HTTP サーバーを起動
    Server,
    /// InsightEngine でクエリを実行
    Insight(String),
    /// MediaEngine でクエリを実行
    Media(String),
    /// PDF エクスポート
    ExportPdf(ExportArgs),
    /// HTML 再生成
    RegenerateHtml(ExportArgs),
    /// Markdown 再生成
    RegenerateMd(ExportArgs),
    /// レポートのみ生成
    ReportOnly(ReportOnlyArgs),
}

/// パイプライン引数
#[derive(Debug)]
struct PipelineArgs {
    query: String,
    skip_crawl: bool,
    platforms: Option<Vec<String>>,
}

/// MindSpider 引数
#[derive(Debug)]
struct MindSpiderArgs {
    /// サブコマンド: broad-topic, deep-sentiment, complete, setup, init-db, status
    subcommand: String,
    date: Option<String>,
    platforms: Option<Vec<String>>,
    keywords_count: usize,
    max_keywords: usize,
    max_notes: usize,
    test_mode: bool,
}

/// エクスポート引数
#[derive(Debug)]
struct ExportArgs {
    /// 入力ファイルパス
    input_path: String,
    /// 出力ファイルパス (任意)
    output_path: Option<String>,
}

/// レポートのみ生成引数
#[derive(Debug)]
struct ReportOnlyArgs {
    /// レポートソースディレクトリ
    source_dir: String,
    /// 出力ディレクトリ
    output_dir: Option<String>,
}

/// コマンドライン引数を解析
fn parse_args() -> Command {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        return Command::Help;
    }

    match args[1].as_str() {
        "--help" | "-h" | "help" => Command::Help,
        "--version" | "-v" => {
            println!("BettaFish Rust v{}", VERSION);
            std::process::exit(0);
        }
        "--config" | "config" => Command::ShowConfig,
        "--status" | "status" => Command::Status,

        // HTTP サーバーモード
        "server" | "serve" => Command::Server,

        // QueryEngine モード
        "query" | "--query" => {
            let query = if args.len() > 2 {
                args[2..].join(" ")
            } else {
                "人工智能最新发展趋势分析".to_string()
            };
            Command::Query(query)
        }

        // InsightEngine モード
        "insight" => {
            let query = if args.len() > 2 {
                args[2..].join(" ")
            } else {
                "人工智能最新发展趋势分析".to_string()
            };
            Command::Insight(query)
        }

        // MediaEngine モード
        "media" => {
            let query = if args.len() > 2 {
                args[2..].join(" ")
            } else {
                "人工智能最新发展趋势分析".to_string()
            };
            Command::Media(query)
        }

        // PDF エクスポート
        "export-pdf" => {
            let input_path = if args.len() > 2 {
                args[2].clone()
            } else {
                "reports".to_string()
            };
            let output_path = if args.len() > 3 {
                Some(args[3].clone())
            } else {
                None
            };
            Command::ExportPdf(ExportArgs {
                input_path,
                output_path,
            })
        }

        // HTML 再生成
        "regenerate-html" => {
            let input_path = if args.len() > 2 {
                args[2].clone()
            } else {
                "reports".to_string()
            };
            let output_path = if args.len() > 3 {
                Some(args[3].clone())
            } else {
                None
            };
            Command::RegenerateHtml(ExportArgs {
                input_path,
                output_path,
            })
        }

        // Markdown 再生成
        "regenerate-md" => {
            let input_path = if args.len() > 2 {
                args[2].clone()
            } else {
                "reports".to_string()
            };
            let output_path = if args.len() > 3 {
                Some(args[3].clone())
            } else {
                None
            };
            Command::RegenerateMd(ExportArgs {
                input_path,
                output_path,
            })
        }

        // レポートのみ生成
        "report-only" => {
            let source_dir = if args.len() > 2 {
                args[2].clone()
            } else {
                "reports".to_string()
            };
            let output_dir = if args.len() > 3 {
                Some(args[3].clone())
            } else {
                None
            };
            Command::ReportOnly(ReportOnlyArgs {
                source_dir,
                output_dir,
            })
        }

        // MindSpider モード
        "mindspider" | "spider" => {
            let subcommand = if args.len() > 2 {
                args[2].clone()
            } else {
                "complete".to_string()
            };

            let mut date = None;
            let mut platforms = None;
            let mut keywords_count: usize = 100;
            let mut max_keywords: usize = 50;
            let mut max_notes: usize = 50;
            let mut test_mode = false;

            let mut i = 3;
            while i < args.len() {
                match args[i].as_str() {
                    "--date" => {
                        if i + 1 < args.len() {
                            date = Some(args[i + 1].clone());
                            i += 1;
                        }
                    }
                    "--platforms" => {
                        let mut plist = Vec::new();
                        i += 1;
                        while i < args.len() && !args[i].starts_with("--") {
                            plist.push(args[i].clone());
                            i += 1;
                        }
                        if !plist.is_empty() {
                            platforms = Some(plist);
                        }
                        continue; // i already incremented past platforms
                    }
                    "--keywords-count" => {
                        if i + 1 < args.len() {
                            keywords_count = args[i + 1].parse().unwrap_or(100);
                            i += 1;
                        }
                    }
                    "--max-keywords" => {
                        if i + 1 < args.len() {
                            max_keywords = args[i + 1].parse().unwrap_or(50);
                            i += 1;
                        }
                    }
                    "--max-notes" => {
                        if i + 1 < args.len() {
                            max_notes = args[i + 1].parse().unwrap_or(50);
                            i += 1;
                        }
                    }
                    "--test" => {
                        test_mode = true;
                    }
                    _ => {}
                }
                i += 1;
            }

            Command::MindSpiderCmd(MindSpiderArgs {
                subcommand,
                date,
                platforms,
                keywords_count,
                max_keywords,
                max_notes,
                test_mode,
            })
        }

        // パイプラインモード
        "pipeline" => {
            let query = if args.len() > 2 {
                args[2..].join(" ")
            } else {
                "人工智能最新发展趋势分析".to_string()
            };
            Command::Pipeline(PipelineArgs {
                query,
                skip_crawl: args.iter().any(|a| a == "--skip-crawl"),
                platforms: None,
            })
        }

        // デフォルト: クエリとして扱う
        _ => {
            let query = args[1..].join(" ");
            Command::Query(query)
        }
    }
}

/// ヘルプメッセージを表示
fn print_help() {
    println!(
        r#"
BettaFish Rust v{} - 世論分析プラットフォーム

使用法:
  bettafish [コマンド] [オプション]

コマンド:
  query <テキスト>         QueryEngine でクエリを実行 (デフォルト)
  insight <テキスト>       InsightEngine でクエリを実行 (ローカルDB)
  media <テキスト>         MediaEngine でクエリを実行 (マルチモーダル)
  mindspider <サブコマンド>  MindSpider クローラーを実行
  pipeline <テキスト>      完全パイプラインを実行
  server                   HTTP サーバーを起動
  config                   設定情報を表示
  status                   プロジェクトステータスを表示
  export-pdf <入力> [出力]  PDF エクスポート
  regenerate-html <入力>   HTML を再生成
  regenerate-md <入力>     Markdown を再生成
  report-only <ソース>     レポートのみ生成
  help                     このヘルプを表示

QueryEngine:
  bettafish query "人工知能の最新動向"
  bettafish "テーマを直接入力"

InsightEngine:
  bettafish insight "半導体業界の世論分析"

MediaEngine:
  bettafish media "AI 画像生成の動向"

HTTP サーバー:
  bettafish server

MindSpider サブコマンド:
  broad-topic              トピック抽出を実行
  deep-sentiment           深度センチメントクローリングを実行
  complete                 完全ワークフローを実行
  setup                    プロジェクト初期設定
  init-db                  データベースを初期化
  status                   ステータスを表示

MindSpider オプション:
  --date YYYY-MM-DD        対象日付 (デフォルト: 今日)
  --platforms xhs dy bili   対象プラットフォーム
  --keywords-count N       キーワード数 (デフォルト: 100)
  --max-keywords N         最大キーワード数 (デフォルト: 50)
  --max-notes N            最大ノート数 (デフォルト: 50)
  --test                   テストモード (少量データ)

レポート操作:
  bettafish export-pdf reports/report.json output.pdf
  bettafish regenerate-html reports/report.json
  bettafish regenerate-md reports/report.json
  bettafish report-only reports/

サポートプラットフォーム:
  xhs (小紅書), dy (抖音), ks (快手), bili (B站),
  wb (微博), tieba (貼吧), zhihu (知乎)

例:
  bettafish query "AI最新トレンド"
  bettafish insight "半導体業界の世論"
  bettafish media "AI画像生成の動向"
  bettafish server
  bettafish mindspider broad-topic --keywords-count 50
  bettafish mindspider deep-sentiment --platforms xhs dy --test
  bettafish mindspider complete --date 2025-01-15
  bettafish pipeline "半導体業界の動向分析"
  bettafish export-pdf reports/latest.json
  bettafish report-only reports/
"#,
        VERSION
    );
}

/// 設定情報を表示
fn show_config(settings: &Settings) {
    println!("{}", "=".repeat(60));
    println!("BettaFish 設定情報");
    println!("{}", "=".repeat(60));
    println!();

    println!("--- サーバー ---");
    println!("  Host: {}", settings.host);
    println!("  Port: {}", settings.port);
    println!();

    println!("--- データベース ---");
    println!("  Dialect: {}", settings.db_dialect);
    println!("  Host: {}", settings.db_host);
    println!("  Port: {}", settings.db_port);
    println!("  User: {}", settings.db_user);
    println!("  DB Name: {}", settings.db_name);
    println!();

    let engines = [
        (
            "QueryEngine",
            &settings.query_engine_model_name,
            &settings.query_engine_base_url,
            !settings.query_engine_api_key.is_empty(),
        ),
        (
            "InsightEngine",
            &settings.insight_engine_model_name,
            &settings.insight_engine_base_url,
            !settings.insight_engine_api_key.is_empty(),
        ),
        (
            "MediaEngine",
            &settings.media_engine_model_name,
            &settings.media_engine_base_url,
            !settings.media_engine_api_key.is_empty(),
        ),
        (
            "ReportEngine",
            &settings.report_engine_model_name,
            &settings.report_engine_base_url,
            !settings.report_engine_api_key.is_empty(),
        ),
        (
            "MindSpider",
            &settings.mindspider_model_name,
            &settings.mindspider_base_url,
            !settings.mindspider_api_key.is_empty(),
        ),
        (
            "ForumHost",
            &settings.forum_host_model_name,
            &settings.forum_host_base_url,
            !settings.forum_host_api_key.is_empty(),
        ),
        (
            "KeywordOptimizer",
            &settings.keyword_optimizer_model_name,
            &settings.keyword_optimizer_base_url,
            !settings.keyword_optimizer_api_key.is_empty(),
        ),
    ];

    println!("--- LLM エンジン ---");
    for (name, model, base_url, has_key) in &engines {
        let key_status = if *has_key { "設定済み" } else { "未設定" };
        let url = base_url
            .as_deref()
            .unwrap_or("(デフォルト)");
        println!("  {} | Model: {} | URL: {} | Key: {}",
            name, model, url, key_status);
    }
    println!();

    println!("--- 検索ツール ---");
    println!("  Type: {}", settings.search_tool_type);
    println!(
        "  Tavily Key: {}",
        if settings.tavily_api_key.is_empty() { "未設定" } else { "設定済み" }
    );
    println!(
        "  Anspire Key: {}",
        if settings.anspire_api_key.is_empty() { "未設定" } else { "設定済み" }
    );
    println!();

    println!("--- パラメータ ---");
    println!("  Max Reflections: {}", settings.max_reflections);
    println!("  Max Paragraphs: {}", settings.max_paragraphs);
    println!("  Search Timeout: {}s", settings.search_timeout);
    println!("  Max Search Results: {}", settings.max_search_results);
    println!("  Output Dir: {}", settings.output_dir);
    println!();

    println!(
        "--- サポートプラットフォーム ---\n  {}",
        SUPPORTED_PLATFORMS.join(", ")
    );
    println!("{}", "=".repeat(60));
}

/// QueryEngine を実行
async fn run_query_engine(settings: Settings, query: &str) -> Result<()> {
    info!("研究クエリ: {}", query);
    info!("{}", "=".repeat(60));

    match DeepSearchAgent::new(Some(settings)) {
        Ok(mut agent) => match agent.research(query, true).await {
            Ok(report) => {
                info!("\n{}", "=".repeat(60));
                info!("研究が正常に完了しました！");
                info!("レポートの長さ: {} 文字", report.len());
                info!("{}", "=".repeat(60));

                let progress = agent.get_progress_summary();
                info!(
                    "進捗サマリー: {}",
                    serde_json::to_string_pretty(&progress)?
                );
                Ok(())
            }
            Err(e) => {
                error!("研究中にエラーが発生: {}", e);
                std::process::exit(1);
            }
        },
        Err(e) => {
            error!("エージェントの初期化に失敗: {}", e);
            error!(
                "ヒント: .env ファイルに QUERY_ENGINE_API_KEY と TAVILY_API_KEY を設定してください"
            );
            std::process::exit(1);
        }
    }
}

/// MindSpider コマンドを実行
async fn run_mindspider(settings: Settings, args: MindSpiderArgs) -> Result<()> {
    let spider = MindSpider::new(settings);

    // 日付を解析
    let target_date = if let Some(date_str) = &args.date {
        match chrono::NaiveDate::parse_from_str(date_str, "%Y-%m-%d") {
            Ok(d) => Some(d),
            Err(_) => {
                error!("日付フォーマットが不正です。YYYY-MM-DD 形式を使用してください");
                std::process::exit(1);
            }
        }
    } else {
        None
    };

    // プラットフォームバリデーション
    if let Some(ref platforms) = args.platforms {
        for p in platforms {
            if !SUPPORTED_PLATFORMS.contains(&p.as_str()) {
                // エイリアス解決を試みる
                if let Some(resolved) = bettafish_mindspider::resolve_platform_alias(p) {
                    info!("プラットフォーム '{}' -> '{}' に解決しました", p, resolved);
                } else {
                    error!(
                        "未知のプラットフォーム: '{}'. サポート: {:?}",
                        p, SUPPORTED_PLATFORMS
                    );
                    std::process::exit(1);
                }
            }
        }
    }

    // テストモードの調整
    let max_keywords = if args.test_mode {
        std::cmp::min(args.max_keywords, 10)
    } else {
        args.max_keywords
    };
    let max_notes = if args.test_mode {
        std::cmp::min(args.max_notes, 10)
    } else {
        args.max_notes
    };

    match args.subcommand.as_str() {
        "status" => {
            spider.show_status();
        }
        "setup" => {
            if !spider.check_config() {
                error!("設定チェックに失敗しました");
                std::process::exit(1);
            }
            info!("プロジェクトセットアップ完了");
        }
        "init-db" => {
            if spider.initialize_database().await {
                info!("データベース初期化成功");
            } else {
                error!("データベース初期化失敗");
                std::process::exit(1);
            }
        }
        "broad-topic" => {
            if !spider
                .run_broad_topic_extraction(target_date, args.keywords_count)
                .await
            {
                error!("トピック抽出に失敗しました");
                std::process::exit(1);
            }
        }
        "deep-sentiment" => {
            if !spider
                .run_deep_sentiment_crawling(
                    target_date,
                    args.platforms,
                    max_keywords,
                    max_notes,
                    args.test_mode,
                )
                .await
            {
                error!("深度センチメントクローリングに失敗しました");
                std::process::exit(1);
            }
        }
        "complete" => {
            if !spider
                .run_complete_workflow(
                    target_date,
                    args.platforms,
                    args.keywords_count,
                    max_keywords,
                    max_notes,
                    args.test_mode,
                )
                .await
            {
                error!("完全ワークフローに失敗しました");
                std::process::exit(1);
            }
        }
        other => {
            error!("不明な MindSpider サブコマンド: '{}'", other);
            error!("有効なサブコマンド: broad-topic, deep-sentiment, complete, setup, init-db, status");
            std::process::exit(1);
        }
    }

    Ok(())
}

/// パイプラインを実行 (MindSpider → QueryEngine → Report)
async fn run_pipeline(settings: Settings, args: PipelineArgs) -> Result<()> {
    info!("{}", "=".repeat(60));
    info!("BettaFish 完全パイプライン開始");
    info!("クエリ: {}", args.query);
    info!("{}", "=".repeat(60));

    // ステップ 1: MindSpider でデータ収集 (スキップ可能)
    if !args.skip_crawl {
        info!("\n=== ステップ 1/3: MindSpider データ収集 ===");
        let spider = MindSpider::new(settings.clone());
        let success = spider
            .run_complete_workflow(None, args.platforms, 100, 50, 50, false)
            .await;
        if !success {
            warn!("MindSpider ワークフローが完全には成功しませんでしたが、続行します");
        }
    } else {
        info!("\n=== ステップ 1/3: MindSpider データ収集 (スキップ) ===");
    }

    // ステップ 2: QueryEngine で分析
    info!("\n=== ステップ 2/3: QueryEngine 分析 ===");
    match DeepSearchAgent::new(Some(settings.clone())) {
        Ok(mut agent) => {
            match agent.research(&args.query, true).await {
                Ok(report) => {
                    info!("QueryEngine 分析完了: {} 文字", report.len());

                    // ステップ 3: レポート生成 (スタブ)
                    info!("\n=== ステップ 3/3: レポート生成 ===");
                    info!("ReportEngine はまだ完全実装されていません (スタブ)");
                    info!("分析結果は QueryEngine レポートとして出力されました");
                }
                Err(e) => {
                    error!("QueryEngine 分析エラー: {}", e);
                }
            }
        }
        Err(e) => {
            error!("QueryEngine 初期化失敗: {}", e);
            error!("ヒント: .env に QUERY_ENGINE_API_KEY を設定してください");
        }
    }

    info!("\n{}", "=".repeat(60));
    info!("BettaFish パイプライン完了");
    info!("{}", "=".repeat(60));

    Ok(())
}

/// HTTP サーバーを起動
async fn run_server(settings: Settings) -> Result<()> {
    let app_server = server::AppServer::new(settings);
    app_server.start().await
}

/// PDF エクスポートを実行
async fn run_export_pdf(_settings: Settings, args: ExportArgs) -> Result<()> {
    info!("{}", "=".repeat(60));
    info!("PDF エクスポート");
    info!("入力: {}", args.input_path);
    info!("{}", "=".repeat(60));

    let output_path = args.output_path.unwrap_or_else(|| {
        let stem = std::path::Path::new(&args.input_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("report");
        format!("{}.pdf", stem)
    });

    // ReportEngine の PDFRenderer を使用
    let pdf_renderer = bettafish_report_engine::PDFRenderer::new();
    info!("PDF レンダラーを初期化しました");

    // 入力ファイルを読み取り
    match std::fs::read_to_string(&args.input_path) {
        Ok(content) => {
            match serde_json::from_str::<serde_json::Value>(&content) {
                Ok(document_ir) => {
                    match pdf_renderer.render_to_html(&document_ir, &output_path) {
                        Ok(()) => {
                            info!("PDF エクスポート完了: {}", output_path);
                        }
                        Err(e) => {
                            error!("PDF レンダリング失敗: {}", e);
                            error!("注意: PDF レンダリングには wkhtmltopdf が必要です");
                        }
                    }
                }
                Err(e) => {
                    error!("JSON 解析失敗: {}", e);
                    error!("入力ファイルが有効な IR JSON である必要があります");
                }
            }
        }
        Err(e) => {
            error!("ファイル読み取り失敗: {} - {}", args.input_path, e);
        }
    }

    Ok(())
}

/// HTML 再生成を実行
async fn run_regenerate_html(_settings: Settings, args: ExportArgs) -> Result<()> {
    info!("{}", "=".repeat(60));
    info!("HTML 再生成");
    info!("入力: {}", args.input_path);
    info!("{}", "=".repeat(60));

    let output_path = args.output_path.unwrap_or_else(|| {
        let stem = std::path::Path::new(&args.input_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("report");
        format!("{}.html", stem)
    });

    match std::fs::read_to_string(&args.input_path) {
        Ok(content) => {
            match serde_json::from_str::<serde_json::Value>(&content) {
                Ok(document_ir) => {
                    let renderer = bettafish_report_engine::HTMLRenderer::new();
                    let html = renderer.render(&document_ir);
                    match std::fs::write(&output_path, &html) {
                        Ok(()) => {
                            info!("HTML 再生成完了: {} ({} 文字)", output_path, html.len());
                        }
                        Err(e) => {
                            error!("ファイル書き込み失敗: {}", e);
                        }
                    }
                }
                Err(e) => {
                    error!("JSON 解析失敗: {}", e);
                }
            }
        }
        Err(e) => {
            error!("ファイル読み取り失敗: {} - {}", args.input_path, e);
        }
    }

    Ok(())
}

/// Markdown 再生成を実行
async fn run_regenerate_md(_settings: Settings, args: ExportArgs) -> Result<()> {
    info!("{}", "=".repeat(60));
    info!("Markdown 再生成");
    info!("入力: {}", args.input_path);
    info!("{}", "=".repeat(60));

    let output_path = args.output_path.unwrap_or_else(|| {
        let stem = std::path::Path::new(&args.input_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("report");
        format!("{}.md", stem)
    });

    match std::fs::read_to_string(&args.input_path) {
        Ok(content) => {
            match serde_json::from_str::<serde_json::Value>(&content) {
                Ok(document_ir) => {
                    let md_renderer = bettafish_report_engine::MarkdownRenderer::new();
                    let markdown = md_renderer.render(&document_ir);
                    match std::fs::write(&output_path, &markdown) {
                        Ok(()) => {
                            info!("Markdown 再生成完了: {} ({} 文字)", output_path, markdown.len());
                        }
                        Err(e) => {
                            error!("ファイル書き込み失敗: {}", e);
                        }
                    }
                }
                Err(e) => {
                    error!("JSON 解析失敗: {}", e);
                }
            }
        }
        Err(e) => {
            error!("ファイル読み取り失敗: {} - {}", args.input_path, e);
        }
    }

    Ok(())
}

/// レポートのみ生成を実行
async fn run_report_only(settings: Settings, args: ReportOnlyArgs) -> Result<()> {
    info!("{}", "=".repeat(60));
    info!("レポートのみ生成モード");
    info!("ソース: {}", args.source_dir);
    info!("{}", "=".repeat(60));

    let output_dir = args.output_dir.unwrap_or_else(|| settings.output_dir.clone());

    match bettafish_report_engine::ReportAgent::new(Some(settings)) {
        Ok(_agent) => {
            info!("ReportAgent を初期化しました");
            info!("ソースディレクトリ: {}", args.source_dir);
            info!("出力ディレクトリ: {}", output_dir);

            // ソースディレクトリから JSON ファイルを検索
            let source_path = std::path::Path::new(&args.source_dir);
            if !source_path.exists() {
                error!("ソースディレクトリが存在しません: {}", args.source_dir);
                std::process::exit(1);
            }

            info!("ReportAgent によるレポート生成を開始します");
            info!("注意: 完全な実装にはソースデータの読み込みと統合が必要です");
            info!("現在はスタブ実装として出力ディレクトリの確認のみ行います");

            std::fs::create_dir_all(&output_dir)?;
            info!("出力ディレクトリを確認/作成しました: {}", output_dir);
        }
        Err(e) => {
            error!("ReportAgent 初期化失敗: {}", e);
            error!("ヒント: .env に REPORT_ENGINE_API_KEY を設定してください");
            std::process::exit(1);
        }
    }

    Ok(())
}

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

    info!("BettaFish Rust v{}", VERSION);

    // コマンドを解析
    let command = parse_args();

    // 設定をロード
    let settings = Settings::load();

    match command {
        Command::Help => {
            print_help();
        }
        Command::ShowConfig => {
            show_config(&settings);
        }
        Command::Status => {
            show_config(&settings);
            let spider = MindSpider::new(settings);
            spider.show_status();
        }
        Command::Query(query) => {
            info!("{}", settings.display_query_engine_config());
            run_query_engine(settings, &query).await?;
        }
        Command::Insight(query) => {
            info!("InsightEngine モード");
            single_engine::run_insight_engine(&query, settings).await?;
        }
        Command::Media(query) => {
            info!("MediaEngine モード");
            single_engine::run_media_engine(&query, settings).await?;
        }
        Command::MindSpiderCmd(args) => {
            run_mindspider(settings, args).await?;
        }
        Command::Pipeline(args) => {
            run_pipeline(settings, args).await?;
        }
        Command::Server => {
            run_server(settings).await?;
        }
        Command::ExportPdf(args) => {
            run_export_pdf(settings, args).await?;
        }
        Command::RegenerateHtml(args) => {
            run_regenerate_html(settings, args).await?;
        }
        Command::RegenerateMd(args) => {
            run_regenerate_md(settings, args).await?;
        }
        Command::ReportOnly(args) => {
            run_report_only(settings, args).await?;
        }
    }

    Ok(())
}
