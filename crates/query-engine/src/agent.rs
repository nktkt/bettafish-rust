//! Deep Search Agent メインクラス
//!
//! Python の agent.py の Rust 実装。
//! 全モジュールを統合し、完全な深度検索フローを実現。

use anyhow::{Context, Result};
use chrono::Local;
use regex::Regex;
use serde_json::Value;
use std::fs;
use std::path::Path;
use tracing::{error, info, warn};

use bettafish_config::Settings;
use bettafish_llm::LLMClient;

use crate::nodes::{
    FirstSearchNode, FirstSummaryNode, Node, ReflectionNode, ReflectionSummaryNode,
    ReportFormattingNode, ReportStructureNode, StateMutationNode,
};
use crate::state::State;
use crate::tools::{TavilyNewsAgency, TavilyResponse};

/// Deep Search Agent メインクラス
pub struct DeepSearchAgent {
    config: Settings,
    llm_client: LLMClient,
    search_agency: TavilyNewsAgency,
    first_search_node: FirstSearchNode,
    reflection_node: ReflectionNode,
    first_summary_node: FirstSummaryNode,
    reflection_summary_node: ReflectionSummaryNode,
    report_formatting_node: ReportFormattingNode,
    pub state: State,
}

impl DeepSearchAgent {
    /// Deep Search Agent を初期化
    pub fn new(config: Option<Settings>) -> Result<Self> {
        let config = config.unwrap_or_else(Settings::load);

        let llm_client = LLMClient::new(
            &config.query_engine_api_key,
            &config.query_engine_model_name,
            config.query_engine_base_url.as_deref(),
        )?;

        let search_agency = TavilyNewsAgency::new(&config.tavily_api_key)?;

        // ノードを初期化
        let first_search_node = FirstSearchNode::new(llm_client.clone());
        let reflection_node = ReflectionNode::new(llm_client.clone());
        let first_summary_node = FirstSummaryNode::new(llm_client.clone());
        let reflection_summary_node = ReflectionSummaryNode::new(llm_client.clone());
        let report_formatting_node = ReportFormattingNode::new(llm_client.clone());

        // 出力ディレクトリを作成
        fs::create_dir_all(&config.output_dir).ok();

        info!("Query Agent を初期化しました");
        info!("使用 LLM: {:?}", llm_client.get_model_info());
        info!("検索ツールセット: TavilyNewsAgency (6 種類の検索ツールをサポート)");

        Ok(Self {
            config,
            llm_client,
            search_agency,
            first_search_node,
            reflection_node,
            first_summary_node,
            reflection_summary_node,
            report_formatting_node,
            state: State::default(),
        })
    }

    /// 日付フォーマット（YYYY-MM-DD）を検証
    fn validate_date_format(date_str: &str) -> bool {
        if date_str.is_empty() {
            return false;
        }

        let re = Regex::new(r"^\d{4}-\d{2}-\d{2}$").unwrap();
        if !re.is_match(date_str) {
            return false;
        }

        chrono::NaiveDate::parse_from_str(date_str, "%Y-%m-%d").is_ok()
    }

    /// 指定の検索ツールを実行
    async fn execute_search_tool(
        &self,
        tool_name: &str,
        query: &str,
        start_date: Option<&str>,
        end_date: Option<&str>,
    ) -> TavilyResponse {
        info!("  → 検索ツールを実行: {}", tool_name);

        match tool_name {
            "basic_search_news" => self.search_agency.basic_search_news(query, 7).await,
            "deep_search_news" => self.search_agency.deep_search_news(query).await,
            "search_news_last_24_hours" => {
                self.search_agency.search_news_last_24_hours(query).await
            }
            "search_news_last_week" => self.search_agency.search_news_last_week(query).await,
            "search_images_for_news" => self.search_agency.search_images_for_news(query).await,
            "search_news_by_date" => {
                let sd = start_date.unwrap_or("");
                let ed = end_date.unwrap_or("");
                if sd.is_empty() || ed.is_empty() {
                    warn!("search_news_by_date ツールに start_date と end_date が必要です");
                    self.search_agency.basic_search_news(query, 7).await
                } else {
                    self.search_agency.search_news_by_date(query, sd, ed).await
                }
            }
            _ => {
                warn!("  ⚠️  未知の検索ツール: {}、デフォルトの基礎検索を使用", tool_name);
                self.search_agency.basic_search_news(query, 7).await
            }
        }
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

    /// レポート構造を生成
    async fn generate_report_structure(&mut self, query: &str) -> Result<()> {
        info!("\n[ステップ 1] レポート構造を生成中...");

        let report_structure_node = ReportStructureNode::new(self.llm_client.clone(), query);
        report_structure_node
            .mutate_state(&Value::Null, &mut self.state, 0)
            .await?;

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

    /// 全段落を処理
    async fn process_paragraphs(&mut self) -> Result<()> {
        let total_paragraphs = self.state.paragraphs.len();

        for i in 0..total_paragraphs {
            info!(
                "\n[ステップ 2.{}] 段落を処理中: {}",
                i + 1,
                self.state.paragraphs[i].title
            );
            info!("{}", "-".repeat(50));

            // 初回検索とサマリー
            self.initial_search_and_summary(i).await?;

            // リフレクションループ
            self.reflection_loop(i).await?;

            // 段落完了マーク
            self.state.paragraphs[i].research.mark_completed();

            let progress = (i + 1) as f64 / total_paragraphs as f64 * 100.0;
            info!("段落処理完了 ({:.1}%)", progress);
        }

        Ok(())
    }

    /// 初回検索とサマリーを実行
    async fn initial_search_and_summary(&mut self, paragraph_index: usize) -> Result<()> {
        let title = self.state.paragraphs[paragraph_index].title.clone();
        let content = self.state.paragraphs[paragraph_index].content.clone();

        // 検索入力を準備
        let search_input = serde_json::json!({
            "title": title,
            "content": content,
        });

        // 検索クエリとツール選択を生成
        info!("  - 検索クエリを生成中...");
        let search_output = self.first_search_node.run(&search_input).await?;
        let search_query = search_output
            .get("search_query")
            .and_then(|v| v.as_str())
            .unwrap_or("相关主题研究")
            .to_string();
        let mut search_tool = search_output
            .get("search_tool")
            .and_then(|v| v.as_str())
            .unwrap_or("basic_search_news")
            .to_string();
        let reasoning = search_output
            .get("reasoning")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        info!("  - 検索クエリ: {}", search_query);
        info!("  - 選択されたツール: {}", search_tool);
        info!("  - 推論: {}", reasoning);

        // search_news_by_date の特殊パラメータ処理
        let mut start_date: Option<String> = None;
        let mut end_date: Option<String> = None;

        if search_tool == "search_news_by_date" {
            let sd = search_output.get("start_date").and_then(|v| v.as_str());
            let ed = search_output.get("end_date").and_then(|v| v.as_str());

            if let (Some(sd_val), Some(ed_val)) = (sd, ed) {
                if Self::validate_date_format(sd_val) && Self::validate_date_format(ed_val) {
                    start_date = Some(sd_val.to_string());
                    end_date = Some(ed_val.to_string());
                    info!("  - 時間範囲: {} から {}", sd_val, ed_val);
                } else {
                    info!("  ⚠️  日付フォーマットエラー（YYYY-MM-DD が必要）、基礎検索に変更");
                    search_tool = "basic_search_news".to_string();
                }
            } else {
                info!("  ⚠️  search_news_by_date ツールに時間パラメータが不足、基礎検索に変更");
                search_tool = "basic_search_news".to_string();
            }
        }

        // 検索を実行
        info!("  - ウェブ検索を実行中...");
        let search_response = self
            .execute_search_tool(
                &search_tool,
                &search_query,
                start_date.as_deref(),
                end_date.as_deref(),
            )
            .await;

        // 互換フォーマットに変換
        let search_results = self.convert_search_results(&search_response);

        if !search_results.is_empty() {
            let mut msg = format!("  - {} 件の検索結果を発見", search_results.len());
            for (j, result) in search_results.iter().enumerate() {
                let title_preview: String = result
                    .get("title")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .chars()
                    .take(50)
                    .collect();
                let date_info = result
                    .get("published_date")
                    .and_then(|v| v.as_str())
                    .map(|d| format!(" (公開日: {})", d))
                    .unwrap_or_default();
                msg.push_str(&format!("\n    {}. {}...{}", j + 1, title_preview, date_info));
            }
            info!("{}", msg);
        } else {
            info!("  - 検索結果が見つかりません");
        }

        // 状態の検索履歴を更新
        self.state.paragraphs[paragraph_index]
            .research
            .add_search_results(&search_query, &search_results);

        // 初回サマリーを生成
        info!("  - 初回サマリーを生成中...");
        let formatted_results =
            bettafish_common::text_processing::format_search_results_for_prompt(
                &search_results
                    .iter()
                    .map(|v| {
                        let mut map = std::collections::HashMap::new();
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

        self.first_summary_node
            .mutate_state(&summary_input, &mut self.state, paragraph_index)
            .await?;

        info!("  - 初回サマリー完了");
        Ok(())
    }

    /// リフレクションループを実行
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

            // リフレクション入力を準備
            let reflection_input = serde_json::json!({
                "title": title,
                "content": content,
                "paragraph_latest_state": latest_summary,
            });

            // リフレクション検索クエリを生成
            let reflection_output = self.reflection_node.run(&reflection_input).await?;
            let search_query = reflection_output
                .get("search_query")
                .and_then(|v| v.as_str())
                .unwrap_or("深度研究补充信息")
                .to_string();
            let mut search_tool = reflection_output
                .get("search_tool")
                .and_then(|v| v.as_str())
                .unwrap_or("basic_search_news")
                .to_string();
            let reasoning = reflection_output
                .get("reasoning")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            info!("    リフレクションクエリ: {}", search_query);
            info!("    選択されたツール: {}", search_tool);
            info!("    リフレクション推論: {}", reasoning);

            // search_news_by_date の特殊パラメータ処理
            let mut start_date: Option<String> = None;
            let mut end_date: Option<String> = None;

            if search_tool == "search_news_by_date" {
                let sd = reflection_output.get("start_date").and_then(|v| v.as_str());
                let ed = reflection_output.get("end_date").and_then(|v| v.as_str());

                if let (Some(sd_val), Some(ed_val)) = (sd, ed) {
                    if Self::validate_date_format(sd_val) && Self::validate_date_format(ed_val) {
                        start_date = Some(sd_val.to_string());
                        end_date = Some(ed_val.to_string());
                        info!("    時間範囲: {} から {}", sd_val, ed_val);
                    } else {
                        info!("    ⚠️  日付フォーマットエラー、基礎検索に変更");
                        search_tool = "basic_search_news".to_string();
                    }
                } else {
                    info!("    ⚠️  search_news_by_date ツールに時間パラメータが不足、基礎検索に変更");
                    search_tool = "basic_search_news".to_string();
                }
            }

            // リフレクション検索を実行
            let search_response = self
                .execute_search_tool(
                    &search_tool,
                    &search_query,
                    start_date.as_deref(),
                    end_date.as_deref(),
                )
                .await;

            let search_results = self.convert_search_results(&search_response);

            if !search_results.is_empty() {
                info!("    {} 件のリフレクション検索結果を発見", search_results.len());
            } else {
                info!("    リフレクション検索結果が見つかりません");
            }

            // 検索履歴を更新
            self.state.paragraphs[paragraph_index]
                .research
                .add_search_results(&search_query, &search_results);

            // リフレクションサマリーを生成
            let formatted_results =
                bettafish_common::text_processing::format_search_results_for_prompt(
                    &search_results
                        .iter()
                        .map(|v| {
                            let mut map = std::collections::HashMap::new();
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

            self.reflection_summary_node
                .mutate_state(&reflection_summary_input, &mut self.state, paragraph_index)
                .await?;

            info!("    リフレクション {} 完了", reflection_i + 1);
        }

        Ok(())
    }

    /// 最終レポートを生成
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

    /// レポートをファイルに保存
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

        fs::write(&filepath, report_content)
            .context("レポートの保存に失敗")?;
        info!("レポートを保存しました: {}", filepath.display());

        // 状態を保存（設定が許可する場合）
        if self.config.save_intermediate_states {
            let state_filename = format!("state_{}_{}.json", query_safe, timestamp);
            let state_filepath = Path::new(&self.config.output_dir).join(&state_filename);
            self.state.save_to_file(
                state_filepath.to_str().unwrap_or("state.json"),
            )?;
            info!("状態を保存しました: {}", state_filepath.display());
        }

        Ok(())
    }

    /// 検索レスポンスを Value ベクトルに変換
    fn convert_search_results(&self, response: &TavilyResponse) -> Vec<Value> {
        let max_results = response.results.len().min(10);
        response.results[..max_results]
            .iter()
            .map(|result| {
                serde_json::json!({
                    "title": result.title,
                    "url": result.url,
                    "content": result.content,
                    "score": result.score,
                    "raw_content": result.raw_content,
                    "published_date": result.published_date,
                })
            })
            .collect()
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

/// Deep Search Agent インスタンスを作成する便利関数
#[allow(dead_code)]
pub fn create_agent() -> Result<DeepSearchAgent> {
    DeepSearchAgent::new(None)
}
