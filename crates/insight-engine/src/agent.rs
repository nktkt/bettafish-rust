//! Deep Search Agent メインクラス (InsightEngine 版)
//!
//! Python の agent.py の Rust 実装。
//! MediaCrawlerDB + KeywordOptimizer + SentimentAnalyzer を統合した
//! 完全な深度世論検索フローを実現。

use anyhow::{Context, Result};
use chrono::Local;
use regex::Regex;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
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
use crate::tools::db::{DBResponse, MediaCrawlerDB, QueryResult};
use crate::tools::keyword_optimizer::KeywordOptimizer;
use crate::tools::sentiment::StubSentimentAnalyzer;
use crate::{ENABLE_CLUSTERING, MAX_CLUSTERED_RESULTS, RESULTS_PER_CLUSTER};

/// Deep Search Agent メインクラス (InsightEngine 版)
pub struct DeepSearchAgent {
    config: Settings,
    llm_client: LLMClient,
    search_agency: MediaCrawlerDB,
    keyword_optimizer: KeywordOptimizer,
    #[allow(dead_code)]
    sentiment_analyzer: StubSentimentAnalyzer,
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
            &config.insight_engine_api_key,
            &config.insight_engine_model_name,
            config.insight_engine_base_url.as_deref(),
        )?;

        let search_agency = MediaCrawlerDB::new();

        let keyword_optimizer = KeywordOptimizer::from_config(&config)
            .unwrap_or_else(|e| {
                warn!("KeywordOptimizer 初期化失敗、LLM クライアントで代替: {}", e);
                KeywordOptimizer::new(llm_client.clone())
            });

        let sentiment_analyzer = StubSentimentAnalyzer::new();

        // ノードを初期化
        let first_search_node = FirstSearchNode::new(llm_client.clone());
        let reflection_node = ReflectionNode::new(llm_client.clone());
        let first_summary_node = FirstSummaryNode::new(llm_client.clone());
        let reflection_summary_node = ReflectionSummaryNode::new(llm_client.clone());
        let report_formatting_node = ReportFormattingNode::new(llm_client.clone());

        // 出力ディレクトリを作成
        fs::create_dir_all(&config.output_dir).ok();

        info!("Insight Agent を初期化しました");
        info!("使用 LLM: {:?}", llm_client.get_model_info());
        info!("検索ツールセット: MediaCrawlerDB (5種類のローカルDB検索ツールをサポート)");
        info!("感情分析: StubSentimentAnalyzer (将来 ONNX Runtime 統合で有効化)");

        Ok(Self {
            config,
            llm_client,
            search_agency,
            keyword_optimizer,
            sentiment_analyzer,
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

    /// 結果を去重
    fn deduplicate_results(results: Vec<QueryResult>) -> Vec<QueryResult> {
        let mut seen = HashSet::new();
        let mut unique = Vec::new();

        for result in results {
            let identifier = if let Some(ref url) = result.url {
                if !url.is_empty() {
                    url.clone()
                } else {
                    result.title_or_content.chars().take(100).collect()
                }
            } else {
                result.title_or_content.chars().take(100).collect()
            };

            if seen.insert(identifier) {
                unique.push(result);
            }
        }

        unique
    }

    /// クラスタリングとサンプリング (スタブ実装)
    ///
    /// Python 版では sentence-transformers + KMeans を使用。
    /// Rust 版では将来 ONNX Runtime 統合時に実装。
    /// 現在は熱度スコアでソートして上位 N 件を返すフォールバックを使用。
    fn cluster_and_sample_results(
        results: Vec<QueryResult>,
        max_results: usize,
        _results_per_cluster: usize,
    ) -> Vec<QueryResult> {
        if results.len() <= max_results {
            return results;
        }

        info!(
            "  クラスタリング (スタブ): {} 件 -> 上位 {} 件を熱度順に返却",
            results.len(),
            max_results
        );

        // 熱度スコアでソートして上位を返す
        let mut sorted = results;
        sorted.sort_by(|a, b| {
            b.hotness_score
                .partial_cmp(&a.hotness_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.truncate(max_results);
        sorted
    }

    /// 指定の検索ツールを実行 (KeywordOptimizer + SentimentAnalyzer 統合)
    pub async fn execute_search_tool(
        &self,
        tool_name: &str,
        query: &str,
        kwargs: &HashMap<String, Value>,
    ) -> DBResponse {
        info!("  -> 执行数据库查询工具: {}", tool_name);

        // search_hot_content はキーワード最適化不要
        if tool_name == "search_hot_content" {
            let time_period = kwargs
                .get("time_period")
                .and_then(|v| v.as_str())
                .unwrap_or("week");
            let limit = kwargs
                .get("limit")
                .and_then(|v| v.as_u64())
                .unwrap_or(100) as usize;

            return self
                .search_agency
                .search_hot_content(time_period, limit)
                .await;
        }

        // その他のツールはキーワード最適化を使用
        let optimized = self
            .keyword_optimizer
            .optimize_keywords(query, &format!("使用{}工具进行查询", tool_name))
            .await;

        info!("  原始查询: '{}'", query);
        info!("  优化后关键词: {:?}", optimized.optimized_keywords);

        let mut all_results = Vec::new();

        for keyword in &optimized.optimized_keywords {
            info!("    查询关键词: '{}'", keyword);

            let response = match tool_name {
                "search_topic_globally" => {
                    let limit_per_table =
                        self.config.default_search_topic_globally_limit_per_table;
                    self.search_agency
                        .search_topic_globally(keyword, limit_per_table)
                        .await
                }
                "search_topic_by_date" => {
                    let start_date = kwargs
                        .get("start_date")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let end_date = kwargs
                        .get("end_date")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let limit_per_table =
                        self.config.default_search_topic_by_date_limit_per_table;

                    if start_date.is_empty() || end_date.is_empty() {
                        warn!("search_topic_by_date に start_date/end_date が必要、スキップ");
                        continue;
                    }
                    self.search_agency
                        .search_topic_by_date(keyword, start_date, end_date, limit_per_table)
                        .await
                }
                "get_comments_for_topic" => {
                    let limit = self.config.default_get_comments_for_topic_limit
                        / optimized.optimized_keywords.len().max(1);
                    let limit = limit.max(50);
                    self.search_agency
                        .get_comments_for_topic(keyword, limit)
                        .await
                }
                "search_topic_on_platform" => {
                    let platform = kwargs
                        .get("platform")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let start_date = kwargs.get("start_date").and_then(|v| v.as_str());
                    let end_date = kwargs.get("end_date").and_then(|v| v.as_str());
                    let limit = self.config.default_search_topic_on_platform_limit
                        / optimized.optimized_keywords.len().max(1);
                    let limit = limit.max(30);

                    if platform.is_empty() {
                        warn!("search_topic_on_platform に platform が必要、スキップ");
                        continue;
                    }
                    self.search_agency
                        .search_topic_on_platform(
                            platform,
                            keyword,
                            start_date,
                            end_date,
                            limit,
                        )
                        .await
                }
                _ => {
                    warn!("  未知の検索ツール: {}、デフォルトのグローバル検索を使用", tool_name);
                    self.search_agency
                        .search_topic_globally(
                            keyword,
                            self.config.default_search_topic_globally_limit_per_table,
                        )
                        .await
                }
            };

            if !response.results.is_empty() {
                info!("     找到 {} 条结果", response.results.len());
                all_results.extend(response.results);
            } else {
                info!("     未找到结果");
            }
        }

        // 去重
        let unique_results = Self::deduplicate_results(all_results);
        let total_count = unique_results.len();
        info!(
            "  总计找到结果，去重后 {} 条",
            total_count
        );

        // クラスタリング
        let final_results = if ENABLE_CLUSTERING {
            Self::cluster_and_sample_results(
                unique_results,
                MAX_CLUSTERED_RESULTS,
                RESULTS_PER_CLUSTER,
            )
        } else {
            unique_results
        };

        let mut params = HashMap::new();
        params.insert(
            "original_query".to_string(),
            Value::String(query.to_string()),
        );
        params.insert(
            "optimized_keywords".to_string(),
            serde_json::json!(optimized.optimized_keywords),
        );
        params.insert(
            "optimization_reasoning".to_string(),
            Value::String(optimized.reasoning),
        );
        for (k, v) in kwargs {
            params.insert(k.clone(), v.clone());
        }

        let results_count = final_results.len();
        DBResponse {
            tool_name: format!("{}_optimized", tool_name),
            parameters: params,
            results: final_results,
            results_count,
            error_message: None,
            metadata: HashMap::new(),
        }
    }

    /// 深度研究を実行
    pub async fn research(&mut self, query: &str, save_report: bool) -> Result<String> {
        info!("\n{}", "=".repeat(60));
        info!("开始深度研究: {}", query);
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

    /// 検索結果を DB レスポンスから Value ベクトルに変換
    fn convert_db_results(&self, response: &DBResponse) -> Vec<Value> {
        let max_results = if self.config.max_search_results_for_llm > 0 {
            response
                .results
                .len()
                .min(self.config.max_search_results_for_llm)
        } else {
            response.results.len()
        };

        response.results[..max_results]
            .iter()
            .map(|result| {
                serde_json::json!({
                    "title": result.title_or_content,
                    "url": result.url.as_deref().unwrap_or(""),
                    "content": result.title_or_content,
                    "score": result.hotness_score,
                    "raw_content": result.title_or_content,
                    "published_date": result.publish_time,
                    "platform": result.platform,
                    "content_type": result.content_type,
                    "author": result.author_nickname,
                    "engagement": result.engagement,
                })
            })
            .collect()
    }

    /// 検索パラメータを構築
    fn build_search_kwargs(
        &self,
        search_output: &Value,
        search_tool: &str,
    ) -> HashMap<String, Value> {
        let mut kwargs = HashMap::new();

        // 日付パラメータ
        if let Some(sd) = search_output.get("start_date").and_then(|v| v.as_str()) {
            kwargs.insert("start_date".to_string(), Value::String(sd.to_string()));
        }
        if let Some(ed) = search_output.get("end_date").and_then(|v| v.as_str()) {
            kwargs.insert("end_date".to_string(), Value::String(ed.to_string()));
        }

        // プラットフォームパラメータ
        if let Some(platform) = search_output.get("platform").and_then(|v| v.as_str()) {
            kwargs.insert("platform".to_string(), Value::String(platform.to_string()));
        }

        // 時間期間パラメータ
        if search_tool == "search_hot_content" {
            let time_period = search_output
                .get("time_period")
                .and_then(|v| v.as_str())
                .unwrap_or("week");
            kwargs.insert(
                "time_period".to_string(),
                Value::String(time_period.to_string()),
            );
            kwargs.insert(
                "limit".to_string(),
                serde_json::json!(self.config.default_search_hot_content_limit),
            );
        }

        kwargs
    }

    /// 検索ツール選択を検証・修正
    fn validate_search_tool(&self, search_output: &Value, mut tool: String) -> String {
        // search_topic_by_date の日付検証
        if tool == "search_topic_by_date" {
            let sd = search_output.get("start_date").and_then(|v| v.as_str());
            let ed = search_output.get("end_date").and_then(|v| v.as_str());

            match (sd, ed) {
                (Some(sd_val), Some(ed_val)) => {
                    if !Self::validate_date_format(sd_val) || !Self::validate_date_format(ed_val) {
                        info!("  日付フォーマットエラー、グローバル検索に変更");
                        tool = "search_topic_globally".to_string();
                    } else {
                        info!("  時間範囲: {} から {}", sd_val, ed_val);
                    }
                }
                _ => {
                    info!("  search_topic_by_date に時間パラメータが不足、グローバル検索に変更");
                    tool = "search_topic_globally".to_string();
                }
            }
        }

        // search_topic_on_platform のプラットフォーム検証
        if tool == "search_topic_on_platform" {
            let platform = search_output.get("platform").and_then(|v| v.as_str());
            if platform.is_none() || platform.unwrap().is_empty() {
                warn!("  search_topic_on_platform にプラットフォームが不足、グローバル検索に変更");
                tool = "search_topic_globally".to_string();
            } else {
                info!("  指定プラットフォーム: {}", platform.unwrap());
            }
        }

        tool
    }

    /// 初回検索とサマリーを実行
    async fn initial_search_and_summary(&mut self, paragraph_index: usize) -> Result<()> {
        let title = self.state.paragraphs[paragraph_index].title.clone();
        let content = self.state.paragraphs[paragraph_index].content.clone();

        let search_input = serde_json::json!({
            "title": title,
            "content": content,
        });

        info!("  - 検索クエリを生成中...");
        let search_output: Value = self.first_search_node.run(&search_input).await?;
        let search_query = search_output
            .get("search_query")
            .and_then(|v: &Value| v.as_str())
            .unwrap_or("相关主题研究")
            .to_string();
        let search_tool = search_output
            .get("search_tool")
            .and_then(|v: &Value| v.as_str())
            .unwrap_or("search_topic_globally")
            .to_string();
        let reasoning = search_output
            .get("reasoning")
            .and_then(|v: &Value| v.as_str())
            .unwrap_or("");

        info!("  - 検索クエリ: {}", search_query);
        info!("  - 選択されたツール: {}", search_tool);
        info!("  - 推論: {}", reasoning);

        // ツール検証
        let search_tool = self.validate_search_tool(&search_output, search_tool);

        // パラメータ構築
        let kwargs = self.build_search_kwargs(&search_output, &search_tool);

        // 検索実行
        info!("  - データベースクエリを実行中...");
        let search_response = self
            .execute_search_tool(&search_tool, &search_query, &kwargs)
            .await;

        let search_results = self.convert_db_results(&search_response);

        if !search_results.is_empty() {
            let mut msg = format!("  - {} 件の検索結果を発見", search_results.len());
            for (j, result) in search_results.iter().enumerate().take(5) {
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
                msg.push_str(&format!(
                    "\n    {}. {}...{}",
                    j + 1,
                    title_preview,
                    date_info
                ));
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
                self.config.max_content_length,
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

            let reflection_input = serde_json::json!({
                "title": title,
                "content": content,
                "paragraph_latest_state": latest_summary,
            });

            let reflection_output: Value = self.reflection_node.run(&reflection_input).await?;
            let search_query = reflection_output
                .get("search_query")
                .and_then(|v: &Value| v.as_str())
                .unwrap_or("深度研究补充信息")
                .to_string();
            let search_tool = reflection_output
                .get("search_tool")
                .and_then(|v: &Value| v.as_str())
                .unwrap_or("search_topic_globally")
                .to_string();
            let reasoning = reflection_output
                .get("reasoning")
                .and_then(|v: &Value| v.as_str())
                .unwrap_or("");

            info!("    リフレクションクエリ: {}", search_query);
            info!("    選択されたツール: {}", search_tool);
            info!("    リフレクション推論: {}", reasoning);

            // ツール検証
            let search_tool = self.validate_search_tool(&reflection_output, search_tool);

            // パラメータ構築
            let kwargs = self.build_search_kwargs(&reflection_output, &search_tool);

            // 検索実行
            let search_response = self
                .execute_search_tool(&search_tool, &search_query, &kwargs)
                .await;

            let search_results = self.convert_db_results(&search_response);

            if !search_results.is_empty() {
                info!(
                    "    {} 件のリフレクション検索結果を発見",
                    search_results.len()
                );
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
                    self.config.max_content_length,
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
            Ok(val) => {
                let val: Value = val;
                val.as_str().unwrap_or("").to_string()
            }
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

        let filename = format!("insight_report_{}_{}.md", query_safe, timestamp);
        let filepath = Path::new(&self.config.output_dir).join(&filename);

        fs::write(&filepath, report_content).context("レポートの保存に失敗")?;
        info!("レポートを保存しました: {}", filepath.display());

        // 状態を保存
        if self.config.save_intermediate_states {
            let state_filename = format!("insight_state_{}_{}.json", query_safe, timestamp);
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
    #[allow(dead_code)]
    pub fn load_state(&mut self, filepath: &str) -> Result<()> {
        self.state = State::load_from_file(filepath)?;
        info!("状態を {} から読み込みました", filepath);
        Ok(())
    }

    /// 状態をファイルに保存
    #[allow(dead_code)]
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
