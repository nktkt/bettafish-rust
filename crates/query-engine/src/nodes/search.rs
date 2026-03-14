//! 検索ノード実装
//!
//! Python の search_node.py の Rust 実装。
//! 検索クエリとリフレクションクエリの生成を担当。

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use tracing::{error, info, warn};

use bettafish_common::text_processing::{
    clean_json_tags, extract_clean_response, fix_incomplete_json, remove_reasoning_from_output,
};
use bettafish_llm::{InvokeOptions, LLMClient};

use super::base::Node;
use crate::prompts::{system_prompt_first_search, system_prompt_reflection};

/// LLM 出力から検索クエリと推論を抽出する共通処理
fn process_search_output(output: &str, default_query: &str, default_reasoning: &str) -> Value {
    let cleaned = remove_reasoning_from_output(output);
    let cleaned = clean_json_tags(&cleaned);
    info!("クリーニング後の出力: {}", cleaned);

    let result: Value = match serde_json::from_str(&cleaned) {
        Ok(v) => {
            info!("JSON 解析成功");
            v
        }
        Err(e) => {
            error!("JSON 解析失敗: {}", e);
            let extracted = extract_clean_response(&cleaned);
            if extracted.get("error").is_some() {
                if let Some(fixed) = fix_incomplete_json(&cleaned) {
                    match serde_json::from_str(&fixed) {
                        Ok(v) => {
                            info!("JSON 修復成功");
                            v
                        }
                        Err(_) => {
                            error!("JSON 修復失敗");
                            return serde_json::json!({
                                "search_query": default_query,
                                "reasoning": default_reasoning,
                            });
                        }
                    }
                } else {
                    error!("JSON 修復不可、デフォルトクエリを使用");
                    return serde_json::json!({
                        "search_query": default_query,
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
        warn!("検索クエリが見つかりません、デフォルトクエリを使用");
        return serde_json::json!({
            "search_query": default_query,
            "reasoning": default_reasoning,
        });
    }

    // search_tool フィールドも含める
    let mut output_obj = serde_json::json!({
        "search_query": search_query,
        "reasoning": result.get("reasoning").and_then(|v| v.as_str()).unwrap_or(""),
    });

    if let Some(tool) = result.get("search_tool").and_then(|v| v.as_str()) {
        output_obj["search_tool"] = Value::String(tool.to_string());
    }
    if let Some(sd) = result.get("start_date").and_then(|v| v.as_str()) {
        output_obj["start_date"] = Value::String(sd.to_string());
    }
    if let Some(ed) = result.get("end_date").and_then(|v| v.as_str()) {
        output_obj["end_date"] = Value::String(ed.to_string());
    }

    output_obj
}

/// 段落の初回検索クエリを生成するノード
pub struct FirstSearchNode {
    llm_client: LLMClient,
}

impl FirstSearchNode {
    pub fn new(llm_client: LLMClient) -> Self {
        Self { llm_client }
    }
}

#[async_trait]
impl Node for FirstSearchNode {
    fn name(&self) -> &str {
        "FirstSearchNode"
    }

    fn validate_input(&self, input_data: &Value) -> bool {
        input_data.get("title").is_some() && input_data.get("content").is_some()
    }

    async fn run(&self, input_data: &Value) -> Result<Value> {
        if !self.validate_input(input_data) {
            anyhow::bail!("入力データのフォーマットエラー、title と content フィールドが必要");
        }

        let message = serde_json::to_string(input_data)?;
        info!("初回検索クエリを生成中");

        let system_prompt = system_prompt_first_search();
        let response = self
            .llm_client
            .stream_invoke_to_string(&system_prompt, &message, &InvokeOptions::default())
            .await?;

        let result = process_search_output(
            &response,
            "相关主题研究",
            "由于解析失败，使用默认搜索查询",
        );

        info!(
            "検索クエリ生成: {}",
            result.get("search_query").and_then(|v| v.as_str()).unwrap_or("N/A")
        );

        Ok(result)
    }
}

/// リフレクションによる新規検索クエリ生成ノード
pub struct ReflectionNode {
    llm_client: LLMClient,
}

impl ReflectionNode {
    pub fn new(llm_client: LLMClient) -> Self {
        Self { llm_client }
    }
}

#[async_trait]
impl Node for ReflectionNode {
    fn name(&self) -> &str {
        "ReflectionNode"
    }

    fn validate_input(&self, input_data: &Value) -> bool {
        input_data.get("title").is_some()
            && input_data.get("content").is_some()
            && input_data.get("paragraph_latest_state").is_some()
    }

    async fn run(&self, input_data: &Value) -> Result<Value> {
        if !self.validate_input(input_data) {
            anyhow::bail!("入力データのフォーマットエラー、title、content、paragraph_latest_state フィールドが必要");
        }

        let message = serde_json::to_string(input_data)?;
        info!("リフレクション中、新しい検索クエリを生成");

        let system_prompt = system_prompt_reflection();
        let response = self
            .llm_client
            .stream_invoke_to_string(&system_prompt, &message, &InvokeOptions::default())
            .await?;

        let result = process_search_output(
            &response,
            "深度研究补充信息",
            "由于解析失败，使用默认反思搜索查询",
        );

        info!(
            "リフレクション検索クエリ生成: {}",
            result.get("search_query").and_then(|v| v.as_str()).unwrap_or("N/A")
        );

        Ok(result)
    }
}
