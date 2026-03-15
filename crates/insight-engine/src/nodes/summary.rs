//! サマリーノード実装 (InsightEngine 版)

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use tracing::{error, info};

use bettafish_common::text_processing::{
    clean_json_tags, fix_incomplete_json, remove_reasoning_from_output,
};
use bettafish_llm::{InvokeOptions, LLMClient};

use super::base::{Node, StateMutationNode};
use crate::prompts::{system_prompt_first_summary, system_prompt_reflection_summary};
use crate::state::State;

/// LLM 出力からサマリーコンテンツを抽出する共通処理
fn extract_summary_content(output: &str, key: &str, fallback_msg: &str) -> String {
    let cleaned = remove_reasoning_from_output(output);
    let cleaned = clean_json_tags(&cleaned);
    info!("クリーニング後の出力: {}", cleaned);

    match serde_json::from_str::<Value>(&cleaned) {
        Ok(result) => {
            info!("JSON 解析成功");
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
                        info!("JSON 修復成功");
                        if let Some(content) = result.get(key).and_then(|v| v.as_str()) {
                            if !content.is_empty() {
                                return content.to_string();
                            }
                        }
                        cleaned
                    }
                    Err(_) => {
                        error!("JSON 修復失敗、クリーニング後のテキストを使用");
                        if cleaned.is_empty() {
                            fallback_msg.to_string()
                        } else {
                            cleaned
                        }
                    }
                }
            } else {
                error!("JSON 修復不可、クリーニング後のテキストを使用");
                if cleaned.is_empty() {
                    fallback_msg.to_string()
                } else {
                    cleaned
                }
            }
        }
    }
}

/// 検索結果に基づく段落の初回サマリー生成ノード
pub struct FirstSummaryNode {
    llm_client: LLMClient,
}

impl FirstSummaryNode {
    pub fn new(llm_client: LLMClient) -> Self {
        Self { llm_client }
    }
}

#[async_trait]
impl Node for FirstSummaryNode {
    fn name(&self) -> &str {
        "FirstSummaryNode"
    }

    fn validate_input(&self, input_data: &Value) -> bool {
        ["title", "content", "search_query", "search_results"]
            .iter()
            .all(|field| input_data.get(*field).is_some())
    }

    async fn run(&self, input_data: &Value) -> Result<Value> {
        if !self.validate_input(input_data) {
            anyhow::bail!("入力データのフォーマットエラー");
        }

        let message = serde_json::to_string(input_data)?;
        info!("初回段落サマリーを生成中");

        let system_prompt = system_prompt_first_summary();
        let response = self
            .llm_client
            .stream_invoke_to_string(&system_prompt, &message, &InvokeOptions::default())
            .await?;

        let content = extract_summary_content(
            &response,
            "paragraph_latest_state",
            "段落サマリー生成失敗",
        );

        info!("初回段落サマリーの生成に成功");
        Ok(Value::String(content))
    }
}

#[async_trait]
impl StateMutationNode for FirstSummaryNode {
    async fn mutate_state(
        &self,
        input_data: &Value,
        state: &mut State,
        paragraph_index: usize,
    ) -> Result<()> {
        let summary = self.run(input_data).await?;
        let summary_str = summary.as_str().unwrap_or("");

        if paragraph_index < state.paragraphs.len() {
            state.paragraphs[paragraph_index].research.latest_summary = summary_str.to_string();
            info!("段落 {} の初回サマリーを更新しました", paragraph_index);
        } else {
            anyhow::bail!("段落インデックス {} が範囲外です", paragraph_index);
        }

        state.update_timestamp();
        Ok(())
    }
}

/// リフレクション検索結果に基づく段落サマリー更新ノード
pub struct ReflectionSummaryNode {
    llm_client: LLMClient,
}

impl ReflectionSummaryNode {
    pub fn new(llm_client: LLMClient) -> Self {
        Self { llm_client }
    }
}

#[async_trait]
impl Node for ReflectionSummaryNode {
    fn name(&self) -> &str {
        "ReflectionSummaryNode"
    }

    fn validate_input(&self, input_data: &Value) -> bool {
        [
            "title",
            "content",
            "search_query",
            "search_results",
            "paragraph_latest_state",
        ]
        .iter()
        .all(|field| input_data.get(*field).is_some())
    }

    async fn run(&self, input_data: &Value) -> Result<Value> {
        if !self.validate_input(input_data) {
            anyhow::bail!("入力データのフォーマットエラー");
        }

        let message = serde_json::to_string(input_data)?;
        info!("リフレクションサマリーを生成中");

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

        info!("リフレクションサマリーの生成に成功");
        Ok(Value::String(content))
    }
}

#[async_trait]
impl StateMutationNode for ReflectionSummaryNode {
    async fn mutate_state(
        &self,
        input_data: &Value,
        state: &mut State,
        paragraph_index: usize,
    ) -> Result<()> {
        let updated_summary = self.run(input_data).await?;
        let summary_str = updated_summary.as_str().unwrap_or("");

        if paragraph_index < state.paragraphs.len() {
            state.paragraphs[paragraph_index].research.latest_summary = summary_str.to_string();
            state.paragraphs[paragraph_index]
                .research
                .increment_reflection();
            info!(
                "段落 {} のリフレクションサマリーを更新しました",
                paragraph_index
            );
        } else {
            anyhow::bail!("段落インデックス {} が範囲外です", paragraph_index);
        }

        state.update_timestamp();
        Ok(())
    }
}
