//! レポート構造生成ノード (InsightEngine 版)

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use tracing::{error, info, warn};

use bettafish_common::text_processing::{
    clean_json_tags, extract_clean_response, fix_incomplete_json, remove_reasoning_from_output,
};
use bettafish_llm::{InvokeOptions, LLMClient};

use super::base::{Node, StateMutationNode};
use crate::prompts::system_prompt_report_structure;
use crate::state::State;

/// レポート構造生成ノード
pub struct ReportStructureNode {
    llm_client: LLMClient,
    query: String,
}

impl ReportStructureNode {
    pub fn new(llm_client: LLMClient, query: &str) -> Self {
        Self {
            llm_client,
            query: query.to_string(),
        }
    }

    /// デフォルトのレポート構造を生成
    fn generate_default_structure() -> Vec<Value> {
        info!("デフォルトレポート構造を生成");
        vec![
            serde_json::json!({
                "title": "舆情事件概述与背景",
                "content": "全面梳理事件起因、发展脉络、关键节点和涉及的各方立场"
            }),
            serde_json::json!({
                "title": "公众情感与观点分析",
                "content": "深入分析公众的情感倾向、主要观点分布、争议焦点和价值观冲突"
            }),
        ]
    }
}

#[async_trait]
impl Node for ReportStructureNode {
    fn name(&self) -> &str {
        "ReportStructureNode"
    }

    async fn run(&self, _input_data: &Value) -> Result<Value> {
        info!("クエリに対するレポート構造を生成中: {}", self.query);

        let system_prompt = system_prompt_report_structure();
        let response = self
            .llm_client
            .stream_invoke_to_string(&system_prompt, &self.query, &InvokeOptions::default())
            .await?;

        let cleaned = remove_reasoning_from_output(&response);
        let cleaned = clean_json_tags(&cleaned);

        info!("クリーニング後の出力: {}", cleaned);

        let report_structure: Vec<Value> = match serde_json::from_str(&cleaned) {
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
                                error!("JSON 修復失敗、デフォルト構造を使用");
                                return Ok(Value::Array(Self::generate_default_structure()));
                            }
                        }
                    } else {
                        error!("JSON 修復不可、デフォルト構造を使用");
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
                    warn!(
                        "段落 {} にタイトルまたはコンテンツがありません、スキップ",
                        i + 1
                    );
                    continue;
                }

                validated.push(serde_json::json!({
                    "title": title,
                    "content": content,
                }));
            } else {
                warn!("段落 {} は辞書形式ではありません、スキップ", i + 1);
            }
        }

        if validated.is_empty() {
            warn!("有効な段落構造がありません、デフォルト構造を使用");
            return Ok(Value::Array(Self::generate_default_structure()));
        }

        info!("{} 個の段落構造の検証に成功", validated.len());
        Ok(Value::Array(validated))
    }
}

#[async_trait]
impl StateMutationNode for ReportStructureNode {
    async fn mutate_state(
        &self,
        _input_data: &Value,
        state: &mut State,
        _paragraph_index: usize,
    ) -> Result<()> {
        let report_structure = self.run(&Value::Null).await?;

        state.query = self.query.clone();
        if state.report_title.is_empty() {
            state.report_title = format!("关于'{}'的深度舆情分析报告", self.query);
        }

        if let Some(paragraphs) = report_structure.as_array() {
            for p in paragraphs {
                let title = p.get("title").and_then(|v| v.as_str()).unwrap_or("");
                let content = p.get("content").and_then(|v| v.as_str()).unwrap_or("");
                state.add_paragraph(title, content);
            }
            info!("{} 個の段落を状態に追加しました", paragraphs.len());
        }

        Ok(())
    }
}
