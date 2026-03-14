//! レポートフォーマットノード
//!
//! Python の formatting_node.py の Rust 実装。
//! 最終研究結果を美しい Markdown レポートにフォーマット。

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use tracing::info;

use bettafish_common::text_processing::{clean_markdown_tags, remove_reasoning_from_output};
use bettafish_llm::{InvokeOptions, LLMClient};

use super::base::Node;
use crate::prompts::system_prompt_report_formatting;

/// 最終レポートのフォーマットノード
pub struct ReportFormattingNode {
    llm_client: LLMClient,
}

impl ReportFormattingNode {
    pub fn new(llm_client: LLMClient) -> Self {
        Self { llm_client }
    }

    /// 手動フォーマット（バックアップメソッド）
    pub fn format_report_manually(
        &self,
        paragraphs_data: &[Value],
        report_title: &str,
    ) -> String {
        info!("手動フォーマットメソッドを使用");

        let title = if report_title.is_empty() {
            "深度研究报告"
        } else {
            report_title
        };

        let mut lines = vec![
            format!("# {}", title),
            String::new(),
            "---".to_string(),
            String::new(),
        ];

        for (i, paragraph) in paragraphs_data.iter().enumerate() {
            let default_title = format!("段落 {}", i + 1);
            let p_title = paragraph
                .get("title")
                .and_then(|v| v.as_str())
                .unwrap_or(&default_title);
            let content = paragraph
                .get("paragraph_latest_state")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            if !content.is_empty() {
                lines.push(format!("## {}", p_title));
                lines.push(String::new());
                lines.push(content.to_string());
                lines.push(String::new());
                lines.push("---".to_string());
                lines.push(String::new());
            }
        }

        if paragraphs_data.len() > 1 {
            lines.push("## 结论".to_string());
            lines.push(String::new());
            lines.push(
                "本报告通过深度搜索和研究，对相关主题进行了全面分析。\
                 以上各个方面的内容为理解该主题提供了重要参考。"
                    .to_string(),
            );
            lines.push(String::new());
        }

        lines.join("\n")
    }
}

#[async_trait]
impl Node for ReportFormattingNode {
    fn name(&self) -> &str {
        "ReportFormattingNode"
    }

    fn validate_input(&self, input_data: &Value) -> bool {
        if let Some(arr) = input_data.as_array() {
            arr.iter().all(|item| {
                item.get("title").is_some() && item.get("paragraph_latest_state").is_some()
            })
        } else {
            false
        }
    }

    async fn run(&self, input_data: &Value) -> Result<Value> {
        if !self.validate_input(input_data) {
            anyhow::bail!(
                "入力データのフォーマットエラー、title と paragraph_latest_state を含むリストが必要"
            );
        }

        let message = serde_json::to_string(input_data)?;
        info!("最終レポートをフォーマット中");

        let system_prompt = system_prompt_report_formatting();
        let response = self
            .llm_client
            .stream_invoke_to_string(&system_prompt, &message, &InvokeOptions::default())
            .await?;

        // レスポンスを処理
        let cleaned = remove_reasoning_from_output(&response);
        let cleaned = clean_markdown_tags(&cleaned);

        let result = if cleaned.trim().is_empty() {
            "# 报告生成失败\n\n无法生成有效的报告内容。".to_string()
        } else if !cleaned.trim().starts_with('#') {
            format!("# 深度研究报告\n\n{}", cleaned.trim())
        } else {
            cleaned.trim().to_string()
        };

        info!("フォーマット済みレポートの生成に成功");
        Ok(Value::String(result))
    }
}
