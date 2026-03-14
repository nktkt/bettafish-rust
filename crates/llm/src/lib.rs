//! BettaFish LLM クライアント
//!
//! Python の llms/base.py の Rust 実装。
//! OpenAI 互換 API をサポートする統一 LLM クライアント。

use anyhow::{Context, Result};
use chrono::Local;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
#[allow(unused_imports)]
use tracing::info;

use bettafish_common::retry::{llm_retry_config, with_retry};

/// OpenAI 互換 LLM クライアント
#[derive(Debug, Clone)]
pub struct LLMClient {
    api_key: String,
    base_url: String,
    pub model_name: String,
    #[allow(dead_code)]
    timeout: Duration,
    client: Client,
}

/// チャットメッセージ
#[derive(Debug, Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// チャット完了リクエスト
#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

/// チャット完了レスポンス
#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: Option<MessageContent>,
    delta: Option<DeltaContent>,
}

#[derive(Debug, Deserialize)]
struct MessageContent {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DeltaContent {
    content: Option<String>,
}

/// ストリーミングチャンクレスポンス
#[derive(Debug, Deserialize)]
struct StreamChunkResponse {
    choices: Vec<Choice>,
}

/// LLM 呼び出しのオプションパラメータ
#[derive(Debug, Clone, Default)]
pub struct InvokeOptions {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub frequency_penalty: Option<f64>,
}

impl LLMClient {
    /// 新しい LLM クライアントを作成
    pub fn new(api_key: &str, model_name: &str, base_url: Option<&str>) -> Result<Self> {
        if api_key.is_empty() {
            anyhow::bail!("Query Engine LLM API key is required.");
        }
        if model_name.is_empty() {
            anyhow::bail!("Query Engine model name is required.");
        }

        let timeout_str = std::env::var("LLM_REQUEST_TIMEOUT")
            .or_else(|_| std::env::var("QUERY_ENGINE_REQUEST_TIMEOUT"))
            .unwrap_or_else(|_| "1800".to_string());
        let timeout_secs: f64 = timeout_str.parse().unwrap_or(1800.0);

        let client = Client::builder()
            .timeout(Duration::from_secs_f64(timeout_secs))
            .build()
            .context("HTTP クライアントの作成に失敗")?;

        let base_url = base_url
            .filter(|u| !u.is_empty())
            .unwrap_or("https://api.openai.com/v1")
            .trim_end_matches('/')
            .to_string();

        Ok(Self {
            api_key: api_key.to_string(),
            base_url,
            model_name: model_name.to_string(),
            timeout: Duration::from_secs_f64(timeout_secs),
            client,
        })
    }

    /// 時刻プレフィックスを生成
    fn time_prefix() -> String {
        let now = Local::now();
        format!("今天的实际时间是{}年{}月{}日{}时{}分",
            now.format("%Y"), now.format("%m"), now.format("%d"),
            now.format("%H"), now.format("%M"))
    }

    /// メッセージを構築
    fn build_messages(&self, system_prompt: &str, user_prompt: &str) -> Vec<ChatMessage> {
        let time_prefix = Self::time_prefix();
        let user_content = if user_prompt.is_empty() {
            time_prefix
        } else {
            format!("{}\n{}", time_prefix, user_prompt)
        };

        vec![
            ChatMessage {
                role: "system".to_string(),
                content: system_prompt.to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: user_content,
            },
        ]
    }

    /// 同期（非ストリーム）LLM 呼び出し（リトライ付き）
    pub async fn invoke(
        &self,
        system_prompt: &str,
        user_prompt: &str,
        options: &InvokeOptions,
    ) -> Result<String> {
        let config = llm_retry_config();
        let messages = self.build_messages(system_prompt, user_prompt);
        let opts = options.clone();

        with_retry(&config, "LLMClient::invoke", || {
            let messages = messages.clone();
            let opts = opts.clone();
            async move {
                self.invoke_internal(&messages, &opts, false).await
            }
        })
        .await
    }

    /// ストリーミング LLM 呼び出し → 完全文字列を返却（リトライ付き）
    pub async fn stream_invoke_to_string(
        &self,
        system_prompt: &str,
        user_prompt: &str,
        options: &InvokeOptions,
    ) -> Result<String> {
        let config = llm_retry_config();
        let messages = self.build_messages(system_prompt, user_prompt);
        let opts = options.clone();

        with_retry(&config, "LLMClient::stream_invoke_to_string", || {
            let messages = messages.clone();
            let opts = opts.clone();
            async move {
                self.stream_internal(&messages, &opts).await
            }
        })
        .await
    }

    /// 内部: 非ストリーム呼び出し
    async fn invoke_internal(
        &self,
        messages: &[ChatMessage],
        options: &InvokeOptions,
        _stream: bool,
    ) -> Result<String> {
        let url = format!("{}/chat/completions", self.base_url);

        let request = ChatCompletionRequest {
            model: self.model_name.clone(),
            messages: messages.to_vec(),
            temperature: options.temperature,
            top_p: options.top_p,
            presence_penalty: options.presence_penalty,
            frequency_penalty: options.frequency_penalty,
            stream: Some(false),
        };

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .context("LLM API リクエストの送信に失敗")?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("LLM API エラー ({}): {}", status, body);
        }

        let resp: ChatCompletionResponse = response
            .json()
            .await
            .context("LLM レスポンスの解析に失敗")?;

        let content = resp
            .choices
            .first()
            .and_then(|c| c.message.as_ref())
            .and_then(|m| m.content.as_ref())
            .map(|s| s.trim().to_string())
            .unwrap_or_default();

        Ok(content)
    }

    /// 内部: ストリーミング呼び出し
    async fn stream_internal(
        &self,
        messages: &[ChatMessage],
        options: &InvokeOptions,
    ) -> Result<String> {
        let url = format!("{}/chat/completions", self.base_url);

        let request = ChatCompletionRequest {
            model: self.model_name.clone(),
            messages: messages.to_vec(),
            temperature: options.temperature,
            top_p: options.top_p,
            presence_penalty: options.presence_penalty,
            frequency_penalty: options.frequency_penalty,
            stream: Some(true),
        };

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .context("ストリーミングリクエストの送信に失敗")?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("LLM API エラー ({}): {}", status, body);
        }

        let mut byte_chunks: Vec<Vec<u8>> = Vec::new();
        let mut stream = response.bytes_stream();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.context("ストリームチャンクの読み取りに失敗")?;
            let chunk_str = String::from_utf8_lossy(&chunk);

            // SSE フォーマットの各行を処理
            for line in chunk_str.lines() {
                let line = line.trim();
                if line.is_empty() || line == "data: [DONE]" {
                    continue;
                }
                if let Some(data) = line.strip_prefix("data: ") {
                    if let Ok(parsed) = serde_json::from_str::<StreamChunkResponse>(data) {
                        for choice in &parsed.choices {
                            if let Some(delta) = &choice.delta {
                                if let Some(content) = &delta.content {
                                    byte_chunks.push(content.as_bytes().to_vec());
                                }
                            }
                        }
                    }
                }
            }
        }

        // 全バイトチャンクを結合して一括デコード（UTF-8 マルチバイト文字の切断防止）
        let all_bytes: Vec<u8> = byte_chunks.into_iter().flatten().collect();
        let result = String::from_utf8(all_bytes)
            .unwrap_or_else(|e| String::from_utf8_lossy(&e.into_bytes()).to_string());

        Ok(result.trim().to_string())
    }

    /// レスポンスの検証
    pub fn validate_response(response: Option<&str>) -> String {
        response.map(|s| s.trim().to_string()).unwrap_or_default()
    }

    /// モデル情報を取得
    pub fn get_model_info(&self) -> serde_json::Value {
        serde_json::json!({
            "provider": self.model_name,
            "model": self.model_name,
            "api_base": self.base_url,
        })
    }
}
