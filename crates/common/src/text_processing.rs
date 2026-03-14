//! テキスト処理ユーティリティ
//!
//! Python の text_processing.py の Rust 実装。
//! LLM 出力のクリーニング、JSON 解析・修復など。

use regex::Regex;
use serde_json::Value;
use std::collections::HashMap;

/// JSON タグの除去
///
/// ```json ... ``` マークダウンブロックを除去する。
pub fn clean_json_tags(text: &str) -> String {
    let re1 = Regex::new(r"```json\s*").unwrap();
    let re2 = Regex::new(r"```\s*$").unwrap();
    let re3 = Regex::new(r"```").unwrap();

    let text = re1.replace_all(text, "");
    let text = re2.replace_all(&text, "");
    let text = re3.replace_all(&text, "");

    text.trim().to_string()
}

/// Markdown タグの除去
pub fn clean_markdown_tags(text: &str) -> String {
    let re1 = Regex::new(r"```markdown\s*").unwrap();
    let re2 = Regex::new(r"```\s*$").unwrap();
    let re3 = Regex::new(r"```").unwrap();

    let text = re1.replace_all(text, "");
    let text = re2.replace_all(&text, "");
    let text = re3.replace_all(&text, "");

    text.trim().to_string()
}

/// LLM 出力から推論プロセス部分を除去
///
/// JSON の開始位置（`{` または `[`）を見つけてそれ以降を返す。
pub fn remove_reasoning_from_output(text: &str) -> String {
    // 最初の { または [ を探す
    for (i, ch) in text.char_indices() {
        if ch == '{' || ch == '[' {
            return text[i..].trim().to_string();
        }
    }

    // JSON マーカーが見つからない場合、既知のパターンを除去
    let patterns = [
        r"(?:reasoning|推理|思考|分析)[:：]\s*.*?(?=\{|\[)",
        r"(?:explanation|解释|说明)[:：]\s*.*?(?=\{|\[)",
        r"^.*?(?=\{|\[)",
    ];

    let mut result = text.to_string();
    for pattern in &patterns {
        if let Ok(re) = Regex::new(pattern) {
            result = re.replace_all(&result, "").to_string();
        }
    }

    result.trim().to_string()
}

/// レスポンスから JSON コンテンツを抽出・クリーン
pub fn extract_clean_response(text: &str) -> Value {
    let cleaned = clean_json_tags(text);
    let cleaned = remove_reasoning_from_output(&cleaned);

    // 直接パース試行
    if let Ok(val) = serde_json::from_str::<Value>(&cleaned) {
        return val;
    }

    // 不完全 JSON の修復試行
    if let Some(fixed) = fix_incomplete_json(&cleaned) {
        if let Ok(val) = serde_json::from_str::<Value>(&fixed) {
            return val;
        }
    }

    // JSON オブジェクトを検索
    if let Ok(re) = Regex::new(r"\{.*\}") {
        if let Some(m) = re.find(&cleaned) {
            if let Ok(val) = serde_json::from_str::<Value>(m.as_str()) {
                return val;
            }
        }
    }

    // JSON 配列を検索
    if let Ok(re) = Regex::new(r"\[.*\]") {
        if let Some(m) = re.find(&cleaned) {
            if let Ok(val) = serde_json::from_str::<Value>(m.as_str()) {
                return val;
            }
        }
    }

    // 全て失敗した場合
    let truncated: String = cleaned.chars().take(200).collect();
    tracing::error!("JSON レスポンスの解析に失敗: {}...", truncated);
    serde_json::json!({
        "error": "JSON解析失敗",
        "raw_text": cleaned
    })
}

/// 不完全な JSON を修復
pub fn fix_incomplete_json(text: &str) -> Option<String> {
    // 余分なカンマを除去
    let re_trailing_comma_obj = Regex::new(r",\s*\}").unwrap();
    let re_trailing_comma_arr = Regex::new(r",\s*\]").unwrap();
    let mut text = re_trailing_comma_obj.replace_all(text, "}").to_string();
    text = re_trailing_comma_arr.replace_all(&text, "]").to_string();

    // 既に有効な JSON か確認
    if serde_json::from_str::<Value>(&text).is_ok() {
        return Some(text);
    }

    // 括弧の不一致を修復
    let open_braces = text.matches('{').count();
    let close_braces = text.matches('}').count();
    let open_brackets = text.matches('[').count();
    let close_brackets = text.matches(']').count();

    if open_braces > close_braces {
        text.push_str(&"}".repeat(open_braces - close_braces));
    }
    if open_brackets > close_brackets {
        text.push_str(&"]".repeat(open_brackets - close_brackets));
    }

    // 修復後の検証
    if serde_json::from_str::<Value>(&text).is_ok() {
        return Some(text);
    }

    // より積極的な修復
    fix_aggressive_json(&text)
}

/// より積極的な JSON 修復
fn fix_aggressive_json(text: &str) -> Option<String> {
    let re = Regex::new(r"\{[^{}]*\}").unwrap();
    let objects: Vec<&str> = re.find_iter(text).map(|m| m.as_str()).collect();

    if objects.len() >= 2 {
        let joined = objects.join(",");
        let result = format!("[{}]", joined);
        if serde_json::from_str::<Value>(&result).is_ok() {
            return Some(result);
        }
    } else if objects.len() == 1 {
        let result = format!("[{}]", objects[0]);
        if serde_json::from_str::<Value>(&result).is_ok() {
            return Some(result);
        }
    }

    None
}

/// コンテンツを指定長に切り詰め
pub fn truncate_content(content: &str, max_length: usize) -> String {
    if content.len() <= max_length {
        return content.to_string();
    }

    let truncated: String = content.chars().take(max_length).collect();

    // 単語境界で切り詰め試行
    if let Some(last_space) = truncated.rfind(' ') {
        if last_space > max_length * 8 / 10 {
            return format!("{}...", &truncated[..last_space]);
        }
    }

    format!("{}...", truncated)
}

/// 検索結果をプロンプト用にフォーマット
pub fn format_search_results_for_prompt(
    search_results: &[HashMap<String, Value>],
    max_length: usize,
) -> Vec<String> {
    let mut formatted = Vec::new();

    for result in search_results {
        if let Some(content) = result.get("content").and_then(|v| v.as_str()) {
            formatted.push(truncate_content(content, max_length));
        }
    }

    formatted
}

/// JSON スキーマの必須フィールド検証
pub fn validate_json_schema(data: &Value, required_fields: &[&str]) -> bool {
    if let Some(obj) = data.as_object() {
        required_fields.iter().all(|field| obj.contains_key(*field))
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_json_tags() {
        let input = "```json\n{\"key\": \"value\"}\n```";
        let result = clean_json_tags(input);
        assert_eq!(result, "{\"key\": \"value\"}");
    }

    #[test]
    fn test_remove_reasoning() {
        let input = "Let me think about this... {\"key\": \"value\"}";
        let result = remove_reasoning_from_output(input);
        assert_eq!(result, "{\"key\": \"value\"}");
    }

    #[test]
    fn test_truncate_content() {
        let content = "a".repeat(100);
        let result = truncate_content(&content, 50);
        assert!(result.len() <= 53); // 50 + "..."
    }

    #[test]
    fn test_fix_incomplete_json() {
        // 閉じ括弧が不足しているケース
        let input = r#"{"key": "value"}"#;
        let result = fix_incomplete_json(input);
        assert!(result.is_some());

        // 末尾カンマの修復
        let input2 = r#"{"key": "value",}"#;
        let result2 = fix_incomplete_json(input2);
        assert!(result2.is_some());
    }
}
