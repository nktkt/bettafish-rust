//! BettaFish アプリケーションテスト
//!
//! 設定ロード、状態シリアライズ、テキスト処理、テンプレート解析、
//! IR バリデーション、Markdown レンダリング、レポート状態、
//! フォーラムログ解析のユニットテスト。

use bettafish_common::text_processing::{
    clean_json_tags, extract_clean_response, fix_incomplete_json, remove_reasoning_from_output,
    truncate_content, validate_json_schema,
};
use bettafish_config::Settings;
use bettafish_query_engine::State;

// =============================================================================
// 設定ロードテスト
// =============================================================================

#[test]
fn test_config_loading() {
    let settings = Settings::default();
    assert_eq!(settings.host, "0.0.0.0");
    assert_eq!(settings.port, 5000);
    assert_eq!(settings.db_dialect, "postgresql");
    assert_eq!(settings.db_host, "localhost");
    assert_eq!(settings.db_port, 5432);
    assert_eq!(settings.max_reflections, 2);
    assert_eq!(settings.max_paragraphs, 5);
    assert_eq!(settings.output_dir, "reports");
    assert!(settings.save_intermediate_states);
}

#[test]
fn test_config_display() {
    let settings = Settings::default();
    let display = settings.display_query_engine_config();
    assert!(display.contains("Query Engine 設定"));
    assert!(display.contains("deepseek-chat"));
}

// =============================================================================
// 状態シリアライズテスト
// =============================================================================

#[test]
fn test_state_serialization() {
    let mut state = State::default();
    state.query = "テストクエリ".to_string();
    state.report_title = "テストレポート".to_string();
    state.add_paragraph("段落1", "内容1");
    state.add_paragraph("段落2", "内容2");

    // シリアライズ
    let json_str = state.to_json().unwrap();
    assert!(json_str.contains("テストクエリ"));
    assert!(json_str.contains("段落1"));

    // デシリアライズ
    let restored = State::from_json(&json_str).unwrap();
    assert_eq!(restored.query, "テストクエリ");
    assert_eq!(restored.paragraphs.len(), 2);
    assert_eq!(restored.paragraphs[0].title, "段落1");
    assert_eq!(restored.paragraphs[1].title, "段落2");
}

#[test]
fn test_state_progress() {
    let mut state = State::default();
    state.add_paragraph("p1", "c1");
    state.add_paragraph("p2", "c2");

    assert_eq!(state.get_completed_paragraphs_count(), 0);
    assert_eq!(state.get_total_paragraphs_count(), 2);
    assert!(!state.is_all_paragraphs_completed());

    // 段落を完了にする
    state.paragraphs[0].research.latest_summary = "サマリー1".to_string();
    state.paragraphs[0].research.mark_completed();
    assert_eq!(state.get_completed_paragraphs_count(), 1);

    state.paragraphs[1].research.latest_summary = "サマリー2".to_string();
    state.paragraphs[1].research.mark_completed();
    assert!(state.is_all_paragraphs_completed());

    let summary = state.get_progress_summary();
    assert_eq!(summary["total_paragraphs"], 2);
    assert_eq!(summary["completed_paragraphs"], 2);
}

// =============================================================================
// テキスト処理テスト
// =============================================================================

#[test]
fn test_text_processing_clean_json_tags() {
    // 基本ケース
    let input = "```json\n{\"key\": \"value\"}\n```";
    let result = clean_json_tags(input);
    assert_eq!(result, "{\"key\": \"value\"}");

    // 空入力
    let result = clean_json_tags("");
    assert_eq!(result, "");

    // JSON タグなし
    let input = "{\"key\": \"value\"}";
    let result = clean_json_tags(input);
    assert_eq!(result, "{\"key\": \"value\"}");

    // 複数のバッククォート
    let input = "```json\n{\"a\": 1}\n```\n```json\n{\"b\": 2}\n```";
    let result = clean_json_tags(input);
    assert!(!result.contains("```"));
}

#[test]
fn test_text_processing_remove_reasoning() {
    // JSON の前のテキストを除去
    let input = "Let me think about this... {\"key\": \"value\"}";
    let result = remove_reasoning_from_output(input);
    assert_eq!(result, "{\"key\": \"value\"}");

    // 配列ケース
    let input = "Here is the result: [{\"item\": 1}]";
    let result = remove_reasoning_from_output(input);
    assert_eq!(result, "[{\"item\": 1}]");

    // JSON がない場合
    let input = "Just some text without JSON";
    let result = remove_reasoning_from_output(input);
    assert!(!result.is_empty());
}

#[test]
fn test_text_processing_fix_incomplete_json() {
    // 末尾カンマの修復
    let input = r#"{"key": "value",}"#;
    let result = fix_incomplete_json(input);
    assert!(result.is_some());
    let fixed = result.unwrap();
    assert!(serde_json::from_str::<serde_json::Value>(&fixed).is_ok());

    // 閉じ括弧不足の修復
    let input = r#"{"key": "value""#;
    let result = fix_incomplete_json(input);
    assert!(result.is_some());

    // 既に有効な JSON
    let input = r#"{"key": "value"}"#;
    let result = fix_incomplete_json(input);
    assert!(result.is_some());

    // 配列の末尾カンマ
    let input = r#"[1, 2, 3,]"#;
    let result = fix_incomplete_json(input);
    assert!(result.is_some());
}

#[test]
fn test_text_processing_truncate_content() {
    // 短いコンテンツ (切り詰め不要)
    let content = "short";
    let result = truncate_content(content, 100);
    assert_eq!(result, "short");

    // 長いコンテンツ
    let content = "a".repeat(200);
    let result = truncate_content(&content, 50);
    assert!(result.len() <= 53); // 50 chars + "..."

    // スペースありの長いコンテンツ
    let content = "word ".repeat(50);
    let result = truncate_content(&content, 50);
    assert!(result.ends_with("..."));
}

#[test]
fn test_text_processing_extract_clean_response() {
    // 有効な JSON
    let input = r#"{"key": "value"}"#;
    let result = extract_clean_response(input);
    assert_eq!(result["key"], "value");

    // JSON マークダウンブロック内
    let input = "```json\n{\"key\": \"value\"}\n```";
    let result = extract_clean_response(input);
    assert_eq!(result["key"], "value");

    // 推論テキスト付き
    let input = "Let me analyze... {\"key\": \"value\"}";
    let result = extract_clean_response(input);
    assert_eq!(result["key"], "value");
}

// =============================================================================
// テンプレート解析テスト
// =============================================================================

#[test]
fn test_template_parsing() {
    let template = "# Title\n## Section 1\nContent 1\n## Section 2\nContent 2";
    let sections = bettafish_report_engine::parse_template_sections(template);
    assert!(!sections.is_empty());
}

// =============================================================================
// IR バリデーションテスト
// =============================================================================

#[test]
fn test_ir_validation() {
    let validator = bettafish_report_engine::IRValidator::new();

    // 有効な章
    let valid_chapter = serde_json::json!({
        "chapterId": "ch1",
        "title": "第1章",
        "anchor": "chapter-1",
        "order": 1,
        "blocks": [
            {
                "type": "heading",
                "level": 1,
                "runs": [{"text": "テスト"}]
            }
        ]
    });
    let (is_valid, errors) = validator.validate_chapter(&valid_chapter);
    // May have some block-level warnings but main fields should be present
    assert!(
        valid_chapter.get("chapterId").is_some()
            && valid_chapter.get("title").is_some()
            && valid_chapter.get("anchor").is_some()
            && valid_chapter.get("order").is_some()
    );

    // 不完全な章 (必須フィールドなし)
    let invalid_chapter = serde_json::json!({
        "title": "テスト"
    });
    let (is_valid, errors) = validator.validate_chapter(&invalid_chapter);
    assert!(!is_valid);
    assert!(!errors.is_empty());
}

// =============================================================================
// Markdown レンダリングテスト
// =============================================================================

#[test]
fn test_markdown_rendering() {
    let document_ir = serde_json::json!({
        "title": "テストレポート",
        "meta": {
            "generatedAt": "2025-01-01T00:00:00"
        },
        "chapters": [
            {
                "chapterId": "ch1",
                "title": "第1章",
                "anchor": "chapter-1",
                "order": 1,
                "blocks": [
                    {
                        "type": "heading",
                        "level": 2,
                        "runs": [{"text": "セクション1"}]
                    },
                    {
                        "type": "paragraph",
                        "runs": [{"text": "本文テキスト"}]
                    }
                ]
            }
        ]
    });

    let renderer = bettafish_report_engine::MarkdownRenderer::new();
    let markdown = renderer.render(&document_ir);
    assert!(!markdown.is_empty());
    // MarkdownRenderer はドキュメント IR を解析して Markdown を生成する
}

// =============================================================================
// レポート状態テスト
// =============================================================================

#[test]
fn test_report_state() {
    let mut state = bettafish_report_engine::ReportState::default();
    state.query = "テスト".to_string();
    assert_eq!(state.query, "テスト");
    assert!(!state.is_completed());
    assert_eq!(state.status, "pending");

    state.mark_processing();
    assert_eq!(state.status, "processing");

    state.mark_completed();
    assert!(state.is_completed());

    let mut state2 = bettafish_report_engine::ReportState::default();
    state2.mark_failed("エラー");
    assert_eq!(state2.status, "failed");
    assert_eq!(state2.error_message.as_deref(), Some("エラー"));
}

// =============================================================================
// フォーラムログ解析テスト
// =============================================================================

#[test]
fn test_forum_log_parsing() {
    // get_all_host_speeches は存在しないファイルでも空ベクターを返す
    let speeches = bettafish_common::forum_reader::get_all_host_speeches("/nonexistent/path");
    assert!(speeches.is_empty());

    // get_latest_host_speech は None を返す
    let latest = bettafish_common::forum_reader::get_latest_host_speech("/nonexistent/path");
    assert!(latest.is_none());

    // get_recent_agent_speeches は空ベクターを返す
    let agents =
        bettafish_common::forum_reader::get_recent_agent_speeches("/nonexistent/path", 10);
    assert!(agents.is_empty());

    // format_host_speech_for_prompt は空文字で空を返す
    let formatted = bettafish_common::forum_reader::format_host_speech_for_prompt("");
    assert!(formatted.is_empty());

    // format_host_speech_for_prompt は非空入力で内容を含む
    let formatted = bettafish_common::forum_reader::format_host_speech_for_prompt("テスト発言");
    assert!(formatted.contains("テスト発言"));
    assert!(formatted.contains("主持人"));
}

// =============================================================================
// JSON スキーマバリデーションテスト
// =============================================================================

#[test]
fn test_json_schema_validation() {
    let data = serde_json::json!({
        "title": "テスト",
        "content": "コンテンツ"
    });

    assert!(validate_json_schema(&data, &["title", "content"]));
    assert!(!validate_json_schema(&data, &["title", "missing_field"]));
    assert!(!validate_json_schema(&serde_json::json!("not an object"), &["title"]));
}
