//! BettaFish ReportEngine
//!
//! Python の ReportEngine パッケージの完全 Rust 実装。
//! テンプレートベースのレポート生成エンジン。
//!
//! ## モジュール構成
//! - **ir**: IR スキーマ定義と検証 (schema.py, validator.py)
//! - **core**: テンプレート解析、章ストレージ、ドキュメント合成
//! - **nodes**: ノードパイプライン (テンプレート選択、レイアウト、予算、章生成)
//! - **renderers**: HTML/PDF/Markdown レンダリング
//! - **state**: レポート状態管理
//! - **prompts**: 全システムプロンプト
//! - **json_parser**: 堅牢な JSON パーサー

#![allow(dead_code)]

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{Local, Utc};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{error, info, warn};

use bettafish_config::Settings;
use bettafish_llm::{InvokeOptions, LLMClient};
use bettafish_common::text_processing::{
    clean_json_tags, clean_markdown_tags, extract_clean_response,
    fix_incomplete_json, remove_reasoning_from_output,
};

/// バージョン情報
pub const VERSION: &str = "1.0.0";

// =============================================================================
// IR スキーマ定義 (ir/schema.py)
// =============================================================================

/// IR ドキュメントスキーマバージョン
pub const IR_VERSION: &str = "1.0";

/// サポートされるブロックタイプ (16 種類)
pub const ALLOWED_BLOCK_TYPES: &[&str] = &[
    "heading", "paragraph", "list", "table", "swotTable", "pestTable",
    "blockquote", "engineQuote", "hr", "code", "math", "figure",
    "callout", "kpiGrid", "widget", "toc",
];

/// サポートされるインラインマークタイプ (12 種類)
pub const ALLOWED_INLINE_MARKS: &[&str] = &[
    "bold", "italic", "underline", "strike", "code", "link",
    "color", "font", "highlight", "subscript", "superscript", "math",
];

/// エンジンエージェントタイトル
pub const ENGINE_AGENT_TITLES: &[(&str, &str)] = &[
    ("insight", "Insight Agent"),
    ("media", "Media Agent"),
    ("query", "Query Agent"),
];

// =============================================================================
// IR バリデーター (ir/validator.py)
// =============================================================================

/// IR バリデーター
pub struct IRValidator {
    schema_version: String,
}

impl IRValidator {
    pub fn new() -> Self {
        Self {
            schema_version: IR_VERSION.to_string(),
        }
    }

    /// 章を検証
    pub fn validate_chapter(&self, chapter: &Value) -> (bool, Vec<String>) {
        let mut errors = Vec::new();

        // 必須フィールドチェック
        for field in &["chapterId", "title", "anchor", "order"] {
            if chapter.get(*field).is_none() {
                errors.push(format!("章に必須フィールド '{}' がありません", field));
            }
        }

        // blocks 配列チェック
        match chapter.get("blocks") {
            Some(blocks) => {
                if let Some(arr) = blocks.as_array() {
                    if arr.is_empty() {
                        errors.push("章の blocks 配列が空です".to_string());
                    }
                    for (i, block) in arr.iter().enumerate() {
                        self.validate_block(block, &format!("blocks[{}]", i), &mut errors);
                    }
                } else {
                    errors.push("blocks はJSON配列でなければなりません".to_string());
                }
            }
            None => errors.push("章に 'blocks' フィールドがありません".to_string()),
        }

        (errors.is_empty(), errors)
    }

    /// ブロックを検証
    fn validate_block(&self, block: &Value, path: &str, errors: &mut Vec<String>) {
        let block_type = match block.get("type").and_then(|v| v.as_str()) {
            Some(t) => t,
            None => {
                errors.push(format!("{}: ブロックに 'type' フィールドがありません", path));
                return;
            }
        };

        if !ALLOWED_BLOCK_TYPES.contains(&block_type) {
            errors.push(format!("{}: 未知のブロックタイプ '{}'", path, block_type));
            return;
        }

        match block_type {
            "heading" => self.validate_heading_block(block, path, errors),
            "paragraph" => self.validate_paragraph_block(block, path, errors),
            "list" => self.validate_list_block(block, path, errors),
            "table" => self.validate_table_block(block, path, errors),
            "swotTable" => self.validate_swot_block(block, path, errors),
            "pestTable" => self.validate_pest_block(block, path, errors),
            "blockquote" => self.validate_blockquote_block(block, path, errors),
            "engineQuote" => self.validate_engine_quote_block(block, path, errors),
            _ => {} // hr, code, math, figure, callout, kpiGrid, widget, toc は基本検証のみ
        }
    }

    fn validate_heading_block(&self, block: &Value, path: &str, errors: &mut Vec<String>) {
        if block.get("level").and_then(|v| v.as_i64()).is_none() {
            errors.push(format!("{}: heading に 'level' (整数) が必要です", path));
        }
        if block.get("text").and_then(|v| v.as_str()).is_none() {
            errors.push(format!("{}: heading に 'text' (文字列) が必要です", path));
        }
    }

    fn validate_paragraph_block(&self, block: &Value, path: &str, errors: &mut Vec<String>) {
        match block.get("inlines") {
            Some(inlines) => {
                if let Some(arr) = inlines.as_array() {
                    if arr.is_empty() {
                        errors.push(format!("{}: paragraph の inlines 配列が空です", path));
                    }
                    for (i, inline) in arr.iter().enumerate() {
                        self.validate_inline_run(inline, &format!("{}.inlines[{}]", path, i), errors);
                    }
                }
            }
            None => errors.push(format!("{}: paragraph に 'inlines' が必要です", path)),
        }
    }

    fn validate_inline_run(&self, inline: &Value, path: &str, errors: &mut Vec<String>) {
        if let Some(marks) = inline.get("marks").and_then(|v| v.as_array()) {
            for (i, mark) in marks.iter().enumerate() {
                if let Some(mark_type) = mark.get("type").and_then(|v| v.as_str()) {
                    if !ALLOWED_INLINE_MARKS.contains(&mark_type) {
                        errors.push(format!("{}.marks[{}]: 未知のマークタイプ '{}'", path, i, mark_type));
                    }
                }
            }
        }
    }

    fn validate_list_block(&self, block: &Value, path: &str, errors: &mut Vec<String>) {
        if let Some(list_type) = block.get("listType").and_then(|v| v.as_str()) {
            if !["ordered", "bullet", "task"].contains(&list_type) {
                errors.push(format!("{}: 無効な listType '{}'", path, list_type));
            }
        }
        if let Some(items) = block.get("items").and_then(|v| v.as_array()) {
            for (i, item) in items.iter().enumerate() {
                if let Some(blocks) = item.as_array() {
                    for (j, b) in blocks.iter().enumerate() {
                        self.validate_block(b, &format!("{}.items[{}][{}]", path, i, j), errors);
                    }
                }
            }
        }
    }

    fn validate_table_block(&self, block: &Value, path: &str, errors: &mut Vec<String>) {
        if let Some(rows) = block.get("rows").and_then(|v| v.as_array()) {
            if rows.is_empty() {
                errors.push(format!("{}: table の rows が空です", path));
            }
            for (i, row) in rows.iter().enumerate() {
                if let Some(cells) = row.get("cells").and_then(|v| v.as_array()) {
                    for (j, cell) in cells.iter().enumerate() {
                        if let Some(blocks) = cell.get("blocks").and_then(|v| v.as_array()) {
                            for (k, b) in blocks.iter().enumerate() {
                                self.validate_block(b, &format!("{}.rows[{}].cells[{}].blocks[{}]", path, i, j, k), errors);
                            }
                        }
                    }
                }
            }
        }
    }

    fn validate_swot_block(&self, block: &Value, path: &str, errors: &mut Vec<String>) {
        for quadrant in &["strengths", "weaknesses", "opportunities", "threats"] {
            if let Some(items) = block.get(*quadrant).and_then(|v| v.as_array()) {
                for (i, item) in items.iter().enumerate() {
                    if let Some(impact) = item.get("impact").and_then(|v| v.as_str()) {
                        let valid_impacts = ["低", "中低", "中", "中高", "高", "极高"];
                        if !valid_impacts.contains(&impact) {
                            errors.push(format!("{}.{}[{}]: 無効な impact '{}'", path, quadrant, i, impact));
                        }
                    }
                }
            }
        }
    }

    fn validate_pest_block(&self, _block: &Value, _path: &str, _errors: &mut Vec<String>) {
        // PEST ブロックの基本構造検証
    }

    fn validate_blockquote_block(&self, block: &Value, path: &str, errors: &mut Vec<String>) {
        if let Some(blocks) = block.get("blocks").and_then(|v| v.as_array()) {
            for (i, b) in blocks.iter().enumerate() {
                self.validate_block(b, &format!("{}.blocks[{}]", path, i), errors);
            }
        }
    }

    fn validate_engine_quote_block(&self, block: &Value, path: &str, errors: &mut Vec<String>) {
        if let Some(engine) = block.get("engine").and_then(|v| v.as_str()) {
            if !["insight", "media", "query"].contains(&engine) {
                errors.push(format!("{}: 無効な engine '{}'", path, engine));
            }
        } else {
            errors.push(format!("{}: engineQuote に 'engine' が必要です", path));
        }
    }
}

impl Default for IRValidator {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// コア: テンプレートパーサー (core/template_parser.py)
// =============================================================================

/// テンプレートセクション
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateSection {
    pub title: String,
    pub slug: String,
    pub order: usize,
    pub depth: usize,
    pub raw_title: String,
    pub number: String,
    pub chapter_id: String,
    pub outline: Vec<String>,
}

impl TemplateSection {
    pub fn to_dict(&self) -> Value {
        serde_json::json!({
            "title": self.title,
            "slug": self.slug,
            "order": self.order,
            "depth": self.depth,
            "raw_title": self.raw_title,
            "number": self.number,
            "chapter_id": self.chapter_id,
            "outline": self.outline,
        })
    }
}

const SECTION_ORDER_STEP: usize = 10;

/// Markdown テンプレートからセクションリストを解析
pub fn parse_template_sections(template_md: &str) -> Vec<TemplateSection> {
    let heading_re = Regex::new(r"^(#{1,6})\s+(.+)$").unwrap();
    let bullet_re = Regex::new(r"^[-*+]\s+(.+)$").unwrap();
    let number_re = Regex::new(r"^(\d+(?:\.\d+)*)\s+(.+)$").unwrap();

    let mut sections = Vec::new();
    let mut used_slugs = HashSet::new();
    let mut current_outline: Vec<String> = Vec::new();
    let mut order_counter = 0;

    for line in template_md.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if let Some(caps) = heading_re.captures(trimmed) {
            // 前のセクションのアウトラインを保存
            if let Some(last) = sections.last_mut() {
                let last_section: &mut TemplateSection = last;
                last_section.outline = current_outline.clone();
            }
            current_outline.clear();

            let depth = caps[1].len();
            let raw_title = caps[2].trim().to_string();
            let title = strip_markup(&raw_title);
            let slug = ensure_unique_slug(&build_slug("", &title), &mut used_slugs);

            order_counter += SECTION_ORDER_STEP;
            let chapter_id = format!("S{}", sections.len() + 1);

            sections.push(TemplateSection {
                title,
                slug,
                order: order_counter,
                depth,
                raw_title,
                number: String::new(),
                chapter_id,
                outline: Vec::new(),
            });
        } else if let Some(caps) = bullet_re.captures(trimmed) {
            let content = caps[1].trim().to_string();
            // 番号付きタイトルのチェック
            if let Some(num_caps) = number_re.captures(&content) {
                let number = num_caps[1].to_string();
                let label = strip_markup(num_caps[2].trim());
                let slug = ensure_unique_slug(&build_slug(&number, &label), &mut used_slugs);

                if let Some(last) = sections.last_mut() {
                    let last_section: &mut TemplateSection = last;
                    last_section.outline = current_outline.clone();
                }
                current_outline.clear();

                order_counter += SECTION_ORDER_STEP;
                let chapter_id = format!("S{}", sections.len() + 1);

                sections.push(TemplateSection {
                    title: label,
                    slug,
                    order: order_counter,
                    depth: 2,
                    raw_title: content,
                    number,
                    chapter_id,
                    outline: Vec::new(),
                });
            } else {
                current_outline.push(strip_markup(&content));
            }
        } else {
            current_outline.push(trimmed.to_string());
        }
    }

    // 最後のセクションのアウトラインを保存
    if let Some(last) = sections.last_mut() {
        last.outline = current_outline;
    }

    sections
}

fn strip_markup(text: &str) -> String {
    let re = Regex::new(r"\*\*|__").unwrap();
    re.replace_all(text, "").trim().to_string()
}

fn build_slug(number: &str, title: &str) -> String {
    if !number.is_empty() {
        format!("section-{}", number.replace('.', "-"))
    } else {
        let slug = slugify_text(title);
        if slug.is_empty() {
            "section".to_string()
        } else {
            format!("section-{}", slug)
        }
    }
}

fn slugify_text(text: &str) -> String {
    let re = Regex::new(r"[^\w\s\u{4e00}-\u{9fff}-]").unwrap();
    let cleaned = re.replace_all(text, "");
    cleaned.to_lowercase().split_whitespace().collect::<Vec<_>>().join("-")
}

fn ensure_unique_slug(slug: &str, used: &mut HashSet<String>) -> String {
    let mut result = slug.to_string();
    let mut counter = 2;
    while used.contains(&result) {
        result = format!("{}-{}", slug, counter);
        counter += 1;
    }
    used.insert(result.clone());
    result
}

// =============================================================================
// コア: 章ストレージ (core/chapter_storage.py)
// =============================================================================

/// 章レコード
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChapterRecord {
    pub chapter_id: String,
    pub slug: String,
    pub title: String,
    pub order: usize,
    pub status: String,
    pub files: HashMap<String, String>,
    pub errors: Vec<String>,
    pub updated_at: String,
}

/// 章ストレージ管理
pub struct ChapterStorage {
    base_dir: PathBuf,
    manifests: HashMap<String, Value>,
}

impl ChapterStorage {
    pub fn new(base_dir: &str) -> Self {
        fs::create_dir_all(base_dir).ok();
        Self {
            base_dir: PathBuf::from(base_dir),
            manifests: HashMap::new(),
        }
    }

    /// セッションを開始
    pub fn start_session(&mut self, report_id: &str, metadata: &Value) -> Result<PathBuf> {
        let run_dir = self.base_dir.join(report_id);
        fs::create_dir_all(&run_dir)?;

        let manifest = serde_json::json!({
            "report_id": report_id,
            "metadata": metadata,
            "chapters": [],
            "created_at": Utc::now().to_rfc3339(),
        });

        let manifest_path = run_dir.join("manifest.json");
        fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)?;
        self.manifests.insert(report_id.to_string(), manifest);

        info!("ChapterStorage: セッション開始 {}", report_id);
        Ok(run_dir)
    }

    /// 章ディレクトリを準備
    pub fn begin_chapter(&self, run_dir: &Path, chapter_meta: &Value) -> Result<PathBuf> {
        let slug = chapter_meta.get("slug").and_then(|v| v.as_str()).unwrap_or("unknown");
        let order = chapter_meta.get("order").and_then(|v| v.as_u64()).unwrap_or(0);

        let safe_slug = slug.replace(['/', '\\', ' '], "-");
        let chapter_dir = run_dir.join(format!("{:03}-{}", order, safe_slug));
        fs::create_dir_all(&chapter_dir)?;

        Ok(chapter_dir)
    }

    /// 章を永続化
    pub fn persist_chapter(
        &self,
        _run_dir: &Path,
        chapter_dir: &Path,
        payload: &Value,
        errors: &[String],
    ) -> Result<PathBuf> {
        let chapter_path = chapter_dir.join("chapter.json");
        fs::write(&chapter_path, serde_json::to_string_pretty(payload)?)?;

        if !errors.is_empty() {
            let errors_path = chapter_dir.join("errors.json");
            fs::write(&errors_path, serde_json::to_string_pretty(&errors)?)?;
        }

        Ok(chapter_path)
    }

    /// 章を読み込み
    pub fn load_chapters(&self, run_dir: &Path) -> Result<Vec<Value>> {
        let mut chapters = Vec::new();

        if !run_dir.exists() {
            return Ok(chapters);
        }

        let mut entries: Vec<_> = fs::read_dir(run_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .collect();
        entries.sort_by_key(|e| e.file_name());

        for entry in entries {
            let chapter_path = entry.path().join("chapter.json");
            if chapter_path.exists() {
                let content = fs::read_to_string(&chapter_path)?;
                if let Ok(chapter) = serde_json::from_str::<Value>(&content) {
                    chapters.push(chapter);
                }
            }
        }

        Ok(chapters)
    }
}

// =============================================================================
// コア: ドキュメントスティッチャー (core/stitcher.py)
// =============================================================================

/// ドキュメント合成器
pub struct DocumentComposer {
    seen_anchors: HashSet<String>,
}

impl DocumentComposer {
    pub fn new() -> Self {
        Self {
            seen_anchors: HashSet::new(),
        }
    }

    /// ドキュメントを構築
    pub fn build_document(
        &mut self,
        report_id: &str,
        metadata: &Value,
        chapters: &[Value],
    ) -> Value {
        self.seen_anchors.clear();

        let toc_anchor_map = self.build_toc_anchor_map(metadata);

        let mut sorted_chapters: Vec<Value> = chapters.to_vec();
        sorted_chapters.sort_by_key(|c| c.get("order").and_then(|v| v.as_u64()).unwrap_or(0));

        // デフォルト chapterId と一意なアンカーを割り当て
        for (idx, chapter) in sorted_chapters.iter_mut().enumerate() {
            if chapter.get("chapterId").is_none() {
                chapter["chapterId"] = Value::String(format!("S{}", idx + 1));
            }

            let chapter_id = chapter.get("chapterId").and_then(|v| v.as_str()).unwrap_or("");
            let anchor = if let Some(custom) = toc_anchor_map.get(chapter_id) {
                self.ensure_unique_anchor(custom)
            } else if let Some(existing) = chapter.get("anchor").and_then(|v| v.as_str()) {
                self.ensure_unique_anchor(existing)
            } else {
                self.ensure_unique_anchor(&format!("section-{}", idx))
            };
            chapter["anchor"] = Value::String(anchor);
        }

        serde_json::json!({
            "version": IR_VERSION,
            "reportId": report_id,
            "metadata": metadata,
            "chapters": sorted_chapters,
            "generatedAt": Utc::now().to_rfc3339(),
        })
    }

    fn ensure_unique_anchor(&mut self, anchor: &str) -> String {
        let mut result = anchor.to_string();
        let mut counter = 2;
        while self.seen_anchors.contains(&result) {
            result = format!("{}-{}", anchor, counter);
            counter += 1;
        }
        self.seen_anchors.insert(result.clone());
        result
    }

    fn build_toc_anchor_map(&self, metadata: &Value) -> HashMap<String, String> {
        let mut map = HashMap::new();
        if let Some(toc) = metadata.get("toc") {
            if let Some(entries) = toc.get("customEntries").and_then(|v| v.as_array()) {
                for entry in entries {
                    if let (Some(id), Some(anchor)) = (
                        entry.get("chapterId").and_then(|v| v.as_str()),
                        entry.get("anchor").and_then(|v| v.as_str()),
                    ) {
                        map.insert(id.to_string(), anchor.to_string());
                    }
                }
            }
        }
        map
    }
}

impl Default for DocumentComposer {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// 状態管理 (state/state.py)
// =============================================================================

/// レポートメタデータ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMetadata {
    pub query: String,
    pub template_used: String,
    pub generation_time: f64,
    pub timestamp: String,
}

impl Default for ReportMetadata {
    fn default() -> Self {
        Self {
            query: String::new(),
            template_used: String::new(),
            generation_time: 0.0,
            timestamp: Local::now().to_rfc3339(),
        }
    }
}

/// レポート状態
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportState {
    pub task_id: String,
    pub query: String,
    pub status: String,
    pub query_engine_report: String,
    pub media_engine_report: String,
    pub insight_engine_report: String,
    pub forum_logs: String,
    pub selected_template: String,
    pub html_content: String,
    pub markdown_content: String,
    pub metadata: ReportMetadata,
    pub error_message: Option<String>,
}

impl Default for ReportState {
    fn default() -> Self {
        Self {
            task_id: format!("report_{}", Utc::now().format("%Y%m%d_%H%M%S")),
            query: String::new(),
            status: "pending".to_string(),
            query_engine_report: String::new(),
            media_engine_report: String::new(),
            insight_engine_report: String::new(),
            forum_logs: String::new(),
            selected_template: String::new(),
            html_content: String::new(),
            markdown_content: String::new(),
            metadata: ReportMetadata::default(),
            error_message: None,
        }
    }
}

impl ReportState {
    pub fn mark_processing(&mut self) {
        self.status = "processing".to_string();
    }

    pub fn mark_completed(&mut self) {
        self.status = "completed".to_string();
    }

    pub fn mark_failed(&mut self, error: &str) {
        self.status = "failed".to_string();
        self.error_message = Some(error.to_string());
    }

    pub fn is_completed(&self) -> bool {
        self.status == "completed"
    }

    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    pub fn save_to_file(&self, path: &str) -> Result<()> {
        fs::write(path, self.to_json()?)?;
        Ok(())
    }

    pub fn load_from_file(path: &str) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        Ok(serde_json::from_str(&content)?)
    }
}

// =============================================================================
// 堅牢な JSON パーサー (utils/json_parser.py)
// =============================================================================

/// 堅牢な JSON パーサー
pub struct RobustJSONParser {
    enable_json_repair: bool,
    max_repair_attempts: usize,
}

impl RobustJSONParser {
    pub fn new(enable_json_repair: bool, max_repair_attempts: usize) -> Self {
        Self {
            enable_json_repair,
            max_repair_attempts,
        }
    }

    /// JSON テキストを解析
    pub fn parse(
        &self,
        raw_text: &str,
        context_name: &str,
        expected_keys: Option<&[&str]>,
    ) -> Result<Value> {
        let cleaned = self.clean_response(raw_text);

        // 直接パース試行
        if let Ok(val) = serde_json::from_str::<Value>(&cleaned) {
            if let Some(keys) = expected_keys {
                self.validate_expected_keys(&val, keys, context_name);
            }
            return Ok(val);
        }

        // ローカル修復試行
        let repaired = self.apply_local_repairs(&cleaned);
        if let Ok(val) = serde_json::from_str::<Value>(&repaired) {
            info!("{}: ローカル修復後にJSON解析成功", context_name);
            return Ok(val);
        }

        // fix_incomplete_json 試行
        if self.enable_json_repair {
            if let Some(fixed) = fix_incomplete_json(&repaired) {
                if let Ok(val) = serde_json::from_str::<Value>(&fixed) {
                    info!("{}: JSON修復後に解析成功", context_name);
                    return Ok(val);
                }
            }
        }

        // extract_clean_response 試行
        let extracted = extract_clean_response(&repaired);
        if extracted.get("error").is_none() {
            return Ok(extracted);
        }

        anyhow::bail!("{}: 全てのJSON解析戦略が失敗しました", context_name)
    }

    fn clean_response(&self, raw: &str) -> String {
        let mut text = raw.to_string();

        // Markdown コードブロックを除去
        text = clean_json_tags(&text);

        // thinking タグを除去
        let re_thinking = Regex::new(r"(?s)<thinking>.*?</thinking>").unwrap();
        text = re_thinking.replace_all(&text, "").to_string();

        // 推論部分を除去
        text = remove_reasoning_from_output(&text);

        text.trim().to_string()
    }

    fn apply_local_repairs(&self, text: &str) -> String {
        let mut result = text.to_string();

        // コロン=パターンを修復 ("key"= → "key":)
        let re_colon_eq = Regex::new(r#"(":\s*)="#).unwrap();
        result = re_colon_eq.replace_all(&result, "$1").to_string();

        // 制御文字をエスケープ
        result = result.replace('\t', "\\t");

        // 末尾カンマを除去
        let re_trailing = Regex::new(r",\s*([}\]])").unwrap();
        result = re_trailing.replace_all(&result, "$1").to_string();

        result
    }

    fn validate_expected_keys(&self, data: &Value, keys: &[&str], context_name: &str) {
        if let Some(obj) = data.as_object() {
            for key in keys {
                if !obj.contains_key(*key) {
                    warn!("{}: 期待キー '{}' がありません", context_name, key);
                }
            }
        }
    }
}

impl Default for RobustJSONParser {
    fn default() -> Self {
        Self::new(true, 3)
    }
}

// =============================================================================
// プロンプト定義 (prompts/prompts.py)
// =============================================================================

pub fn system_prompt_template_selection() -> String {
    r#"你是一个报告模板选择专家。根据用户的查询主题、分析报告内容和论坛讨论日志，选择最合适的报告模板。

请以JSON格式返回选择结果：
{
  "template_name": "选择的模板名称",
  "selection_reason": "选择理由"
}

只返回JSON对象，不要有额外文字。"#.to_string()
}

pub fn system_prompt_document_layout() -> String {
    r##"你是一个专业的文档设计师。根据提供的报告模板、分析内容和论坛日志，生成文档的全局设计。

请以JSON格式返回设计文档：
{
  "title": "报告标题",
  "subtitle": "副标题（可选）",
  "tocPlan": [
    {"chapterId": "S1", "anchor": "section-1", "display": "章节显示名", "description": "章节描述"}
  ],
  "hero": {"headline": "核心发现", "subtext": "摘要"},
  "themeTokens": {"primaryColor": "#4A90E2", "fontFamily": "sans-serif"}
}

只返回JSON对象，不要有额外文字。"##.to_string()
}

pub fn system_prompt_word_budget() -> String {
    r#"你是一个内容规划专家。根据报告模板和设计文档，为每个章节规划字数预算。

请以JSON格式返回字数预算：
{
  "totalWords": 10000,
  "globalGuidelines": ["写作指导1", "写作指导2"],
  "chapters": [
    {"chapterId": "S1", "targetWords": 2000, "minWords": 1500, "maxWords": 2500}
  ]
}

只返回JSON对象，不要有额外文字。"#.to_string()
}

pub fn system_prompt_chapter_json() -> String {
    format!(r#"你是一个专业的报告章节生成器。根据提供的章节信息和上下文数据，生成符合IR规范的章节JSON。

支持的块类型: {}
支持的内联标记: {}

请生成包含以下结构的JSON：
{{
  "chapterId": "S1",
  "title": "章节标题",
  "anchor": "section-anchor",
  "order": 10,
  "blocks": [
    {{"type": "heading", "level": 2, "text": "标题文本", "anchor": "heading-anchor"}},
    {{"type": "paragraph", "inlines": [{{"text": "段落内容", "marks": []}}]}},
    {{"type": "table", "rows": [{{"cells": [{{"blocks": [{{"type": "paragraph", "inlines": [{{"text": "单元格", "marks": []}}]}}]}}]}}]}}
  ]
}}

确保所有块类型和内联标记在允许列表中。只返回JSON对象。"#,
        ALLOWED_BLOCK_TYPES.join(", "),
        ALLOWED_INLINE_MARKS.join(", ")
    )
}

pub fn system_prompt_chapter_json_repair() -> String {
    r#"你是一个JSON修复专家。以下是一个有错误的章节JSON和具体的验证错误列表。
请修复JSON使其通过验证，保持原有内容不变。只返回修复后的JSON对象。"#.to_string()
}

pub fn system_prompt_chapter_json_recovery() -> String {
    r#"你是一个JSON恢复专家。以下是LLM生成的原始文本，但无法解析为有效的JSON。
请从原始文本中提取信息，生成一个有效的章节JSON。只返回JSON对象。"#.to_string()
}

pub fn build_chapter_user_prompt(payload: &Value) -> String {
    serde_json::to_string_pretty(payload).unwrap_or_default()
}

pub fn build_document_layout_prompt(payload: &Value) -> String {
    serde_json::to_string_pretty(payload).unwrap_or_default()
}

pub fn build_word_budget_prompt(payload: &Value) -> String {
    serde_json::to_string_pretty(payload).unwrap_or_default()
}

pub fn build_chapter_repair_prompt(raw_json: &str, errors: &[String]) -> String {
    format!("原始JSON:\n```json\n{}\n```\n\n验证错误:\n{}", raw_json, errors.join("\n"))
}

// =============================================================================
// ノード基底トレイト
// =============================================================================

#[async_trait]
trait Node: Send + Sync {
    fn name(&self) -> &str;
    async fn run(&self, input_data: &Value) -> Result<Value>;
}

// =============================================================================
// ノード: テンプレート選択 (nodes/template_selection_node.py)
// =============================================================================

pub struct TemplateSelectionNode {
    llm_client: LLMClient,
    json_parser: RobustJSONParser,
}

impl TemplateSelectionNode {
    pub fn new(llm_client: LLMClient) -> Self {
        Self {
            llm_client,
            json_parser: RobustJSONParser::default(),
        }
    }

    pub async fn run(&self, query: &str, reports: &HashMap<String, String>, forum_logs: &str) -> Result<Value> {
        let payload = serde_json::json!({
            "query": query,
            "reports_summary": reports.iter().map(|(k, v)| {
                let summary: String = v.chars().take(1000).collect();
                (k.clone(), summary)
            }).collect::<HashMap<String, String>>(),
            "forum_logs_summary": forum_logs.chars().take(500).collect::<String>(),
        });

        let system_prompt = system_prompt_template_selection();
        let user_prompt = serde_json::to_string_pretty(&payload)?;

        let response = self.llm_client
            .invoke(&system_prompt, &user_prompt, &InvokeOptions::default())
            .await?;

        self.json_parser.parse(&response, "TemplateSelection", Some(&["template_name"]))
    }
}

// =============================================================================
// ノード: ドキュメントレイアウト (nodes/document_layout_node.py)
// =============================================================================

pub struct DocumentLayoutNode {
    llm_client: LLMClient,
    json_parser: RobustJSONParser,
}

impl DocumentLayoutNode {
    pub fn new(llm_client: LLMClient) -> Self {
        Self {
            llm_client,
            json_parser: RobustJSONParser::default(),
        }
    }

    pub async fn run(
        &self,
        sections: &[TemplateSection],
        template_markdown: &str,
        reports: &HashMap<String, String>,
        forum_logs: &str,
        query: &str,
    ) -> Result<Value> {
        let sections_json: Vec<Value> = sections.iter().map(|s| s.to_dict()).collect();
        let payload = serde_json::json!({
            "query": query,
            "sections": sections_json,
            "template_markdown": template_markdown,
            "reports": reports,
            "forum_logs": forum_logs.chars().take(2000).collect::<String>(),
        });

        let system_prompt = system_prompt_document_layout();
        let user_prompt = build_document_layout_prompt(&payload);

        let response = self.llm_client
            .invoke(&system_prompt, &user_prompt, &InvokeOptions::default())
            .await?;

        self.json_parser.parse(&response, "DocumentLayout", Some(&["title", "tocPlan"]))
    }
}

// =============================================================================
// ノード: ワードバジェット (nodes/word_budget_node.py)
// =============================================================================

pub struct WordBudgetNode {
    llm_client: LLMClient,
    json_parser: RobustJSONParser,
}

impl WordBudgetNode {
    pub fn new(llm_client: LLMClient) -> Self {
        Self {
            llm_client,
            json_parser: RobustJSONParser::default(),
        }
    }

    pub async fn run(
        &self,
        sections: &[TemplateSection],
        design: &Value,
        reports: &HashMap<String, String>,
        forum_logs: &str,
        query: &str,
    ) -> Result<Value> {
        let sections_json: Vec<Value> = sections.iter().map(|s| s.to_dict()).collect();
        let payload = serde_json::json!({
            "query": query,
            "sections": sections_json,
            "design": design,
            "reports": reports,
            "forum_logs": forum_logs.chars().take(2000).collect::<String>(),
        });

        let system_prompt = system_prompt_word_budget();
        let user_prompt = build_word_budget_prompt(&payload);

        let response = self.llm_client
            .invoke(&system_prompt, &user_prompt, &InvokeOptions::default())
            .await?;

        let mut result = self.json_parser.parse(&response, "WordBudget", Some(&["totalWords", "chapters"]))?;

        // デフォルト値を適用
        if result.get("totalWords").and_then(|v| v.as_u64()).is_none() {
            result["totalWords"] = serde_json::json!(10000);
        }
        if result.get("globalGuidelines").and_then(|v| v.as_array()).is_none() {
            result["globalGuidelines"] = serde_json::json!([]);
        }

        Ok(result)
    }
}

// =============================================================================
// ノード: 章生成 (nodes/chapter_generation_node.py)
// =============================================================================

/// 章 JSON 解析エラー
#[derive(Debug, thiserror::Error)]
pub enum ChapterError {
    #[error("JSON解析エラー: {message}")]
    JsonParseError { message: String, raw_text: Option<String> },
    #[error("コンテンツエラー: {message}")]
    ContentError { message: String, body_characters: usize, non_heading_blocks: usize },
    #[error("検証エラー: {message}")]
    ValidationError { message: String, errors: Vec<String> },
}

/// 章生成ノード
pub struct ChapterGenerationNode {
    llm_client: LLMClient,
    validator: IRValidator,
    storage: ChapterStorage,
    json_parser: RobustJSONParser,
    // コンテンツ密度チェック定数
    min_non_heading_blocks: usize,
    min_body_characters: usize,
    min_narrative_characters: usize,
}

/// インラインマークエイリアスマッピング
const INLINE_MARK_ALIASES: &[(&str, &str)] = &[
    ("strong", "bold"),
    ("em", "italic"),
    ("del", "strike"),
    ("s", "strike"),
    ("u", "underline"),
    ("a", "link"),
    ("sup", "superscript"),
    ("sub", "subscript"),
];

impl ChapterGenerationNode {
    pub fn new(llm_client: LLMClient, storage: ChapterStorage) -> Self {
        Self {
            llm_client,
            validator: IRValidator::new(),
            storage,
            json_parser: RobustJSONParser::new(true, 3),
            min_non_heading_blocks: 2,
            min_body_characters: 600,
            min_narrative_characters: 300,
        }
    }

    /// 章を生成
    pub async fn run(
        &self,
        section: &TemplateSection,
        context: &Value,
        run_dir: &Path,
    ) -> Result<Value> {
        // 章ディレクトリを準備
        let chapter_meta = serde_json::json!({
            "slug": section.slug,
            "order": section.order,
        });
        let chapter_dir = self.storage.begin_chapter(run_dir, &chapter_meta)?;

        // LLM ペイロードを構築
        let payload = self.build_payload(section, context);
        let system_prompt = system_prompt_chapter_json();
        let user_prompt = build_chapter_user_prompt(&payload);

        // LLM 呼び出し（ストリーミング）
        let raw_text = self.llm_client
            .stream_invoke_to_string(&system_prompt, &user_prompt, &InvokeOptions::default())
            .await?;

        // stream.raw に保存
        let raw_path = chapter_dir.join("stream.raw");
        fs::write(&raw_path, &raw_text)?;

        // JSON 解析
        let mut chapter_json = self.parse_chapter(&raw_text)?;

        // ブロックサニタイズ
        self.sanitize_chapter_blocks(&mut chapter_json);

        // IR 検証
        let (valid, errors) = self.validator.validate_chapter(&chapter_json);
        if !valid {
            warn!("章の検証に失敗: {:?}", errors);
            // 構造修復を試行
            if let Some(repaired) = self.attempt_structural_repair(&chapter_json, &errors, &raw_text).await {
                let (valid2, errors2) = self.validator.validate_chapter(&repaired);
                if valid2 {
                    info!("構造修復成功");
                    self.storage.persist_chapter(run_dir, &chapter_dir, &repaired, &[])?;
                    return Ok(repaired);
                }
                warn!("構造修復後も検証失敗: {:?}", errors2);
            }
        }

        // コンテンツ密度チェック
        self.ensure_content_density(&chapter_json)?;

        // 永続化
        self.storage.persist_chapter(run_dir, &chapter_dir, &chapter_json, &errors)?;

        Ok(chapter_json)
    }

    fn build_payload(&self, section: &TemplateSection, context: &Value) -> Value {
        serde_json::json!({
            "section": section.to_dict(),
            "context": context,
        })
    }

    fn parse_chapter(&self, raw_text: &str) -> Result<Value> {
        self.json_parser.parse(raw_text, "ChapterGeneration", Some(&["chapterId", "title", "blocks"]))
    }

    fn sanitize_chapter_blocks(&self, chapter: &mut Value) {
        if let Some(blocks) = chapter.get_mut("blocks").and_then(|v| v.as_array_mut()) {
            for block in blocks.iter_mut() {
                self.sanitize_block(block);
            }
        }
    }

    fn sanitize_block(&self, block: &mut Value) {
        // インラインマークエイリアスを正規化
        if let Some(inlines) = block.get_mut("inlines").and_then(|v| v.as_array_mut()) {
            for inline in inlines.iter_mut() {
                if let Some(marks) = inline.get_mut("marks").and_then(|v| v.as_array_mut()) {
                    for mark in marks.iter_mut() {
                        if let Some(mark_type) = mark.get("type").and_then(|v| v.as_str()).map(|s| s.to_string()) {
                            for (alias, canonical) in INLINE_MARK_ALIASES {
                                if mark_type == *alias {
                                    mark["type"] = Value::String(canonical.to_string());
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        // テーブルセルの空ブロックを修復
        if let Some(rows) = block.get_mut("rows").and_then(|v| v.as_array_mut()) {
            for row in rows.iter_mut() {
                if let Some(cells) = row.get_mut("cells").and_then(|v| v.as_array_mut()) {
                    for cell in cells.iter_mut() {
                        if let Some(cell_blocks) = cell.get_mut("blocks").and_then(|v| v.as_array_mut()) {
                            if cell_blocks.is_empty() {
                                cell_blocks.push(serde_json::json!({
                                    "type": "paragraph",
                                    "inlines": [{"text": "", "marks": []}]
                                }));
                            }
                        }
                    }
                }
            }
        }

        // 再帰的にネストブロックをサニタイズ
        let nested_keys = ["blocks", "items"];
        for key in &nested_keys {
            if let Some(children) = block.get_mut(*key).and_then(|v| v.as_array_mut()) {
                for child in children.iter_mut() {
                    if child.is_object() {
                        self.sanitize_block(child);
                    } else if let Some(arr) = child.as_array_mut() {
                        for item in arr.iter_mut() {
                            if item.is_object() {
                                self.sanitize_block(item);
                            }
                        }
                    }
                }
            }
        }
    }

    fn ensure_content_density(&self, chapter: &Value) -> Result<()> {
        let empty_vec = vec![];
        let blocks = chapter.get("blocks").and_then(|v| v.as_array()).unwrap_or(&empty_vec);

        let non_heading_blocks = blocks.iter()
            .filter(|b| b.get("type").and_then(|v| v.as_str()) != Some("heading"))
            .count();

        if non_heading_blocks < self.min_non_heading_blocks {
            return Err(ChapterError::ContentError {
                message: format!("非見出しブロック数が不足 ({} < {})", non_heading_blocks, self.min_non_heading_blocks),
                body_characters: 0,
                non_heading_blocks,
            }.into());
        }

        Ok(())
    }

    async fn attempt_structural_repair(&self, chapter: &Value, errors: &[String], raw_text: &str) -> Option<Value> {
        let system_prompt = system_prompt_chapter_json_repair();
        let user_prompt = build_chapter_repair_prompt(
            &serde_json::to_string_pretty(chapter).unwrap_or_default(),
            errors,
        );

        match self.llm_client
            .invoke(&system_prompt, &user_prompt, &InvokeOptions::default())
            .await
        {
            Ok(response) => {
                self.json_parser.parse(&response, "ChapterRepair", None).ok()
            }
            Err(e) => {
                error!("構造修復 LLM 呼び出し失敗: {}", e);
                None
            }
        }
    }
}

// =============================================================================
// レンダラー: Markdown (renderers/markdown_renderer.py)
// =============================================================================

/// Markdown レンダラー
pub struct MarkdownRenderer;

impl MarkdownRenderer {
    pub fn new() -> Self {
        Self
    }

    /// ドキュメント IR を Markdown に変換
    pub fn render(&self, document_ir: &Value) -> String {
        let mut lines = Vec::new();

        // タイトル
        if let Some(title) = document_ir.get("metadata")
            .and_then(|m| m.get("title"))
            .and_then(|v| v.as_str())
        {
            lines.push(format!("# {}", title));
            lines.push(String::new());
        }

        // 章を順番にレンダリング
        if let Some(chapters) = document_ir.get("chapters").and_then(|v| v.as_array()) {
            for chapter in chapters {
                lines.push(self.render_chapter(chapter));
                lines.push(String::new());
            }
        }

        lines.join("\n")
    }

    fn render_chapter(&self, chapter: &Value) -> String {
        let mut lines = Vec::new();

        if let Some(title) = chapter.get("title").and_then(|v| v.as_str()) {
            lines.push(format!("## {}", title));
            lines.push(String::new());
        }

        if let Some(blocks) = chapter.get("blocks").and_then(|v| v.as_array()) {
            for block in blocks {
                let rendered = self.render_block(block);
                if !rendered.is_empty() {
                    lines.push(rendered);
                    lines.push(String::new());
                }
            }
        }

        lines.join("\n")
    }

    fn render_block(&self, block: &Value) -> String {
        let block_type = block.get("type").and_then(|v| v.as_str()).unwrap_or("");

        match block_type {
            "heading" => self.render_heading(block),
            "paragraph" => self.render_paragraph(block),
            "list" => self.render_list(block),
            "table" => self.render_table(block),
            "swotTable" => self.render_swot_table(block),
            "pestTable" => self.render_pest_table(block),
            "blockquote" => self.render_blockquote(block),
            "engineQuote" => self.render_engine_quote(block),
            "hr" => "---".to_string(),
            "code" => self.render_code(block),
            "math" => self.render_math(block),
            "figure" => self.render_figure(block),
            "callout" => self.render_callout(block),
            "kpiGrid" => self.render_kpi_grid(block),
            "widget" => self.render_widget(block),
            "toc" => "[TOC]".to_string(),
            _ => String::new(),
        }
    }

    fn render_heading(&self, block: &Value) -> String {
        let level = block.get("level").and_then(|v| v.as_u64()).unwrap_or(2) as usize;
        let text = block.get("text").and_then(|v| v.as_str()).unwrap_or("");
        let hashes = "#".repeat(level.min(6));
        format!("{} {}", hashes, text)
    }

    fn render_paragraph(&self, block: &Value) -> String {
        self.render_inlines(block.get("inlines").and_then(|v| v.as_array()))
    }

    fn render_inlines(&self, inlines: Option<&Vec<Value>>) -> String {
        let inlines = match inlines {
            Some(arr) => arr,
            None => return String::new(),
        };

        let mut result = String::new();
        for inline in inlines {
            let text = inline.get("text").and_then(|v| v.as_str()).unwrap_or("");
            let marks = inline.get("marks").and_then(|v| v.as_array());

            let mut formatted = text.to_string();
            if let Some(marks) = marks {
                for mark in marks.iter().rev() {
                    let mark_type = mark.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    formatted = match mark_type {
                        "bold" => format!("**{}**", formatted),
                        "italic" => format!("*{}*", formatted),
                        "underline" => format!("<u>{}</u>", formatted),
                        "strike" => format!("~~{}~~", formatted),
                        "code" => format!("`{}`", formatted),
                        "link" => {
                            let href = mark.get("href").and_then(|v| v.as_str()).unwrap_or("#");
                            format!("[{}]({})", formatted, href)
                        }
                        "subscript" => format!("<sub>{}</sub>", formatted),
                        "superscript" => format!("<sup>{}</sup>", formatted),
                        "math" => format!("${}$", formatted),
                        _ => formatted,
                    };
                }
            }
            result.push_str(&formatted);
        }

        result
    }

    fn render_list(&self, block: &Value) -> String {
        let list_type = block.get("listType").and_then(|v| v.as_str()).unwrap_or("bullet");
        let items = block.get("items").and_then(|v| v.as_array());
        let mut lines = Vec::new();

        if let Some(items) = items {
            for (i, item) in items.iter().enumerate() {
                let prefix = match list_type {
                    "ordered" => format!("{}. ", i + 1),
                    "task" => "- [ ] ".to_string(),
                    _ => "- ".to_string(),
                };

                if let Some(blocks) = item.as_array() {
                    let content: Vec<String> = blocks.iter()
                        .map(|b| self.render_block(b))
                        .filter(|s| !s.is_empty())
                        .collect();
                    lines.push(format!("{}{}", prefix, content.join(" ")));
                }
            }
        }

        lines.join("\n")
    }

    fn render_table(&self, block: &Value) -> String {
        let rows = match block.get("rows").and_then(|v| v.as_array()) {
            Some(r) => r,
            None => return String::new(),
        };
        if rows.is_empty() { return String::new(); }

        let mut md_rows: Vec<Vec<String>> = Vec::new();
        for row in rows {
            if let Some(cells) = row.get("cells").and_then(|v| v.as_array()) {
                let cell_texts: Vec<String> = cells.iter().map(|cell| {
                    if let Some(blocks) = cell.get("blocks").and_then(|v| v.as_array()) {
                        blocks.iter()
                            .map(|b| self.render_block(b))
                            .collect::<Vec<_>>()
                            .join(" ")
                            .replace('|', "\\|")
                    } else {
                        String::new()
                    }
                }).collect();
                md_rows.push(cell_texts);
            }
        }

        if md_rows.is_empty() { return String::new(); }

        let mut lines = Vec::new();
        // ヘッダー行
        lines.push(format!("| {} |", md_rows[0].join(" | ")));
        lines.push(format!("| {} |", md_rows[0].iter().map(|_| "---").collect::<Vec<_>>().join(" | ")));
        // データ行
        for row in &md_rows[1..] {
            lines.push(format!("| {} |", row.join(" | ")));
        }

        lines.join("\n")
    }

    fn render_swot_table(&self, block: &Value) -> String {
        let mut lines = vec!["### SWOT 分析".to_string(), String::new()];
        for (key, label) in &[("strengths", "強み"), ("weaknesses", "弱み"), ("opportunities", "機会"), ("threats", "脅威")] {
            lines.push(format!("**{}:**", label));
            if let Some(items) = block.get(*key).and_then(|v| v.as_array()) {
                for item in items {
                    let text = if let Some(s) = item.as_str() {
                        s.to_string()
                    } else {
                        item.get("title").or(item.get("text")).or(item.get("detail"))
                            .and_then(|v| v.as_str()).unwrap_or("").to_string()
                    };
                    if !text.is_empty() { lines.push(format!("- {}", text)); }
                }
            }
            lines.push(String::new());
        }
        lines.join("\n")
    }

    fn render_pest_table(&self, block: &Value) -> String {
        let mut lines = vec!["### PEST 分析".to_string(), String::new()];
        for (key, label) in &[("political", "政治"), ("economic", "経済"), ("social", "社会"), ("technological", "技術")] {
            lines.push(format!("**{}:**", label));
            if let Some(items) = block.get(*key).and_then(|v| v.as_array()) {
                for item in items {
                    let text = if let Some(s) = item.as_str() { s.to_string() }
                    else { item.get("title").and_then(|v| v.as_str()).unwrap_or("").to_string() };
                    if !text.is_empty() { lines.push(format!("- {}", text)); }
                }
            }
            lines.push(String::new());
        }
        lines.join("\n")
    }

    fn render_blockquote(&self, block: &Value) -> String {
        if let Some(blocks) = block.get("blocks").and_then(|v| v.as_array()) {
            let content: Vec<String> = blocks.iter().map(|b| self.render_block(b)).collect();
            content.iter().map(|l| format!("> {}", l)).collect::<Vec<_>>().join("\n")
        } else { String::new() }
    }

    fn render_engine_quote(&self, block: &Value) -> String {
        let engine = block.get("engine").and_then(|v| v.as_str()).unwrap_or("unknown");
        let title = block.get("title").and_then(|v| v.as_str()).unwrap_or("");
        let agent_title = ENGINE_AGENT_TITLES.iter()
            .find(|(k, _)| *k == engine).map(|(_, v)| *v).unwrap_or(engine);
        let mut lines = vec![format!("> **{} - {}:**", agent_title, title)];
        if let Some(blocks) = block.get("blocks").and_then(|v| v.as_array()) {
            for b in blocks {
                let rendered = self.render_block(b);
                if !rendered.is_empty() { lines.push(format!("> {}", rendered)); }
            }
        }
        lines.join("\n")
    }

    fn render_code(&self, block: &Value) -> String {
        let lang = block.get("lang").and_then(|v| v.as_str()).unwrap_or("");
        let content = block.get("content").and_then(|v| v.as_str()).unwrap_or("");
        format!("```{}\n{}\n```", lang, content)
    }

    fn render_math(&self, block: &Value) -> String {
        let latex = block.get("latex").and_then(|v| v.as_str()).unwrap_or("");
        format!("$$\n{}\n$$", latex)
    }

    fn render_figure(&self, block: &Value) -> String {
        let img = block.get("img").unwrap_or(&Value::Null);
        let src = img.get("src").and_then(|v| v.as_str()).unwrap_or("");
        let alt = img.get("alt").and_then(|v| v.as_str()).unwrap_or("");
        let caption = block.get("caption").and_then(|v| v.as_str());
        let mut result = format!("![{}]({})", alt, src);
        if let Some(cap) = caption { result.push_str(&format!("\n*{}*", cap)); }
        result
    }

    fn render_callout(&self, block: &Value) -> String {
        let tone = block.get("tone").and_then(|v| v.as_str()).unwrap_or("info");
        let title = block.get("title").and_then(|v| v.as_str()).unwrap_or("");
        let mut lines = vec![format!("> **[{}] {}**", tone.to_uppercase(), title)];
        if let Some(blocks) = block.get("blocks").and_then(|v| v.as_array()) {
            for b in blocks {
                let rendered = self.render_block(b);
                if !rendered.is_empty() { lines.push(format!("> {}", rendered)); }
            }
        }
        lines.join("\n")
    }

    fn render_kpi_grid(&self, block: &Value) -> String {
        let items = match block.get("items").and_then(|v| v.as_array()) {
            Some(items) => items, None => return String::new(),
        };
        let mut lines = vec!["| 指標 | 値 | 変化 |".to_string(), "| --- | --- | --- |".to_string()];
        for item in items {
            let label = item.get("label").and_then(|v| v.as_str()).unwrap_or("");
            let value = item.get("value").and_then(|v| v.as_str()).unwrap_or("");
            let delta = item.get("delta").and_then(|v| v.as_str()).unwrap_or("-");
            lines.push(format!("| {} | {} | {} |", label, value, delta));
        }
        lines.join("\n")
    }

    fn render_widget(&self, block: &Value) -> String {
        let widget_type = block.get("widgetType").and_then(|v| v.as_str()).unwrap_or("unknown");
        format!("[Widget: {} - チャートは HTML レンダリングで表示されます]", widget_type)
    }
}

impl Default for MarkdownRenderer {
    fn default() -> Self { Self::new() }
}

// =============================================================================
// レンダラー: HTML (renderers/html_renderer.py - 簡略版コア構造)
// =============================================================================

/// HTML レンダラー
pub struct HTMLRenderer {
    markdown_renderer: MarkdownRenderer,
}

impl HTMLRenderer {
    pub fn new() -> Self {
        Self { markdown_renderer: MarkdownRenderer::new() }
    }

    /// ドキュメント IR を HTML に変換
    pub fn render(&self, document_ir: &Value) -> String {
        let title = document_ir.get("metadata")
            .and_then(|m| m.get("title"))
            .and_then(|v| v.as_str())
            .unwrap_or("レポート");

        let md_content = self.markdown_renderer.render(document_ir);
        // 簡易 Markdown → HTML 変換
        let html_body = self.markdown_to_html(&md_content);

        format!(r#"<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 900px; margin: 0 auto; padding: 2em; line-height: 1.6; color: #333; }}
h1 {{ border-bottom: 2px solid #4A90E2; padding-bottom: 0.3em; }}
h2 {{ color: #4A90E2; border-bottom: 1px solid #eee; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background-color: #4A90E2; color: white; }}
blockquote {{ border-left: 4px solid #4A90E2; margin: 1em 0; padding: 0.5em 1em; background: #f9f9f9; }}
code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
pre {{ background: #f4f4f4; padding: 1em; border-radius: 5px; overflow-x: auto; }}
</style>
</head>
<body>
{html_body}
</body>
</html>"#)
    }

    fn markdown_to_html(&self, md: &str) -> String {
        let mut html = String::new();
        for line in md.lines() {
            if line.starts_with("# ") {
                html.push_str(&format!("<h1>{}</h1>\n", &line[2..]));
            } else if line.starts_with("## ") {
                html.push_str(&format!("<h2>{}</h2>\n", &line[3..]));
            } else if line.starts_with("### ") {
                html.push_str(&format!("<h3>{}</h3>\n", &line[4..]));
            } else if line.starts_with("- ") {
                html.push_str(&format!("<li>{}</li>\n", &line[2..]));
            } else if line.starts_with("> ") {
                html.push_str(&format!("<blockquote>{}</blockquote>\n", &line[2..]));
            } else if line.starts_with("---") {
                html.push_str("<hr>\n");
            } else if line.starts_with("|") {
                html.push_str(&format!("{}\n", line));
            } else if !line.trim().is_empty() {
                html.push_str(&format!("<p>{}</p>\n", line));
            }
        }
        html
    }
}

impl Default for HTMLRenderer {
    fn default() -> Self { Self::new() }
}

// =============================================================================
// レンダラー: PDF (renderers/pdf_renderer.py - トレイト定義)
// =============================================================================

/// PDF レンダラートレイト
///
/// WeasyPrint の Rust 等価ライブラリが存在しないため、トレイト定義のみ。
#[async_trait]
pub trait PDFRendererTrait: Send + Sync {
    async fn render(&self, document_ir: &Value, output_path: &str) -> Result<bool>;
}

/// PDF レンダラースタブ
pub struct PDFRenderer {
    html_renderer: HTMLRenderer,
}

impl PDFRenderer {
    pub fn new() -> Self {
        Self { html_renderer: HTMLRenderer::new() }
    }

    /// HTML を生成して保存（PDF 変換はスタブ）
    pub fn render_to_html(&self, document_ir: &Value, output_path: &str) -> Result<()> {
        let html = self.html_renderer.render(document_ir);
        fs::write(output_path, html)?;
        Ok(())
    }
}

impl Default for PDFRenderer {
    fn default() -> Self { Self::new() }
}

// =============================================================================
// レンダラー: チャート SVG (renderers/chart_to_svg.py - トレイト定義)
// =============================================================================

/// チャート SVG 変換トレイト
pub trait ChartToSVGConverter: Send + Sync {
    fn convert_widget_to_svg(&self, widget_data: &Value) -> Option<String>;
}

/// デフォルトカラーパレット
pub const DEFAULT_CHART_COLORS: &[&str] = &[
    "#4A90E2", "#E85D75", "#50C878", "#FFB347",
    "#9B59B6", "#3498DB", "#E67E22", "#16A085",
];

// =============================================================================
// レンダラー: Math SVG (renderers/math_to_svg.py - トレイト定義)
// =============================================================================

/// Math SVG 変換トレイト
pub trait MathToSVGConverter: Send + Sync {
    fn convert_to_svg(&self, latex: &str, display_mode: bool) -> Option<String>;
}

// =============================================================================
// ReportEngine 設定
// =============================================================================

#[derive(Debug, Clone, Deserialize)]
pub struct ReportEngineConfig {
    pub api_key: String,
    pub base_url: Option<String>,
    pub model_name: String,
}

impl Default for ReportEngineConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: None,
            model_name: "gemini-2.5-pro".to_string(),
        }
    }
}

// =============================================================================
// ReportAgent メインオーケストレーター (agent.py)
// =============================================================================

/// ReportAgent メインオーケストレーター
pub struct ReportAgent {
    llm_client: LLMClient,
    storage: ChapterStorage,
    composer: DocumentComposer,
    validator: IRValidator,
    html_renderer: HTMLRenderer,
    markdown_renderer: MarkdownRenderer,
    pdf_renderer: PDFRenderer,
    template_selection_node: TemplateSelectionNode,
    document_layout_node: DocumentLayoutNode,
    word_budget_node: WordBudgetNode,
    state: ReportState,
    output_dir: String,
}

impl ReportAgent {
    /// ReportAgent を初期化
    pub fn new(config: Option<Settings>) -> Result<Self> {
        let config = config.unwrap_or_else(Settings::load);

        let llm_client = LLMClient::new(
            &config.report_engine_api_key,
            &config.report_engine_model_name,
            config.report_engine_base_url.as_deref(),
        )?;

        let output_dir = config.output_dir.clone();
        fs::create_dir_all(&output_dir).ok();

        let storage = ChapterStorage::new(&format!("{}/chapters", output_dir));
        let composer = DocumentComposer::new();
        let validator = IRValidator::new();
        let html_renderer = HTMLRenderer::new();
        let markdown_renderer = MarkdownRenderer::new();
        let pdf_renderer = PDFRenderer::new();
        let template_selection_node = TemplateSelectionNode::new(llm_client.clone());
        let document_layout_node = DocumentLayoutNode::new(llm_client.clone());
        let word_budget_node = WordBudgetNode::new(llm_client.clone());

        info!("ReportAgent を初期化しました");
        info!("使用 LLM: {:?}", llm_client.get_model_info());

        Ok(Self {
            llm_client,
            storage,
            composer,
            validator,
            html_renderer,
            markdown_renderer,
            pdf_renderer,
            template_selection_node,
            document_layout_node,
            word_budget_node,
            state: ReportState::default(),
            output_dir,
        })
    }

    /// レポートを生成
    pub async fn generate_report(
        &mut self,
        query: &str,
        reports: &HashMap<String, String>,
        forum_logs: &str,
        template_markdown: Option<&str>,
    ) -> Result<String> {
        self.state.query = query.to_string();
        self.state.mark_processing();

        info!("\n{}", "=".repeat(60));
        info!("ReportEngine: レポート生成開始");
        info!("クエリ: {}", query);
        info!("{}", "=".repeat(60));

        // テンプレートを取得
        let template = template_markdown.unwrap_or("# レポート\n## 概要\n## 分析\n## 結論");

        // Step 1: テンプレート解析
        info!("[Step 1] テンプレートを解析中...");
        let sections = parse_template_sections(template);
        info!("{} セクションを検出", sections.len());

        // Step 2: ドキュメントレイアウト設計
        info!("[Step 2] ドキュメントレイアウトを設計中...");
        let design = self.document_layout_node
            .run(&sections, template, reports, forum_logs, query)
            .await?;
        info!("レイアウト設計完了");

        // Step 3: ワードバジェット計画
        info!("[Step 3] ワードバジェットを計画中...");
        let budget = self.word_budget_node
            .run(&sections, &design, reports, forum_logs, query)
            .await?;
        info!("ワードバジェット計画完了");

        // Step 4: 章の生成
        info!("[Step 4] 章を生成中...");
        let report_id = &self.state.task_id.clone();
        let metadata = serde_json::json!({
            "title": design.get("title").and_then(|v| v.as_str()).unwrap_or(query),
            "query": query,
            "design": design,
            "budget": budget,
        });

        let run_dir = self.storage.start_session(report_id, &metadata)?;

        let context = serde_json::json!({
            "reports": reports,
            "forum_logs": forum_logs,
            "design": design,
            "budget": budget,
        });

        let chapter_gen = ChapterGenerationNode::new(
            self.llm_client.clone(),
            ChapterStorage::new(run_dir.to_str().unwrap_or("chapters")),
        );

        for (i, section) in sections.iter().enumerate() {
            info!("  章 {}/{} を生成中: {}", i + 1, sections.len(), section.title);
            match chapter_gen.run(section, &context, &run_dir).await {
                Ok(_) => info!("  章 {} 生成完了", i + 1),
                Err(e) => {
                    error!("  章 {} 生成失敗: {}", i + 1, e);
                }
            }
        }

        // Step 5: ドキュメント合成
        info!("[Step 5] ドキュメントを合成中...");
        let chapters = self.storage.load_chapters(&run_dir)?;
        let document_ir = self.composer.build_document(report_id, &metadata, &chapters);

        // IR を保存
        let ir_path = PathBuf::from(&self.output_dir).join(format!("{}_ir.json", report_id));
        fs::write(&ir_path, serde_json::to_string_pretty(&document_ir)?)?;
        info!("IR を保存: {}", ir_path.display());

        // Step 6: レンダリング
        info!("[Step 6] レポートをレンダリング中...");

        // HTML
        let html_content = self.html_renderer.render(&document_ir);
        let html_path = PathBuf::from(&self.output_dir).join(format!("{}.html", report_id));
        fs::write(&html_path, &html_content)?;
        self.state.html_content = html_content;
        info!("HTML を保存: {}", html_path.display());

        // Markdown
        let md_content = self.markdown_renderer.render(&document_ir);
        let md_path = PathBuf::from(&self.output_dir).join(format!("{}.md", report_id));
        fs::write(&md_path, &md_content)?;
        self.state.markdown_content = md_content.clone();
        info!("Markdown を保存: {}", md_path.display());

        // 状態更新
        self.state.mark_completed();

        info!("\n{}", "=".repeat(60));
        info!("ReportEngine: レポート生成完了！");
        info!("{}", "=".repeat(60));

        Ok(md_content)
    }

    /// 進捗を取得
    pub fn get_status(&self) -> &ReportState {
        &self.state
    }
}

/// ReportAgent を作成するファクトリ関数
pub fn create_agent() -> Result<ReportAgent> {
    ReportAgent::new(None)
}

/// ReportEngine トレイト
#[async_trait]
pub trait ReportEngineTrait: Send + Sync {
    async fn queue_report(&mut self, query: &str, reports: &[String]) -> Result<String>;
    async fn get_status(&self, task_id: &str) -> Result<Value>;
    async fn download(&self, task_id: &str, format: &str) -> Result<Vec<u8>>;
}

// =============================================================================
// テスト
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ir_validator_valid_chapter() {
        let validator = IRValidator::new();
        let chapter = serde_json::json!({
            "chapterId": "ch1",
            "title": "Chapter 1",
            "anchor": "chapter-1",
            "order": 1,
            "blocks": [
                {
                    "type": "heading",
                    "level": 2,
                    "text": "Test heading"
                },
                {
                    "type": "paragraph",
                    "inlines": [{"text": "Some content"}]
                }
            ]
        });
        let (is_valid, errors) = validator.validate_chapter(&chapter);
        // With the proper fields present it should be valid
        assert!(is_valid, "Expected valid chapter, errors: {:?}", errors);
    }

    #[test]
    fn test_ir_validator_missing_fields() {
        let validator = IRValidator::new();
        let chapter = serde_json::json!({
            "title": "Only title"
        });
        let (is_valid, errors) = validator.validate_chapter(&chapter);
        assert!(!is_valid);
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_ir_validator_empty_blocks() {
        let validator = IRValidator::new();
        let chapter = serde_json::json!({
            "chapterId": "ch1",
            "title": "Chapter 1",
            "anchor": "chapter-1",
            "order": 1,
            "blocks": []
        });
        let (is_valid, errors) = validator.validate_chapter(&chapter);
        assert!(!is_valid);
        assert!(errors.iter().any(|e| e.contains("空")));
    }

    #[test]
    fn test_template_parser() {
        let template = "# Main Title\n\n## Section A\nContent A\n\n## Section B\nContent B";
        let sections = parse_template_sections(template);
        assert!(sections.len() >= 2, "Expected at least 2 sections, got {}", sections.len());
    }

    #[test]
    fn test_template_parser_empty() {
        let sections = parse_template_sections("");
        // Empty template should return empty or minimal sections
        assert!(sections.is_empty() || sections.len() == 1);
    }

    #[test]
    fn test_markdown_renderer() {
        let renderer = MarkdownRenderer::new();
        let doc = serde_json::json!({
            "title": "Test Report",
            "meta": {"generatedAt": "2025-01-01"},
            "chapters": [
                {
                    "chapterId": "ch1",
                    "title": "Chapter One",
                    "anchor": "ch1",
                    "order": 1,
                    "blocks": [
                        {
                            "type": "heading",
                            "level": 2,
                            "runs": [{"text": "Heading"}]
                        },
                        {
                            "type": "paragraph",
                            "runs": [{"text": "Some text here."}]
                        }
                    ]
                }
            ]
        });
        let md = renderer.render(&doc);
        assert!(!md.is_empty());
    }

    #[test]
    fn test_html_renderer() {
        let renderer = HTMLRenderer::new();
        let doc = serde_json::json!({
            "title": "Test",
            "chapters": []
        });
        let html = renderer.render(&doc);
        assert!(!html.is_empty());
    }

    #[test]
    fn test_report_state_lifecycle() {
        let mut state = ReportState::default();
        assert_eq!(state.status, "pending");
        assert!(!state.is_completed());

        state.mark_processing();
        assert_eq!(state.status, "processing");

        state.mark_completed();
        assert!(state.is_completed());

        let json = state.to_json().unwrap();
        assert!(json.contains("completed"));
    }

    #[test]
    fn test_report_state_failure() {
        let mut state = ReportState::default();
        state.mark_failed("Something went wrong");
        assert_eq!(state.status, "failed");
        assert_eq!(state.error_message.as_deref(), Some("Something went wrong"));
    }

    #[test]
    fn test_document_composer() {
        let mut composer = DocumentComposer::new();
        let chapters = vec![
            serde_json::json!({
                "chapterId": "ch1",
                "title": "Chapter 1",
                "anchor": "chapter-1",
                "order": 1,
                "blocks": []
            })
        ];
        let metadata = serde_json::json!({"query": "test"});
        let doc = composer.build_document("report_1", &metadata, &chapters);
        assert!(doc.get("chapters").is_some());
        assert!(doc.get("reportId").is_some());
    }

    #[test]
    fn test_robust_json_parser() {
        let parser = RobustJSONParser::new(true, 3);
        let valid_json = r#"{"key": "value"}"#;
        let result = parser.parse(valid_json, "test", None);
        assert!(result.is_ok());
        assert_eq!(result.unwrap()["key"], "value");
    }

    #[test]
    fn test_robust_json_parser_repair() {
        let parser = RobustJSONParser::new(true, 3);
        let broken_json = r#"{"key": "value",}"#;
        let result = parser.parse(broken_json, "test", None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_allowed_block_types() {
        assert_eq!(ALLOWED_BLOCK_TYPES.len(), 16);
        assert!(ALLOWED_BLOCK_TYPES.contains(&"heading"));
        assert!(ALLOWED_BLOCK_TYPES.contains(&"paragraph"));
        assert!(ALLOWED_BLOCK_TYPES.contains(&"table"));
    }

    #[test]
    fn test_allowed_inline_marks() {
        assert_eq!(ALLOWED_INLINE_MARKS.len(), 12);
        assert!(ALLOWED_INLINE_MARKS.contains(&"bold"));
        assert!(ALLOWED_INLINE_MARKS.contains(&"italic"));
    }
}
