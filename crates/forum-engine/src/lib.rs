//! BettaFish ForumEngine
//!
//! Python の ForumEngine の Rust 実装。
//! LLM パワードのモデレーター（ForumHost）とリアルタイムログ集約（LogMonitor）。

#![allow(dead_code)]

use anyhow::Result;
use chrono::Local;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::io::{Read as _, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tracing::{error, info, warn};

use bettafish_config::Settings;
use bettafish_llm::{InvokeOptions, LLMClient};

/// バージョン情報
pub const VERSION: &str = "1.0.0";

// =============================================================================
// ForumHost - LLM モデレーター
// =============================================================================

/// 論壇主持人
///
/// Qwen3-235B モデルを使用してスマートモデレーターとして機能。
/// 各 Agent の発言を総合して討論を指導する。
pub struct ForumHost {
    llm_client: LLMClient,
    /// 過去のサマリーを追跡（重複防止）
    previous_summaries: Vec<String>,
}

/// 解析済み Agent 発言
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedSpeech {
    pub timestamp: String,
    pub speaker: String,
    pub content: String,
}

/// フォーラムログの解析結果
#[derive(Debug, Clone)]
struct ParsedContent {
    agent_speeches: Vec<ParsedSpeech>,
}

impl ForumHost {
    /// ForumHost を初期化
    pub fn new(config: Option<&Settings>) -> Result<Self> {
        let settings;
        let cfg = match config {
            Some(c) => c,
            None => {
                settings = Settings::load();
                &settings
            }
        };

        if cfg.forum_host_api_key.is_empty() {
            anyhow::bail!("未找到论坛主持人API密钥，请在环境变量文件中设置FORUM_HOST_API_KEY");
        }

        let llm_client = LLMClient::new(
            &cfg.forum_host_api_key,
            &cfg.forum_host_model_name,
            cfg.forum_host_base_url.as_deref(),
        )?;

        info!("ForumHost を初期化しました (model: {})", cfg.forum_host_model_name);

        Ok(Self {
            llm_client,
            previous_summaries: Vec::new(),
        })
    }

    /// Agent 発言からホスト発言を生成
    pub async fn generate_host_speech(&mut self, forum_logs: &[String]) -> Result<Option<String>> {
        // フォーラムログを解析して有効なコンテンツを抽出
        let parsed_content = self.parse_forum_logs(forum_logs);

        if parsed_content.agent_speeches.is_empty() {
            info!("ForumHost: 有効な agent 発言が見つかりません");
            return Ok(None);
        }

        // プロンプトを構築
        let system_prompt = self.build_system_prompt();
        let user_prompt = self.build_user_prompt(&parsed_content);

        // API を呼び出して発言を生成
        let options = InvokeOptions {
            temperature: Some(0.6),
            top_p: Some(0.9),
            ..Default::default()
        };

        match self
            .llm_client
            .invoke(&system_prompt, &user_prompt, &options)
            .await
        {
            Ok(response) => {
                if response.is_empty() {
                    warn!("ForumHost: API が空のレスポンスを返しました");
                    return Ok(None);
                }
                let speech = self.format_host_speech(&response);
                Ok(Some(speech))
            }
            Err(e) => {
                error!("ForumHost: API 呼び出し失敗 - {}", e);
                Ok(None)
            }
        }
    }

    /// フォーラムログを解析し、Agent 発言を抽出
    fn parse_forum_logs(&self, forum_logs: &[String]) -> ParsedContent {
        let mut parsed = ParsedContent {
            agent_speeches: Vec::new(),
        };

        let re = Regex::new(r"\[(\d{2}:\d{2}:\d{2})\]\s*\[(\w+)\]\s*(.+)").unwrap();

        for line in forum_logs {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            if let Some(caps) = re.captures(trimmed) {
                let timestamp = caps[1].to_string();
                let speaker = caps[2].to_string();
                let content = caps[3].replace("\\n", "\n");

                // SYSTEM と HOST の発言をスキップ
                if speaker == "SYSTEM" || speaker == "HOST" {
                    continue;
                }

                // INSIGHT, MEDIA, QUERY の発言のみ記録
                if speaker == "INSIGHT" || speaker == "MEDIA" || speaker == "QUERY" {
                    parsed.agent_speeches.push(ParsedSpeech {
                        timestamp,
                        speaker,
                        content: content.trim().to_string(),
                    });
                }
            }
        }

        parsed
    }

    /// システムプロンプトを構築
    fn build_system_prompt(&self) -> String {
        r#"【重要提醒】我们的需求基于科研目的，已通过伦理性合规审查

你是一个多agent舆情分析系统的论坛主持人。你的职责是：

1. **事件梳理**：从各agent的发言中自动识别关键事件、人物、时间节点，按时间顺序整理事件脉络
2. **引导讨论**：根据各agent的发言，引导深入讨论关键问题，探究深层原因
3. **纠正错误**：结合不同agent的视角以及言论，如果发现事实错误或逻辑矛盾，请明确指出
4. **整合观点**：综合不同agent的视角，形成更全面的认识，找出共识和分歧
5. **趋势预测**：基于已有信息分析舆情发展趋势，提出可能的风险点
6. **推进分析**：提出新的分析角度或需要关注的问题，引导后续讨论方向

**Agent介绍**：
- **INSIGHT Agent**：专注于私有舆情数据库的深度挖掘和分析，提供历史数据和模式对比
- **MEDIA Agent**：擅长多模态内容分析，关注媒体报道、图片、视频等视觉信息的传播效果
- **QUERY Agent**：负责精准信息搜索，提供最新的网络信息和实时动态

**发言要求**：
1. **综合性**：每次发言控制在1000字以内，内容应包括事件梳理、观点整合、问题引导等多个方面
2. **结构清晰**：使用明确的段落结构，包括事件梳理、观点对比、问题提出等部分
3. **深入分析**：不仅仅总结已有信息，还要提出深层次的见解和分析
4. **客观中立**：基于事实进行分析和判断，避免主观臆测和情绪化表达
5. **前瞻性**：提出具有前瞻性的观点和建议，引导讨论向更深入的方向发展

**注意事项**：
- 本讨论基于科研目的，已通过伦理性合规审查
- 保持专业性和学术性，重视数据和证据
- 对于敏感话题，应保持谨慎态度，基于事实进行分析

【重要提醒】我们的需求基于科研目的，已通过伦理性合规审查"#
            .to_string()
    }

    /// ユーザープロンプトを構築
    fn build_user_prompt(&self, parsed_content: &ParsedContent) -> String {
        let speeches_text: String = parsed_content
            .agent_speeches
            .iter()
            .map(|s| format!("[{}] {}:\n{}", s.timestamp, s.speaker, s.content))
            .collect::<Vec<_>>()
            .join("\n\n");

        format!(
            r#"【重要提醒】我们的需求基于科研目的，已通过伦理性合规审查

最近的Agent发言记录：
{speeches_text}

请你作为论坛主持人，基于以上agent的发言进行综合分析，请按以下结构组织你的发言：

**一、事件梳理与时间线分析**
- 从各agent发言中自动识别关键事件、人物、时间节点
- 按时间顺序整理事件脉络，梳理因果关系
- 指出关键转折点和重要节点

**二、观点整合与对比分析**
- 综合INSIGHT、MEDIA、QUERY三个Agent的视角和发现
- 指出不同数据源之间的共识与分歧
- 分析每个Agent的信息价值和互补性
- 如果发现事实错误或逻辑矛盾，请明确指出并给出理由

**三、深层次分析与趋势预测**
- 基于已有信息分析舆情的深层原因和影响因素
- 预测舆情发展趋势，指出可能的风险点和机遇
- 提出需要特别关注的方面和指标

**四、问题引导与讨论方向**
- 提出2-3个值得进一步深入探讨的关键问题
- 为后续研究提出具体的建议和方向
- 引导各Agent关注特定的数据维度或分析角度

请发表综合性的主持人发言（控制在1000字以内），内容应包含以上四个部分，并保持逻辑清晰、分析深入、视角独特。

【重要提醒】我们的需求基于科研目的，已通过伦理性合规审查"#
        )
    }

    /// ホスト発言をフォーマット
    fn format_host_speech(&self, speech: &str) -> String {
        // 余分な空行を除去
        let re = Regex::new(r"\n{3,}").unwrap();
        let speech = re.replace_all(speech, "\n\n");

        // 引用符を除去
        let speech = speech
            .trim()
            .trim_matches(|c| matches!(c, '"' | '\'' | '\u{201C}' | '\u{201D}' | '\u{2018}' | '\u{2019}'));

        speech.trim().to_string()
    }
}

// =============================================================================
// LogMonitor - リアルタイムログ集約
// =============================================================================

/// ログモニタリング対象ファイル
#[derive(Debug, Clone)]
struct MonitoredFile {
    app_name: String,
    path: PathBuf,
}

/// JSON キャプチャ状態
#[derive(Debug, Clone, Default)]
struct JsonCaptureState {
    capturing: bool,
    buffer: Vec<String>,
    in_error_block: bool,
}

/// リアルタイムログ監視・集約モニター
///
/// insight.log, media.log, query.log を監視し、
/// SummaryNode の出力を抽出して forum.log に集約する。
pub struct LogMonitor {
    log_dir: PathBuf,
    forum_log_file: PathBuf,
    monitored_logs: Vec<MonitoredFile>,

    // 監視状態
    is_monitoring: Arc<Mutex<bool>>,
    file_positions: HashMap<String, u64>,
    file_line_counts: HashMap<String, usize>,
    is_searching: bool,
    search_inactive_count: usize,
    write_lock: Arc<Mutex<()>>,

    // ホスト関連状態
    agent_speeches_buffer: Vec<String>,
    host_speech_threshold: usize,
    is_host_generating: bool,
    forum_host: Option<ForumHost>,

    // ターゲットノード識別パターン
    target_node_patterns: Vec<String>,

    // JSON キャプチャ状態 (app_name -> state)
    json_capture_states: HashMap<String, JsonCaptureState>,
}

impl LogMonitor {
    /// LogMonitor を初期化
    pub fn new(log_dir: &str, forum_host: Option<ForumHost>) -> Self {
        let log_dir = PathBuf::from(log_dir);
        let forum_log_file = log_dir.join("forum.log");

        let monitored_logs = vec![
            MonitoredFile {
                app_name: "insight".to_string(),
                path: log_dir.join("insight.log"),
            },
            MonitoredFile {
                app_name: "media".to_string(),
                path: log_dir.join("media.log"),
            },
            MonitoredFile {
                app_name: "query".to_string(),
                path: log_dir.join("query.log"),
            },
        ];

        let target_node_patterns = vec![
            "FirstSummaryNode".to_string(),
            "ReflectionSummaryNode".to_string(),
            "InsightEngine.nodes.summary_node".to_string(),
            "MediaEngine.nodes.summary_node".to_string(),
            "QueryEngine.nodes.summary_node".to_string(),
            "nodes.summary_node".to_string(),
            "\u{6b63}\u{5728}\u{751f}\u{6210}\u{9996}\u{6b21}\u{6bb5}\u{843d}\u{603b}\u{7ed3}".to_string(), // 正在生成首次段落总结
            "\u{6b63}\u{5728}\u{751f}\u{6210}\u{53cd}\u{601d}\u{603b}\u{7ed3}".to_string(), // 正在生成反思总结
        ];

        let mut json_capture_states = HashMap::new();
        for mf in &monitored_logs {
            json_capture_states.insert(mf.app_name.clone(), JsonCaptureState::default());
        }

        // logs ディレクトリを確保
        fs::create_dir_all(&log_dir).ok();

        Self {
            log_dir,
            forum_log_file,
            monitored_logs,
            is_monitoring: Arc::new(Mutex::new(false)),
            file_positions: HashMap::new(),
            file_line_counts: HashMap::new(),
            is_searching: false,
            search_inactive_count: 0,
            write_lock: Arc::new(Mutex::new(())),
            agent_speeches_buffer: Vec::new(),
            host_speech_threshold: 5,
            is_host_generating: false,
            forum_host,
            target_node_patterns,
            json_capture_states,
        }
    }

    /// forum.log をクリア
    pub fn clear_forum_log(&mut self) {
        if self.forum_log_file.exists() {
            fs::remove_file(&self.forum_log_file).ok();
        }

        // 新しい forum.log を作成
        fs::write(&self.forum_log_file, "").ok();

        let start_time = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
        self.write_to_forum_log(
            &format!("=== ForumEngine 監控開始 - {} ===", start_time),
            Some("SYSTEM"),
        );

        info!("ForumEngine: forum.log をクリアし初期化しました");

        // JSON キャプチャ状態をリセット
        for state in self.json_capture_states.values_mut() {
            *state = JsonCaptureState::default();
        }

        // ホスト関連状態をリセット
        self.agent_speeches_buffer.clear();
        self.is_host_generating = false;
    }

    /// forum.log に書き込み（スレッドセーフ）
    pub fn write_to_forum_log(&self, content: &str, source: Option<&str>) {
        let _lock = self.write_lock.lock().ok();

        match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.forum_log_file)
        {
            Ok(mut f) => {
                use std::io::Write;
                let timestamp = Local::now().format("%H:%M:%S").to_string();
                // 実際の改行を \\n に変換して1行にする
                let content_one_line = content.replace('\n', "\\n").replace('\r', "\\r");
                let line = if let Some(src) = source {
                    format!("[{}] [{}] {}\n", timestamp, src, content_one_line)
                } else {
                    format!("[{}] {}\n", timestamp, content_one_line)
                };
                let _ = f.write_all(line.as_bytes());
                let _ = f.flush();
            }
            Err(e) => {
                error!("ForumEngine: forum.log への書き込みに失敗: {}", e);
            }
        }
    }

    /// ログ行のレベルを検出
    fn get_log_level(&self, line: &str) -> Option<String> {
        let re = Regex::new(r"\|\s*(INFO|ERROR|WARNING|DEBUG|TRACE|CRITICAL)\s*\|").unwrap();
        re.captures(line).map(|caps| caps[1].to_string())
    }

    /// ターゲットログ行（SummaryNode）かチェック
    fn is_target_log_line(&self, line: &str) -> bool {
        // ERROR レベルを除外
        if let Some(ref level) = self.get_log_level(line) {
            if level == "ERROR" {
                return false;
            }
        }

        if line.contains("| ERROR") || line.contains("| ERROR    |") {
            return false;
        }

        // エラーキーワードを除外
        let error_keywords = ["JSON解析失败", "JSON修复失败", "Traceback", "File \""];
        for kw in &error_keywords {
            if line.contains(kw) {
                return false;
            }
        }

        // ターゲットパターンをチェック
        for pattern in &self.target_node_patterns {
            if line.contains(pattern.as_str()) {
                return true;
            }
        }

        false
    }

    /// 有価値なコンテンツかどうかを判定
    fn is_valuable_content(&self, line: &str) -> bool {
        if line.contains("\u{6e05}\u{7406}\u{540e}\u{7684}\u{8f93}\u{51fa}") {
            // "清理后的输出"
            return true;
        }

        let exclude_patterns = [
            "JSON解析失败",
            "JSON修复失败",
            "直接使用清理后的文本",
            "JSON解析成功",
            "成功生成",
            "已更新段落",
            "正在生成",
            "开始处理",
            "处理完成",
            "已读取HOST发言",
            "读取HOST发言失败",
            "未找到HOST发言",
            "调试输出",
            "信息记录",
        ];

        for pattern in &exclude_patterns {
            if line.contains(pattern) {
                return false;
            }
        }

        // タイムスタンプを除去して長さチェック
        let re_old = Regex::new(r"\[\d{2}:\d{2}:\d{2}\]").unwrap();
        let re_new = Regex::new(
            r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}\s*\|\s*[A-Z]+\s*\|[^|]+?\s*-\s*",
        )
        .unwrap();
        let clean = re_old.replace_all(line, "");
        let clean = re_new.replace_all(&clean, "");
        let clean = clean.trim();

        clean.len() >= 30
    }

    /// JSON 開始行かどうかをチェック
    fn is_json_start_line(&self, line: &str) -> bool {
        // "清理后的输出: {"
        line.contains("\u{6e05}\u{7406}\u{540e}\u{7684}\u{8f93}\u{51fa}: {")
    }

    /// JSON コンテンツをフォーマット
    fn format_json_content(&self, json_obj: &Value) -> Option<String> {
        // updated_paragraph_latest_state を優先、次に paragraph_latest_state
        if let Some(content) = json_obj
            .get("updated_paragraph_latest_state")
            .and_then(|v| v.as_str())
        {
            if !content.is_empty() {
                return Some(content.to_string());
            }
        }

        if let Some(content) = json_obj
            .get("paragraph_latest_state")
            .and_then(|v| v.as_str())
        {
            if !content.is_empty() {
                return Some(content.to_string());
            }
        }

        // 予期したフィールドが無い場合は JSON 全体を返す
        Some(format!(
            "\u{6e05}\u{7406}\u{540e}\u{7684}\u{8f93}\u{51fa}: {}",
            serde_json::to_string_pretty(json_obj).unwrap_or_default()
        ))
    }

    /// 複数行から JSON コンテンツを抽出
    fn extract_json_content(&self, json_lines: &[String]) -> Option<String> {
        // "清理后的输出: {" を含む行を探す
        let marker = "\u{6e05}\u{7406}\u{540e}\u{7684}\u{8f93}\u{51fa}: {";
        let json_start_idx = json_lines.iter().position(|l| l.contains(marker))?;

        let first_line = &json_lines[json_start_idx];
        let json_start_pos = first_line.find(marker)?;
        let json_part_start = json_start_pos + marker.len() - 1; // '{' を含める
        let json_part = &first_line[json_part_start..];

        // 単一行 JSON チェック
        if json_part.trim().ends_with('}')
            && json_part.matches('{').count() == json_part.matches('}').count()
        {
            if let Ok(obj) = serde_json::from_str::<Value>(json_part.trim()) {
                return self.format_json_content(&obj);
            }
        }

        // 複数行 JSON
        let re_old = Regex::new(r"^\[\d{2}:\d{2}:\d{2}\]\s*").unwrap();
        let re_new = Regex::new(
            r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}\s*\|\s*[A-Z]+\s*\|[^|]+?\s*-\s*",
        )
        .unwrap();

        let mut json_text = json_part.to_string();
        for line in &json_lines[json_start_idx + 1..] {
            let clean = re_old.replace_all(line, "");
            let clean = re_new.replace_all(&clean, "");
            json_text.push_str(&clean);
        }

        if let Ok(obj) = serde_json::from_str::<Value>(json_text.trim()) {
            return self.format_json_content(&obj);
        }

        None
    }

    /// ノードコンテンツを抽出（タイムスタンプ等の接頭辞を除去）
    fn extract_node_content(&self, line: &str) -> String {
        let mut content = line.to_string();

        // 旧フォーマットのタイムスタンプを除去
        let re_old = Regex::new(r"\[\d{2}:\d{2}:\d{2}\]\s*(.+)").unwrap();
        if let Some(caps) = re_old.captures(&content) {
            content = caps[1].trim().to_string();
        } else {
            // 新フォーマット(loguru)のタイムスタンプを除去
            let re_new = Regex::new(
                r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}\s*\|\s*[A-Z]+\s*\|[^|]+?\s*-\s*(.+)",
            )
            .unwrap();
            if let Some(caps) = re_new.captures(&content) {
                content = caps[1].trim().to_string();
            }
        }

        // 角括弧タグを除去
        let re_bracket = Regex::new(r"^\[.*?\]\s*").unwrap();
        while re_bracket.is_match(&content) {
            content = re_bracket.replace(&content, "").to_string();
        }

        // 共通接頭辞を除去
        let prefixes = [
            "\u{9996}\u{6b21}\u{603b}\u{7ed3}: ",   // 首次总结:
            "\u{53cd}\u{601d}\u{603b}\u{7ed3}: ",   // 反思总结:
            "\u{6e05}\u{7406}\u{540e}\u{7684}\u{8f93}\u{51fa}: ", // 清理后的输出:
        ];

        for prefix in &prefixes {
            if content.starts_with(prefix) {
                content = content[prefix.len()..].to_string();
                break;
            }
        }

        // APP 名称タグを除去
        let app_names = ["INSIGHT", "MEDIA", "QUERY"];
        for name in &app_names {
            let re_app = Regex::new(&format!(r"(?i)^{}\s+", name)).unwrap();
            content = re_app.replace(&content, "").to_string();
        }

        // 余分なスペースをクリーン
        let re_spaces = Regex::new(r"\s+").unwrap();
        content = re_spaces.replace_all(&content, " ").to_string();

        content.trim().to_string()
    }

    /// コンテンツ中の重複タグをクリーン
    fn clean_content_tags(&self, content: &str, _app_name: &str) -> String {
        if content.is_empty() {
            return content.to_string();
        }

        let mut result = content.to_string();
        let all_app_names = ["INSIGHT", "MEDIA", "QUERY"];

        for name in &all_app_names {
            // [APP_NAME] フォーマットを除去
            let re = Regex::new(&format!(r"(?i)\[{}\]\s*", name)).unwrap();
            result = re.replace_all(&result, "").to_string();
            // 単独の APP_NAME を除去
            let re2 = Regex::new(&format!(r"(?i)^{}\s+", name)).unwrap();
            result = re2.replace_all(&result, "").to_string();
        }

        // その他の角括弧タグを除去
        let re_bracket = Regex::new(r"^\[.*?\]\s*").unwrap();
        result = re_bracket.replace(&result, "").to_string();

        // 重複スペースを除去
        let re_spaces = Regex::new(r"\s+").unwrap();
        result = re_spaces.replace_all(&result, " ").to_string();

        result.trim().to_string()
    }

    /// ファイルサイズを取得
    fn get_file_size(&self, file_path: &Path) -> u64 {
        file_path
            .metadata()
            .map(|m| m.len())
            .unwrap_or(0)
    }

    /// ファイル行数を取得
    fn get_file_line_count(&self, file_path: &Path) -> usize {
        if !file_path.exists() {
            return 0;
        }
        fs::read_to_string(file_path)
            .map(|c| c.lines().count())
            .unwrap_or(0)
    }

    /// ファイルから新しい行を読み取り
    fn read_new_lines(&mut self, file_path: &Path, app_name: &str) -> Vec<String> {
        let mut new_lines = Vec::new();

        if !file_path.exists() {
            return new_lines;
        }

        let current_size = self.get_file_size(file_path);
        let last_position = *self.file_positions.get(app_name).unwrap_or(&0);

        // ファイルが縮小された場合（クリアされた）、先頭からやり直し
        if current_size < last_position {
            self.file_positions.insert(app_name.to_string(), 0);
            if let Some(state) = self.json_capture_states.get_mut(app_name) {
                *state = JsonCaptureState::default();
            }
        }

        let last_position = *self.file_positions.get(app_name).unwrap_or(&0);

        if current_size > last_position {
            match fs::File::open(file_path) {
                Ok(mut f) => {
                    if f.seek(SeekFrom::Start(last_position)).is_ok() {
                        let mut content = String::new();
                        if f.read_to_string(&mut content).is_ok() {
                            self.file_positions.insert(
                                app_name.to_string(),
                                last_position + content.len() as u64,
                            );
                            new_lines = content
                                .split('\n')
                                .filter(|l| !l.trim().is_empty())
                                .map(|l| l.trim().to_string())
                                .collect();
                        }
                    }
                }
                Err(e) => {
                    error!("ForumEngine: {} ログの読み取りに失敗: {}", app_name, e);
                }
            }
        }

        new_lines
    }

    /// 行を処理して複数行 JSON コンテンツをキャプチャ
    fn process_lines_for_json(&mut self, lines: &[String], app_name: &str) -> Vec<String> {
        let mut captured_contents = Vec::new();

        // キャプチャ状態をマップから取り出してローカルで操作（借用競合を回避）
        let mut state = self
            .json_capture_states
            .remove(app_name)
            .unwrap_or_default();

        for line in lines {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            // ログレベルをチェックして ERROR ブロック状態を更新
            let log_level = {
                let re = Regex::new(r"\|\s*(INFO|ERROR|WARNING|DEBUG|TRACE|CRITICAL)\s*\|")
                    .unwrap();
                re.captures(trimmed).map(|caps| caps[1].to_string())
            };

            if let Some(ref level) = log_level {
                if level == "ERROR" {
                    state.in_error_block = true;
                    if state.capturing {
                        state.capturing = false;
                        state.buffer.clear();
                    }
                    continue;
                } else if level == "INFO" {
                    state.in_error_block = false;
                }
            }

            if state.in_error_block {
                if state.capturing {
                    state.capturing = false;
                    state.buffer.clear();
                }
                continue;
            }

            let is_target = self.is_target_log_line(trimmed);
            let is_json_start = self.is_json_start_line(trimmed);

            if is_target && is_json_start {
                state.capturing = true;
                state.buffer = vec![trimmed.to_string()];

                // 単行 JSON チェック
                if trimmed.ends_with('}') {
                    let buffer_clone = state.buffer.clone();
                    if let Some(content) = self.extract_json_content(&buffer_clone) {
                        let clean = self.clean_content_tags(&content, app_name);
                        captured_contents.push(clean);
                    }
                    state.capturing = false;
                    state.buffer.clear();
                }
            } else if is_target && self.is_valuable_content(trimmed) {
                let node_content = self.extract_node_content(trimmed);
                let clean = self.clean_content_tags(&node_content, app_name);
                captured_contents.push(clean);
            } else if state.capturing {
                state.buffer.push(trimmed.to_string());

                // JSON 終了チェック
                let re_old = Regex::new(r"^\[\d{2}:\d{2}:\d{2}\]\s*").unwrap();
                let re_new = Regex::new(
                    r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}\s*\|\s*[A-Z]+\s*\|[^|]+?\s*-\s*",
                )
                .unwrap();
                let cleaned_line = re_old.replace_all(trimmed, "");
                let cleaned_line = re_new.replace_all(&cleaned_line, "");
                let cleaned_line = cleaned_line.trim();

                if cleaned_line == "}" || cleaned_line == "] }" {
                    let buffer_clone = state.buffer.clone();
                    if let Some(content) = self.extract_json_content(&buffer_clone) {
                        let clean = self.clean_content_tags(&content, app_name);
                        captured_contents.push(clean);
                    }
                    state.capturing = false;
                    state.buffer.clear();
                }
            }
        }

        // 状態をマップに戻す
        self.json_capture_states.insert(app_name.to_string(), state);

        captured_contents
    }

    /// ホスト発言をトリガー
    async fn trigger_host_speech(&mut self) {
        if self.forum_host.is_none() || self.is_host_generating {
            return;
        }

        let recent_speeches: Vec<String> = self
            .agent_speeches_buffer
            .iter()
            .take(self.host_speech_threshold)
            .cloned()
            .collect();

        if recent_speeches.len() < self.host_speech_threshold {
            return;
        }

        self.is_host_generating = true;
        info!("ForumEngine: ホスト発言を生成中...");

        if let Some(ref mut host) = self.forum_host {
            match host.generate_host_speech(&recent_speeches).await {
                Ok(Some(speech)) => {
                    self.write_to_forum_log(&speech, Some("HOST"));
                    info!("ForumEngine: ホスト発言を記録しました");
                    // 処理済みの発言をクリア
                    if self.agent_speeches_buffer.len() >= self.host_speech_threshold {
                        self.agent_speeches_buffer =
                            self.agent_speeches_buffer[self.host_speech_threshold..].to_vec();
                    }
                }
                Ok(None) => {
                    warn!("ForumEngine: ホスト発言の生成に失敗しました");
                }
                Err(e) => {
                    error!("ForumEngine: ホスト発言トリガー中にエラー: {}", e);
                }
            }
        }

        self.is_host_generating = false;
    }

    /// ログ監視ループを1回実行（非同期）
    pub async fn monitor_tick(&mut self) -> bool {
        if !*self.is_monitoring.lock().unwrap() {
            return false;
        }

        let mut any_growth = false;
        let mut any_shrink = false;
        let mut captured_any = false;

        let monitored: Vec<(String, PathBuf)> = self
            .monitored_logs
            .iter()
            .map(|mf| (mf.app_name.clone(), mf.path.clone()))
            .collect();

        for (app_name, log_file) in &monitored {
            let current_lines = self.get_file_line_count(log_file);
            let previous_lines = *self.file_line_counts.get(app_name.as_str()).unwrap_or(&0);

            if current_lines > previous_lines {
                any_growth = true;
                let new_lines = self.read_new_lines(log_file, app_name);

                // 初回 FirstSummaryNode 検出
                if !self.is_searching {
                    for line in &new_lines {
                        if !line.trim().is_empty()
                            && self.is_target_log_line(line)
                            && (line.contains("FirstSummaryNode")
                                || line.contains("\u{6b63}\u{5728}\u{751f}\u{6210}\u{9996}\u{6b21}\u{6bb5}\u{843d}\u{603b}\u{7ed3}"))
                        {
                            info!(
                                "ForumEngine: {} 中に初回フォーラム発表コンテンツを検出",
                                app_name
                            );
                            self.is_searching = true;
                            self.search_inactive_count = 0;
                            self.clear_forum_log();
                            break;
                        }
                    }
                }

                // 検索状態中のコンテンツ処理
                if self.is_searching {
                    let captured_contents = self.process_lines_for_json(&new_lines, app_name);

                    for content in &captured_contents {
                        let source_tag = app_name.to_uppercase();
                        self.write_to_forum_log(content, Some(&source_tag));
                        captured_any = true;

                        // バッファに追加
                        let timestamp = Local::now().format("%H:%M:%S").to_string();
                        let log_line =
                            format!("[{}] [{}] {}", timestamp, source_tag, content);
                        self.agent_speeches_buffer.push(log_line);

                        // ホスト発言トリガーチェック
                        if self.agent_speeches_buffer.len() >= self.host_speech_threshold
                            && !self.is_host_generating
                        {
                            self.trigger_host_speech().await;
                        }
                    }
                }
            } else if current_lines < previous_lines {
                any_shrink = true;
                self.file_positions.insert(
                    app_name.to_string(),
                    self.get_file_size(log_file),
                );
                if let Some(state) = self.json_capture_states.get_mut(app_name.as_str()) {
                    *state = JsonCaptureState::default();
                }
            }

            self.file_line_counts
                .insert(app_name.to_string(), current_lines);
        }

        // 検索セッション終了チェック
        if self.is_searching {
            if any_shrink {
                self.is_searching = false;
                self.search_inactive_count = 0;
                self.agent_speeches_buffer.clear();
                self.is_host_generating = false;
                let end_time = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
                self.write_to_forum_log(
                    &format!("=== ForumEngine 論壇終了 - {} ===", end_time),
                    Some("SYSTEM"),
                );
            } else if !any_growth && !captured_any {
                self.search_inactive_count += 1;
                if self.search_inactive_count >= 7200 {
                    info!("ForumEngine: 長時間無活動、フォーラムを終了");
                    self.is_searching = false;
                    self.search_inactive_count = 0;
                    self.agent_speeches_buffer.clear();
                    self.is_host_generating = false;
                    let end_time = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
                    self.write_to_forum_log(
                        &format!("=== ForumEngine 論壇終了 - {} ===", end_time),
                        Some("SYSTEM"),
                    );
                }
            } else {
                self.search_inactive_count = 0;
            }
        }

        true
    }

    /// 監視を開始（非同期ループ）
    pub async fn start_monitoring(&mut self) -> Result<()> {
        {
            let mut monitoring = self.is_monitoring.lock().unwrap();
            if *monitoring {
                info!("ForumEngine: フォーラムは既に実行中です");
                return Ok(());
            }
            *monitoring = true;
        }

        info!("ForumEngine: フォーラム論壇を作成中...");

        // ファイルのベースラインを初期化
        let monitored: Vec<(String, PathBuf)> = self
            .monitored_logs
            .iter()
            .map(|mf| (mf.app_name.clone(), mf.path.clone()))
            .collect();

        for (app_name, log_file) in &monitored {
            self.file_line_counts
                .insert(app_name.clone(), self.get_file_line_count(log_file));
            self.file_positions
                .insert(app_name.clone(), self.get_file_size(log_file));
        }

        info!("ForumEngine: フォーラムを開始しました");

        // 監視ループ
        loop {
            if !self.monitor_tick().await {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        }

        info!("ForumEngine: フォーラムログファイルの監視を停止しました");
        Ok(())
    }

    /// 監視を停止
    pub fn stop_monitoring(&self) {
        let mut monitoring = self.is_monitoring.lock().unwrap();
        if !*monitoring {
            info!("ForumEngine: フォーラムは実行していません");
            return;
        }

        *monitoring = false;

        // 終了マーカーを書き込み
        let end_time = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
        self.write_to_forum_log(
            &format!("=== ForumEngine 論壇終了 - {} ===", end_time),
            Some("SYSTEM"),
        );

        info!("ForumEngine: フォーラムを停止しました");
    }

    /// forum.log の内容を取得
    pub fn get_forum_log_content(&self) -> Vec<String> {
        if !self.forum_log_file.exists() {
            return Vec::new();
        }

        match fs::read_to_string(&self.forum_log_file) {
            Ok(content) => content
                .lines()
                .map(|l| l.trim_end_matches(&['\n', '\r'][..]).to_string())
                .collect(),
            Err(e) => {
                error!("ForumEngine: forum.log の読み取りに失敗: {}", e);
                Vec::new()
            }
        }
    }
}

// =============================================================================
// グローバルインスタンス管理
// =============================================================================

/// ForumHost インスタンスを作成
pub fn create_forum_host(config: Option<&Settings>) -> Result<ForumHost> {
    ForumHost::new(config)
}

/// LogMonitor インスタンスを作成
pub fn create_log_monitor(log_dir: &str, forum_host: Option<ForumHost>) -> LogMonitor {
    LogMonitor::new(log_dir, forum_host)
}
