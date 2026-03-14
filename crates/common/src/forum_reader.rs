//! フォーラムログリーダー
//!
//! Python の forum_reader.py の Rust 実装。
//! forum.log から最新の HOST 発言を読み取る。

use regex::Regex;
use std::fs;
use std::path::Path;
use tracing::{debug, error, info};

/// forum.log から最新の HOST 発言を取得
pub fn get_latest_host_speech(log_dir: &str) -> Option<String> {
    let forum_log_path = Path::new(log_dir).join("forum.log");

    if !forum_log_path.exists() {
        debug!("forum.log ファイルが存在しません");
        return None;
    }

    let content = match fs::read_to_string(&forum_log_path) {
        Ok(c) => c,
        Err(e) => {
            error!("forum.log の読み取りに失敗: {}", e);
            return None;
        }
    };

    let re = Regex::new(r"\[(\d{2}:\d{2}:\d{2})\]\s*\[HOST\]\s*(.+)").unwrap();

    // 後ろから検索して最新の HOST 発言を見つける
    for line in content.lines().rev() {
        if let Some(caps) = re.captures(line) {
            let speech = caps[2].replace("\\n", "\n").trim().to_string();
            info!("最新の HOST 発言を発見、長さ: {} 文字", speech.len());
            return Some(speech);
        }
    }

    debug!("HOST 発言が見つかりません");
    None
}

/// forum.log から全ての HOST 発言を取得
pub fn get_all_host_speeches(log_dir: &str) -> Vec<HostSpeech> {
    let forum_log_path = Path::new(log_dir).join("forum.log");

    if !forum_log_path.exists() {
        debug!("forum.log ファイルが存在しません");
        return Vec::new();
    }

    let content = match fs::read_to_string(&forum_log_path) {
        Ok(c) => c,
        Err(e) => {
            error!("forum.log の読み取りに失敗: {}", e);
            return Vec::new();
        }
    };

    let re = Regex::new(r"\[(\d{2}:\d{2}:\d{2})\]\s*\[HOST\]\s*(.+)").unwrap();
    let mut speeches = Vec::new();

    for line in content.lines() {
        if let Some(caps) = re.captures(line) {
            speeches.push(HostSpeech {
                timestamp: caps[1].to_string(),
                content: caps[2].replace("\\n", "\n").trim().to_string(),
            });
        }
    }

    info!("{} 件の HOST 発言を発見", speeches.len());
    speeches
}

/// 最近の Agent 発言を取得（HOST を除く）
pub fn get_recent_agent_speeches(log_dir: &str, limit: usize) -> Vec<AgentSpeech> {
    let forum_log_path = Path::new(log_dir).join("forum.log");

    if !forum_log_path.exists() {
        return Vec::new();
    }

    let content = match fs::read_to_string(&forum_log_path) {
        Ok(c) => c,
        Err(e) => {
            error!("forum.log の読み取りに失敗: {}", e);
            return Vec::new();
        }
    };

    let re = Regex::new(r"\[(\d{2}:\d{2}:\d{2})\]\s*\[(INSIGHT|MEDIA|QUERY)\]\s*(.+)").unwrap();
    let mut speeches = Vec::new();

    for line in content.lines().rev() {
        if let Some(caps) = re.captures(line) {
            speeches.push(AgentSpeech {
                timestamp: caps[1].to_string(),
                agent: caps[2].to_string(),
                content: caps[3].replace("\\n", "\n").trim().to_string(),
            });
            if speeches.len() >= limit {
                break;
            }
        }
    }

    speeches.reverse();
    speeches
}

/// HOST 発言をプロンプト用にフォーマット
pub fn format_host_speech_for_prompt(host_speech: &str) -> String {
    if host_speech.is_empty() {
        return String::new();
    }

    format!(
        "\n### 論坛主持人最新総括\n\
         以下は論壇主持人による各 Agent 討論の最新総括と指導です。その中の観点と提案を参考にしてください：\n\n\
         {}\n\n---\n",
        host_speech
    )
}

/// HOST 発言データ
#[derive(Debug, Clone)]
pub struct HostSpeech {
    pub timestamp: String,
    pub content: String,
}

/// Agent 発言データ
#[derive(Debug, Clone)]
pub struct AgentSpeech {
    pub timestamp: String,
    pub agent: String,
    pub content: String,
}
