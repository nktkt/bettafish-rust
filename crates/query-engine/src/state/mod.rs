//! QueryEngine 状態管理
//!
//! Python の state/state.py の Rust 実装。
//! 全ての状態データ構造と操作メソッドを定義。

use chrono::Local;
use serde::{Deserialize, Serialize};
use std::fs;

/// 単一検索結果の状態
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Search {
    /// 検索クエリ
    pub query: String,
    /// 検索結果の URL
    pub url: String,
    /// 検索結果のタイトル
    pub title: String,
    /// 検索返却コンテンツ
    pub content: String,
    /// 関連度スコア
    pub score: Option<f64>,
    /// タイムスタンプ
    pub timestamp: String,
}

impl Default for Search {
    fn default() -> Self {
        Self {
            query: String::new(),
            url: String::new(),
            title: String::new(),
            content: String::new(),
            score: None,
            timestamp: Local::now().to_rfc3339(),
        }
    }
}

/// 段落研究プロセスの状態
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Research {
    /// 検索履歴
    pub search_history: Vec<Search>,
    /// 段落の最新サマリー
    pub latest_summary: String,
    /// リフレクション反復回数
    pub reflection_iteration: usize,
    /// 研究完了フラグ
    pub is_completed: bool,
}

impl Default for Research {
    fn default() -> Self {
        Self {
            search_history: Vec::new(),
            latest_summary: String::new(),
            reflection_iteration: 0,
            is_completed: false,
        }
    }
}

impl Research {
    /// 検索記録を追加
    pub fn add_search(&mut self, search: Search) {
        self.search_history.push(search);
    }

    /// 検索結果をバッチ追加
    pub fn add_search_results(
        &mut self,
        query: &str,
        results: &[serde_json::Value],
    ) {
        for result in results {
            let search = Search {
                query: query.to_string(),
                url: result
                    .get("url")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                title: result
                    .get("title")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                content: result
                    .get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                score: result.get("score").and_then(|v| v.as_f64()),
                timestamp: Local::now().to_rfc3339(),
            };
            self.add_search(search);
        }
    }

    /// 検索回数を取得
    pub fn get_search_count(&self) -> usize {
        self.search_history.len()
    }

    /// リフレクション回数をインクリメント
    pub fn increment_reflection(&mut self) {
        self.reflection_iteration += 1;
    }

    /// 完了マーク
    pub fn mark_completed(&mut self) {
        self.is_completed = true;
    }
}

/// レポート中の単一段落の状態
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Paragraph {
    /// 段落タイトル
    pub title: String,
    /// 段落の予定コンテンツ（初期計画）
    pub content: String,
    /// 研究進捗
    pub research: Research,
    /// 段落順序
    pub order: usize,
}

impl Default for Paragraph {
    fn default() -> Self {
        Self {
            title: String::new(),
            content: String::new(),
            research: Research::default(),
            order: 0,
        }
    }
}

impl Paragraph {
    /// 段落が完了したか確認
    pub fn is_completed(&self) -> bool {
        self.research.is_completed && !self.research.latest_summary.is_empty()
    }

    /// 最終コンテンツを取得
    pub fn get_final_content(&self) -> &str {
        if self.research.latest_summary.is_empty() {
            &self.content
        } else {
            &self.research.latest_summary
        }
    }
}

/// レポート全体の状態
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct State {
    /// 元のクエリ
    pub query: String,
    /// レポートタイトル
    pub report_title: String,
    /// 段落リスト
    pub paragraphs: Vec<Paragraph>,
    /// 最終レポートコンテンツ
    pub final_report: String,
    /// 完了フラグ
    pub is_completed: bool,
    /// 作成日時
    pub created_at: String,
    /// 更新日時
    pub updated_at: String,
}

impl Default for State {
    fn default() -> Self {
        let now = Local::now().to_rfc3339();
        Self {
            query: String::new(),
            report_title: String::new(),
            paragraphs: Vec::new(),
            final_report: String::new(),
            is_completed: false,
            created_at: now.clone(),
            updated_at: now,
        }
    }
}

impl State {
    /// 段落を追加
    pub fn add_paragraph(&mut self, title: &str, content: &str) -> usize {
        let order = self.paragraphs.len();
        self.paragraphs.push(Paragraph {
            title: title.to_string(),
            content: content.to_string(),
            research: Research::default(),
            order,
        });
        self.update_timestamp();
        order
    }

    /// 指定インデックスの段落を取得
    pub fn get_paragraph(&self, index: usize) -> Option<&Paragraph> {
        self.paragraphs.get(index)
    }

    /// 指定インデックスの段落を可変参照で取得
    pub fn get_paragraph_mut(&mut self, index: usize) -> Option<&mut Paragraph> {
        self.paragraphs.get_mut(index)
    }

    /// 完了段落数を取得
    pub fn get_completed_paragraphs_count(&self) -> usize {
        self.paragraphs.iter().filter(|p| p.is_completed()).count()
    }

    /// 総段落数を取得
    pub fn get_total_paragraphs_count(&self) -> usize {
        self.paragraphs.len()
    }

    /// 全段落が完了したか確認
    pub fn is_all_paragraphs_completed(&self) -> bool {
        !self.paragraphs.is_empty() && self.paragraphs.iter().all(|p| p.is_completed())
    }

    /// レポート完了マーク
    pub fn mark_completed(&mut self) {
        self.is_completed = true;
        self.update_timestamp();
    }

    /// タイムスタンプ更新
    pub fn update_timestamp(&mut self) {
        self.updated_at = Local::now().to_rfc3339();
    }

    /// 進捗サマリーを取得
    pub fn get_progress_summary(&self) -> serde_json::Value {
        let completed = self.get_completed_paragraphs_count();
        let total = self.get_total_paragraphs_count();
        let percentage = if total > 0 {
            completed as f64 / total as f64 * 100.0
        } else {
            0.0
        };

        serde_json::json!({
            "total_paragraphs": total,
            "completed_paragraphs": completed,
            "progress_percentage": percentage,
            "is_completed": self.is_completed,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        })
    }

    /// JSON 文字列に変換
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    /// JSON 文字列から生成
    pub fn from_json(json_str: &str) -> serde_json::Result<Self> {
        serde_json::from_str(json_str)
    }

    /// ファイルに保存
    pub fn save_to_file(&self, filepath: &str) -> anyhow::Result<()> {
        let json = self.to_json()?;
        fs::write(filepath, json)?;
        Ok(())
    }

    /// ファイルから読み込み
    pub fn load_from_file(filepath: &str) -> anyhow::Result<Self> {
        let content = fs::read_to_string(filepath)?;
        let state: Self = serde_json::from_str(&content)?;
        Ok(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_default() {
        let state = State::default();
        assert!(state.query.is_empty());
        assert!(state.paragraphs.is_empty());
        assert!(!state.is_completed);
    }

    #[test]
    fn test_state_add_paragraph() {
        let mut state = State::default();
        let idx = state.add_paragraph("タイトル1", "内容1");
        assert_eq!(idx, 0);
        assert_eq!(state.paragraphs.len(), 1);
        assert_eq!(state.paragraphs[0].title, "タイトル1");
        assert_eq!(state.paragraphs[0].content, "内容1");
        assert_eq!(state.paragraphs[0].order, 0);

        let idx2 = state.add_paragraph("タイトル2", "内容2");
        assert_eq!(idx2, 1);
        assert_eq!(state.paragraphs.len(), 2);
    }

    #[test]
    fn test_state_get_paragraph() {
        let mut state = State::default();
        state.add_paragraph("p1", "c1");
        state.add_paragraph("p2", "c2");

        let p = state.get_paragraph(0);
        assert!(p.is_some());
        assert_eq!(p.unwrap().title, "p1");

        let p2 = state.get_paragraph(5);
        assert!(p2.is_none());
    }

    #[test]
    fn test_state_completion() {
        let mut state = State::default();
        state.add_paragraph("p1", "c1");
        state.add_paragraph("p2", "c2");

        assert!(!state.is_all_paragraphs_completed());
        assert_eq!(state.get_completed_paragraphs_count(), 0);

        state.paragraphs[0].research.latest_summary = "summary".to_string();
        state.paragraphs[0].research.mark_completed();
        assert_eq!(state.get_completed_paragraphs_count(), 1);
        assert!(!state.is_all_paragraphs_completed());

        state.paragraphs[1].research.latest_summary = "summary2".to_string();
        state.paragraphs[1].research.mark_completed();
        assert!(state.is_all_paragraphs_completed());
    }

    #[test]
    fn test_state_serialization() {
        let mut state = State::default();
        state.query = "test query".to_string();
        state.add_paragraph("p1", "c1");

        let json = state.to_json().unwrap();
        assert!(json.contains("test query"));
        assert!(json.contains("p1"));

        let restored = State::from_json(&json).unwrap();
        assert_eq!(restored.query, "test query");
        assert_eq!(restored.paragraphs.len(), 1);
    }

    #[test]
    fn test_state_progress_summary() {
        let mut state = State::default();
        state.add_paragraph("p1", "c1");
        state.add_paragraph("p2", "c2");

        let summary = state.get_progress_summary();
        assert_eq!(summary["total_paragraphs"], 2);
        assert_eq!(summary["completed_paragraphs"], 0);
    }

    #[test]
    fn test_research_add_search() {
        let mut research = Research::default();
        assert_eq!(research.get_search_count(), 0);

        research.add_search(Search::default());
        assert_eq!(research.get_search_count(), 1);
    }

    #[test]
    fn test_research_add_search_results() {
        let mut research = Research::default();
        let results = vec![
            serde_json::json!({
                "url": "https://example.com",
                "title": "Test",
                "content": "Content",
                "score": 0.95
            })
        ];
        research.add_search_results("query", &results);
        assert_eq!(research.get_search_count(), 1);
        assert_eq!(research.search_history[0].query, "query");
        assert_eq!(research.search_history[0].url, "https://example.com");
    }

    #[test]
    fn test_research_reflection() {
        let mut research = Research::default();
        assert_eq!(research.reflection_iteration, 0);
        research.increment_reflection();
        assert_eq!(research.reflection_iteration, 1);
    }

    #[test]
    fn test_paragraph_completion() {
        let mut para = Paragraph::default();
        assert!(!para.is_completed());

        para.research.mark_completed();
        assert!(!para.is_completed()); // needs summary too

        para.research.latest_summary = "summary".to_string();
        assert!(para.is_completed());
    }

    #[test]
    fn test_paragraph_get_final_content() {
        let mut para = Paragraph::default();
        para.content = "original content".to_string();
        assert_eq!(para.get_final_content(), "original content");

        para.research.latest_summary = "updated summary".to_string();
        assert_eq!(para.get_final_content(), "updated summary");
    }
}
