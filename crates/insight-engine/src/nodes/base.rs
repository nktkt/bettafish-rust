//! ノード基底定義
//!
//! Python の base_node.py の Rust 実装。

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;

use crate::state::State;

/// ノード基底トレイト
#[async_trait]
pub trait Node: Send + Sync {
    /// ノード名を返す
    fn name(&self) -> &str;

    /// ノード処理を実行
    async fn run(&self, input_data: &Value) -> Result<Value>;

    /// 入力データを検証
    fn validate_input(&self, input_data: &Value) -> bool {
        let _ = input_data;
        true
    }

    /// 出力データを後処理
    #[allow(dead_code)]
    fn process_output(&self, output: &str) -> Result<Value> {
        let _ = output;
        Ok(Value::Null)
    }
}

/// 状態変更機能付きノードトレイト
#[async_trait]
pub trait StateMutationNode: Node {
    /// 状態を変更
    async fn mutate_state(
        &self,
        input_data: &Value,
        state: &mut State,
        paragraph_index: usize,
    ) -> Result<()>;
}
