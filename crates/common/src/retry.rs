//! リトライ機構モジュール
//!
//! Python の retry_helper.py の Rust 実装。
//! 指数バックオフ付きリトライをサポート。

use std::future::Future;
use std::time::Duration;
use tracing::{info, warn, error};

/// リトライ設定
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// 最大リトライ回数
    pub max_retries: usize,
    /// 初期遅延（秒）
    pub initial_delay: f64,
    /// バックオフ係数
    pub backoff_factor: f64,
    /// 最大遅延（秒）
    pub max_delay: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: 1.0,
            backoff_factor: 2.0,
            max_delay: 60.0,
        }
    }
}

/// LLM API 用リトライ設定
pub fn llm_retry_config() -> RetryConfig {
    RetryConfig {
        max_retries: 6,
        initial_delay: 60.0,
        backoff_factor: 2.0,
        max_delay: 600.0,
    }
}

/// 検索 API 用リトライ設定
pub fn search_api_retry_config() -> RetryConfig {
    RetryConfig {
        max_retries: 5,
        initial_delay: 2.0,
        backoff_factor: 1.6,
        max_delay: 25.0,
    }
}

/// DB 用リトライ設定
pub fn db_retry_config() -> RetryConfig {
    RetryConfig {
        max_retries: 5,
        initial_delay: 1.0,
        backoff_factor: 1.5,
        max_delay: 10.0,
    }
}

/// リトライ付きで非同期関数を実行
///
/// 失敗時は指数バックオフで再試行し、最大回数を超えるとエラーを返す。
pub async fn with_retry<F, Fut, T, E>(
    config: &RetryConfig,
    func_name: &str,
    mut f: F,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    E: std::fmt::Display,
{
    let mut last_error: Option<E> = None;

    for attempt in 0..=config.max_retries {
        match f().await {
            Ok(result) => {
                if attempt > 0 {
                    info!("関数 {} が第 {} 回目の試行で成功", func_name, attempt + 1);
                }
                return Ok(result);
            }
            Err(e) => {
                if attempt == config.max_retries {
                    error!(
                        "関数 {} が {} 回の試行後も失敗。最終エラー: {}",
                        func_name,
                        config.max_retries + 1,
                        e
                    );
                    return Err(e);
                }

                let delay = (config.initial_delay * config.backoff_factor.powi(attempt as i32))
                    .min(config.max_delay);

                warn!(
                    "関数 {} 第 {} 回目の試行失敗: {}",
                    func_name,
                    attempt + 1,
                    e
                );
                info!("{:.1} 秒後に第 {} 回目の試行を実行...", delay, attempt + 2);

                tokio::time::sleep(Duration::from_secs_f64(delay)).await;
                last_error = Some(e);
            }
        }
    }

    Err(last_error.unwrap())
}

/// 優雅なリトライ付きで非同期関数を実行（非クリティカル API 用）
///
/// 全リトライ失敗後もパニックせずデフォルト値を返す。
pub async fn with_graceful_retry<F, Fut, T, E>(
    config: &RetryConfig,
    func_name: &str,
    default: T,
    mut f: F,
) -> T
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    E: std::fmt::Display,
    T: Clone,
{
    for attempt in 0..=config.max_retries {
        match f().await {
            Ok(result) => {
                if attempt > 0 {
                    info!(
                        "非クリティカル API {} が第 {} 回目の試行で成功",
                        func_name,
                        attempt + 1
                    );
                }
                return result;
            }
            Err(e) => {
                if attempt == config.max_retries {
                    warn!(
                        "非クリティカル API {} が {} 回の試行後も失敗。最終エラー: {}",
                        func_name,
                        config.max_retries + 1,
                        e
                    );
                    info!("デフォルト値を返してシステムの継続を保証");
                    return default;
                }

                let delay = (config.initial_delay * config.backoff_factor.powi(attempt as i32))
                    .min(config.max_delay);

                warn!(
                    "非クリティカル API {} 第 {} 回目の試行失敗: {}",
                    func_name,
                    attempt + 1,
                    e
                );
                info!("{:.1} 秒後に第 {} 回目の試行を実行...", delay, attempt + 2);

                tokio::time::sleep(Duration::from_secs_f64(delay)).await;
            }
        }
    }

    default
}
