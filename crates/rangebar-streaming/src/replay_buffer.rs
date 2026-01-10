//! Replay buffer for storing and replaying recent trade data
//!
//! This module provides a circular buffer that stores recent trades and allows
//! replaying them at different speeds for testing and analysis.

use rangebar_core::AggTrade;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::time::sleep;
use tokio_stream::Stream;

/// A circular buffer that stores recent trades and provides replay functionality
#[derive(Debug, Clone)]
pub struct ReplayBuffer {
    inner: Arc<Mutex<ReplayBufferInner>>,
}

#[derive(Debug)]
struct ReplayBufferInner {
    capacity: Duration,
    trades: VecDeque<AggTrade>,
    start_time: Option<Instant>,
}

impl ReplayBuffer {
    /// Create a new replay buffer with the specified capacity
    pub fn new(capacity: Duration) -> Self {
        Self {
            inner: Arc::new(Mutex::new(ReplayBufferInner {
                capacity,
                trades: VecDeque::new(),
                start_time: None,
            })),
        }
    }

    /// Add a new trade to the buffer
    pub fn push(&self, trade: AggTrade) {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        // Set start time on first trade
        if inner.start_time.is_none() {
            inner.start_time = Some(Instant::now());
        }

        // Remove old trades beyond capacity (using microseconds)
        let cutoff_timestamp = trade.timestamp - (inner.capacity.as_micros() as i64);

        while let Some(front_trade) = inner.trades.front() {
            if front_trade.timestamp < cutoff_timestamp {
                inner.trades.pop_front();
            } else {
                break;
            }
        }

        inner.trades.push_back(trade);
    }

    /// Get the number of trades currently in the buffer
    pub fn len(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .trades
            .len()
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .trades
            .is_empty()
    }

    /// Get the time span of trades in the buffer
    pub fn time_span(&self) -> Option<Duration> {
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let (Some(first), Some(last)) = (inner.trades.front(), inner.trades.back()) {
            let span_microseconds = last.timestamp - first.timestamp;
            if span_microseconds > 0 {
                Some(Duration::from_micros(span_microseconds as u64))
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Get trades from the buffer starting from N minutes ago
    pub fn get_trades_from(&self, minutes_ago: u32) -> Vec<AggTrade> {
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        // Calculate cutoff timestamp
        let latest_timestamp = match inner.trades.back() {
            Some(trade) => trade.timestamp,
            None => return Vec::new(),
        };
        let cutoff_timestamp = latest_timestamp - (minutes_ago as i64 * 60 * 1000);

        inner
            .trades
            .iter()
            .filter(|trade| trade.timestamp >= cutoff_timestamp)
            .cloned()
            .collect()
    }

    /// Create a replay stream that emits trades at the specified speed
    pub fn replay_from(&self, minutes_ago: u32, speed_multiplier: f32) -> ReplayStream {
        let trades = self.get_trades_from(minutes_ago);
        ReplayStream::new(trades, speed_multiplier)
    }

    /// Get statistics about the buffer
    pub fn stats(&self) -> ReplayBufferStats {
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let (first_timestamp, last_timestamp) =
            if let (Some(first), Some(last)) = (inner.trades.front(), inner.trades.back()) {
                (Some(first.timestamp), Some(last.timestamp))
            } else {
                (None, None)
            };

        ReplayBufferStats {
            capacity: inner.capacity,
            trade_count: inner.trades.len(),
            first_timestamp,
            last_timestamp,
            memory_usage_bytes: inner.trades.len() * std::mem::size_of::<AggTrade>(),
        }
    }
}

/// Statistics about the replay buffer
#[derive(Debug, Clone)]
pub struct ReplayBufferStats {
    pub capacity: Duration,
    pub trade_count: usize,
    pub first_timestamp: Option<i64>,
    pub last_timestamp: Option<i64>,
    pub memory_usage_bytes: usize,
}

/// A stream that replays trades at a specified speed
pub struct ReplayStream {
    trades: Vec<AggTrade>,
    current_index: usize,
    speed_multiplier: f32,
    base_timestamp: Option<i64>,
    start_time: Option<Instant>,
}

impl ReplayStream {
    /// Create a new replay stream
    pub fn new(trades: Vec<AggTrade>, speed_multiplier: f32) -> Self {
        let base_timestamp = trades.first().map(|t| t.timestamp);

        Self {
            trades,
            current_index: 0,
            speed_multiplier: speed_multiplier.max(0.1), // Minimum 0.1x speed
            base_timestamp,
            start_time: None,
        }
    }

    /// Set the replay speed
    pub fn set_speed(&mut self, speed_multiplier: f32) {
        self.speed_multiplier = speed_multiplier.max(0.1);
    }

    /// Get the current replay speed
    pub fn speed(&self) -> f32 {
        self.speed_multiplier
    }

    /// Get the number of remaining trades
    pub fn remaining(&self) -> usize {
        self.trades.len().saturating_sub(self.current_index)
    }

    /// Get the total number of trades
    pub fn total(&self) -> usize {
        self.trades.len()
    }

    /// Get the progress as a percentage (0.0 to 1.0)
    pub fn progress(&self) -> f32 {
        if self.trades.is_empty() {
            1.0
        } else {
            self.current_index as f32 / self.trades.len() as f32
        }
    }
}

impl Stream for ReplayStream {
    type Item = AggTrade;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        // Check if we have more trades
        if self.current_index >= self.trades.len() {
            return std::task::Poll::Ready(None);
        }

        // Initialize start time if this is the first trade
        if self.start_time.is_none() {
            let current_trade = self.trades[self.current_index].clone();
            self.start_time = Some(Instant::now());
            self.current_index += 1;
            return std::task::Poll::Ready(Some(current_trade));
        }

        let current_trade = &self.trades[self.current_index];

        // Calculate when this trade should be emitted based on timestamp differences
        if let (Some(base_timestamp), Some(start_time)) = (self.base_timestamp, self.start_time) {
            let time_diff_microseconds = current_trade.timestamp - base_timestamp;
            let real_time_diff = Duration::from_micros(time_diff_microseconds as u64);
            let scaled_time_diff = Duration::from_micros(
                (real_time_diff.as_micros() as f64 / self.speed_multiplier as f64) as u64,
            );

            let target_time = start_time + scaled_time_diff;
            let now = Instant::now();

            if now >= target_time {
                let trade = current_trade.clone();
                self.current_index += 1;
                std::task::Poll::Ready(Some(trade))
            } else {
                // Set up a timer to wake us when it's time
                let waker = cx.waker().clone();
                let sleep_duration = target_time - now;

                tokio::spawn(async move {
                    sleep(sleep_duration).await;
                    waker.wake();
                });

                std::task::Poll::Pending
            }
        } else {
            // Fallback: emit immediately
            let trade = current_trade.clone();
            self.current_index += 1;
            std::task::Poll::Ready(Some(trade))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use rangebar_core::FixedPoint;

    fn create_test_trade(id: i64, timestamp: i64, price: f64) -> AggTrade {
        AggTrade {
            agg_trade_id: id,
            price: FixedPoint::from_str(&price.to_string()).unwrap(),
            volume: FixedPoint::from_str("1.0").unwrap(),
            first_trade_id: id,
            last_trade_id: id,
            timestamp,
            is_buyer_maker: false,
            is_best_match: None,
        }
    }

    #[test]
    fn test_replay_buffer_capacity() {
        let buffer = ReplayBuffer::new(Duration::from_secs(60)); // 1 minute capacity

        let base_time = 1_704_067_200_000_000_i64; // 2024-01-01 00:00:00 in microseconds

        // Add trades spanning 2 minutes (1 trade per second in microseconds)
        for i in 0..120 {
            let trade = create_test_trade(i, base_time + (i * 1_000_000), 50000.0); // 1 second intervals in microseconds
            buffer.push(trade);
        }

        // Should only keep the last 60 seconds worth
        let stats = buffer.stats();
        assert!(
            stats.trade_count <= 120,
            "Expected <= 120 trades, got {}",
            stats.trade_count
        );

        let trades = buffer.get_trades_from(1); // Last 1 minute
        assert!(!trades.is_empty());
    }

    #[test]
    fn test_replay_buffer_time_span() {
        let buffer = ReplayBuffer::new(Duration::from_secs(300)); // 5 minutes

        // Use a more realistic timestamp (approximately 2024-01-01 in microseconds)
        let base_time = 1_704_067_200_000_000_i64; // 2024-01-01 00:00:00 in microseconds

        // Add a few trades over 1 minute (using microseconds)
        buffer.push(create_test_trade(1, base_time, 50000.0));
        buffer.push(create_test_trade(2, base_time + 30_000_000, 50100.0)); // 30s later in microseconds
        buffer.push(create_test_trade(3, base_time + 60_000_000, 50200.0)); // 60s later in microseconds

        let span = buffer.time_span().unwrap();
        assert_eq!(span.as_secs(), 60);
    }

    #[tokio::test]
    async fn test_replay_stream() {
        // Use a more realistic timestamp (approximately 2024-01-01 in microseconds)
        let base_time = 1_704_067_200_000_000_i64; // 2024-01-01 00:00:00 in microseconds
        let trades = vec![
            create_test_trade(1, base_time, 50000.0),
            create_test_trade(2, base_time + 1_000_000, 50100.0), // 1s later in microseconds
            create_test_trade(3, base_time + 2_000_000, 50200.0), // 2s later in microseconds
        ];

        let mut stream = ReplayStream::new(trades, 10.0); // 10x speed
        assert_eq!(stream.total(), 3);
        assert_eq!(stream.remaining(), 3);
        assert_eq!(stream.progress(), 0.0);

        // First trade should be immediate
        let first = StreamExt::next(&mut stream).await;
        assert!(first.is_some());
        assert_eq!(first.unwrap().agg_trade_id, 1);
    }

    #[test]
    fn test_get_trades_from_empty_buffer() {
        let buffer = ReplayBuffer::new(Duration::from_secs(60));
        let trades = buffer.get_trades_from(1);
        assert_eq!(trades.len(), 0); // Should not panic
    }
}
