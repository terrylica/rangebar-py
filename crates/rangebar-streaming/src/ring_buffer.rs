//! Fixed-size ring buffer for streaming range bars (Issue #96 Task #9)
//!
//! Replaces unbounded Vec with circular buffer to prevent OOM in long-running sidecars.
//! When full, drops old bars gracefully instead of blocking producers.
//!
//! Design:
//! - Fixed capacity (default 10K bars = 5MB)
//! - O(1) push/pop operations
//! - Thread-safe via parking_lot::Mutex (Issue #89)
//! - Metrics for backpressure, dropped bars, max depth

use std::sync::Arc;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// A single slot in the ring buffer
#[derive(Clone, Debug)]
pub struct RingBufferSlot<T: Clone> {
    value: Option<T>,
}

/// Fixed-size ring buffer with metrics
pub struct RingBuffer<T: Clone> {
    buffer: Vec<RingBufferSlot<T>>,
    write_idx: usize,  // Where next item will be written
    read_idx: usize,   // Where next item will be read
    count: usize,      // Number of items currently in buffer
    /// Metrics accessible from outside
    metrics: Arc<RingBufferMetrics>,
}

/// Metrics for ring buffer operations
#[derive(Debug, Default)]
pub struct RingBufferMetrics {
    /// Number of items pushed
    pub total_pushed: AtomicU64,
    /// Number of items dropped (due to full buffer)
    pub total_dropped: AtomicU64,
    /// Number of items popped
    pub total_popped: AtomicU64,
    /// Maximum depth observed
    pub max_depth: AtomicU64,
}

impl<T: Clone> RingBuffer<T> {
    /// Create a new ring buffer with given capacity
    pub fn new(capacity: usize) -> Self {
        let buffer = vec![
            RingBufferSlot { value: None };
            capacity.max(1)
        ];
        Self {
            buffer,
            write_idx: 0,
            read_idx: 0,
            count: 0,
            metrics: Arc::new(RingBufferMetrics::default()),
        }
    }

    /// Push an item into the buffer
    /// Returns true if item was added, false if buffer was full and old item was dropped
    pub fn push(&mut self, item: T) -> bool {
        let was_full = self.is_full();

        // If full, drop old item
        if was_full {
            self.buffer[self.read_idx].value = None;
            self.read_idx = (self.read_idx + 1) % self.buffer.len();
            self.count = self.count.saturating_sub(1);
            self.metrics.total_dropped.fetch_add(1, Ordering::Relaxed);
        }

        // Push new item
        self.buffer[self.write_idx].value = Some(item);
        self.write_idx = (self.write_idx + 1) % self.buffer.len();
        self.count += 1;

        // Update metrics
        self.metrics.total_pushed.fetch_add(1, Ordering::Relaxed);
        let current_depth = self.count as u64;
        let _ = self.metrics.max_depth.fetch_max(current_depth, Ordering::Relaxed);

        !was_full
    }

    /// Pop an item from the buffer
    pub fn pop(&mut self) -> Option<T> {
        if self.count == 0 {
            return None;
        }

        let slot = &mut self.buffer[self.read_idx];
        let item = slot.value.take();
        self.read_idx = (self.read_idx + 1) % self.buffer.len();
        self.count = self.count.saturating_sub(1);

        if item.is_some() {
            self.metrics.total_popped.fetch_add(1, Ordering::Relaxed);
        }

        item
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Check if buffer is full
    pub fn is_full(&self) -> bool {
        self.count >= self.buffer.len()
    }

    /// Get current count
    pub fn len(&self) -> usize {
        self.count
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// Get metrics reference
    pub fn metrics(&self) -> Arc<RingBufferMetrics> {
        Arc::clone(&self.metrics)
    }

    /// Clear all items
    pub fn clear(&mut self) {
        for slot in &mut self.buffer {
            slot.value = None;
        }
        self.write_idx = 0;
        self.read_idx = 0;
        self.count = 0;
    }
}

/// Thread-safe wrapper for RingBuffer
pub struct ConcurrentRingBuffer<T: Clone + Send + 'static> {
    inner: Arc<Mutex<RingBuffer<T>>>,
}

impl<T: Clone + Send + 'static> ConcurrentRingBuffer<T> {
    /// Create a new concurrent ring buffer
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(RingBuffer::new(capacity))),
        }
    }

    /// Push an item (returns true if not dropped, false if dropped due to full)
    pub fn push(&self, item: T) -> bool {
        let mut buf = self.inner.lock();
        buf.push(item)
    }

    /// Pop an item
    pub fn pop(&self) -> Option<T> {
        let mut buf = self.inner.lock();
        buf.pop()
    }

    /// Get current length
    pub fn len(&self) -> usize {
        self.inner.lock().len()
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.inner.lock().capacity()
    }

    /// Get metrics
    pub fn metrics(&self) -> Arc<RingBufferMetrics> {
        self.inner.lock().metrics()
    }

    /// Cloneable reference
    pub fn clone_ref(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_basic() {
        let mut buf = RingBuffer::new(3);
        assert!(buf.is_empty());
        assert!(!buf.is_full());

        assert!(buf.push(1));
        assert!(!buf.is_empty());
        assert!(!buf.is_full());

        assert!(buf.push(2));
        assert!(buf.push(3));
        assert!(buf.is_full());

        // Next push drops oldest
        assert!(!buf.push(4));
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.metrics().total_dropped.load(Ordering::Relaxed), 1);

        // Pop returns items in order
        assert_eq!(buf.pop(), Some(2));
        assert_eq!(buf.pop(), Some(3));
        assert_eq!(buf.pop(), Some(4));
        assert_eq!(buf.pop(), None);
    }

    #[test]
    fn test_ring_buffer_metrics() {
        let mut buf = RingBuffer::new(2);

        buf.push(1);
        buf.push(2);
        assert_eq!(buf.metrics().total_pushed.load(Ordering::Relaxed), 2);
        assert_eq!(buf.metrics().max_depth.load(Ordering::Relaxed), 2);

        buf.push(3);  // Drops 1
        assert_eq!(buf.metrics().total_dropped.load(Ordering::Relaxed), 1);

        buf.pop();  // Returns item 2
        buf.pop();  // Returns item 3
        buf.pop();  // Returns None (buffer empty)
        // Only 2 items were actually popped (3rd pop was empty)
        assert_eq!(buf.metrics().total_popped.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_concurrent_ring_buffer() {
        let buf = ConcurrentRingBuffer::new(2);
        assert!(buf.push(1));
        assert!(buf.push(2));
        assert!(!buf.push(3));  // Drops

        assert_eq!(buf.pop(), Some(2));
        assert_eq!(buf.len(), 1);
    }
}
