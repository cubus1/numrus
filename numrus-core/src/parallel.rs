//! Thread-parallel execution utilities for SIMD workloads.
//!
//! Uses `std::thread::scope` for zero-overhead scoped parallelism —
//! no runtime, no allocations on the hot path.

/// Execute a closure in parallel over chunks of `[start, end)`.
///
/// Automatically splits work across all available CPU cores.
/// Uses scoped threads — no heap allocation for the thread pool.
///
/// # Arguments
/// * `start` - Start of range (inclusive).
/// * `end` - End of range (exclusive).
/// * `f` - Closure receiving `(chunk_start, chunk_end)`.
#[inline]
pub fn parallel_for_chunks<F>(start: usize, end: usize, f: F)
where
    F: Fn(usize, usize) + Sync + Send + Copy,
{
    if start >= end {
        return;
    }
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let total = end - start;
    let chunk_size = total.div_ceil(num_threads);

    if total <= chunk_size || num_threads <= 1 {
        // Small workload — run inline, no thread overhead
        f(start, end);
        return;
    }

    std::thread::scope(|s| {
        for chunk_start in (start..end).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(end);
            s.spawn(move || {
                f(chunk_start, chunk_end);
            });
        }
    });
}

/// Execute a closure in parallel, collecting results from each chunk.
///
/// Returns a Vec of results, one per chunk (thread).
#[inline]
pub fn parallel_map_chunks<F, R>(start: usize, end: usize, f: F) -> Vec<R>
where
    F: Fn(usize, usize) -> R + Sync + Send + Copy,
    R: Send,
{
    if start >= end {
        return Vec::new();
    }
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let total = end - start;
    let chunk_size = total.div_ceil(num_threads);

    if total <= chunk_size || num_threads <= 1 {
        return vec![f(start, end)];
    }

    std::thread::scope(|s| {
        let handles: Vec<_> = (start..end)
            .step_by(chunk_size)
            .map(|chunk_start| {
                let chunk_end = (chunk_start + chunk_size).min(end);
                s.spawn(move || f(chunk_start, chunk_end))
            })
            .collect();

        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_parallel_for_chunks_covers_range() {
        let counter = AtomicUsize::new(0);
        parallel_for_chunks(0, 1000, |start, end| {
            counter.fetch_add(end - start, Ordering::Relaxed);
        });
        assert_eq!(counter.load(Ordering::Relaxed), 1000);
    }

    #[test]
    fn test_parallel_map_chunks() {
        let results = parallel_map_chunks(0, 100, |start, end| (start..end).sum::<usize>());
        let total: usize = results.iter().sum();
        assert_eq!(total, (0..100).sum::<usize>());
    }

    #[test]
    fn test_empty_range() {
        let counter = AtomicUsize::new(0);
        parallel_for_chunks(5, 5, |_, _| {
            counter.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }
}
