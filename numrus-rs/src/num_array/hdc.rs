//! Hyperdimensional Computing operations for NumArrayU8.
//!
//! Provides the core HDC primitives optimized for AVX-512:
//! - **BIND** (XOR) — already available via `^` operator and BitwiseSimdOps
//! - **BUNDLE** — majority vote across multiple bitpacked vectors
//! - **PERMUTE** — circular bit-lane rotation
//! - **DISTANCE** — hamming distance (already in bitwise.rs)
//! - **DOT_I8** — int8 dot product (VNNI-targetable) for embedding containers
//!
//! ## BUNDLE optimization
//!
//! Hybrid strategy:
//! - **Small n** (≤ 16 vectors): compiler-auto-vectorized per-byte counting.
//!   The compiler recognizes the sequential pattern and emits AVX-512 bytewise ops.
//! - **Large n** (> 16 vectors): ripple-carry bit-parallel counter with explicit
//!   `u64x8` SIMD. Processes 512 bit positions per instruction.
//!
//! Parallelization uses the **blackboard borrow-mut scheme**: the output
//! buffer is split into disjoint mutable regions via `split_at_mut`, giving
//! each thread exclusive ownership of its slice. No `Arc`, no `Mutex`,
//! no lock contention.
//!
//! ## CogRecord container support
//!
//! Designed for 4 × 16384-bit (2048-byte) containers = 8KB CogRecord:
//! - Container 0: META — codebook identity + DN + hashtag zone
//! - Container 1: CAM — content-addressable memory (Hamming via VPOPCNTDQ)
//! - Container 2: B-tree — structural position index
//! - Container 3: Embedding — int8/int4/binary (dot product via VNNI, Hamming via VPOPCNTDQ)
//!
//! All container sizes (2048, 8192, 16384, 65536 bytes) are multiples of 64,
//! so every SIMD path uses full u64x8 vectors with zero scalar tail.

use super::NumArrayU8;
use std::simd::u64x8;

/// Crossover point: use ripple-carry for n > this, naive per-byte for n ≤ this.
/// Below this threshold, compiler auto-vectorization of the byte-level loop
/// outperforms the ripple-carry due to lower overhead (no counter allocation).
const BUNDLE_RIPPLE_THRESHOLD: usize = 16;

/// Minimum total u64 operations before spawning threads for bundle.
/// Thread spawn overhead is ~5-30µs; we want enough work per thread to amortize.
/// total_work = n_vectors × counter_bits × u64_lanes.
const BUNDLE_PARALLEL_WORK: usize = 500_000;

impl NumArrayU8 {
    /// BIND: XOR two hypervectors. Alias for the `^` operator.
    ///
    /// This is the X-crossing: `bind(a, b) = a XOR b`.
    /// Involutory: `bind(bind(a, b), b) == a`.
    #[inline]
    pub fn bind(&self, other: &Self) -> Self {
        self ^ other
    }

    /// PERMUTE: Circular bit-rotation of the entire bitpacked vector by `k` positions.
    ///
    /// Rotates the bit-representation left by `k` bit positions (wrapping).
    /// This places each role in a different bit-plane so that triple-bind
    /// `src ^ permute(rel, 1) ^ permute(tgt, 2)` is unambiguous.
    ///
    /// # Example
    /// ```
    /// use numrus_rs::NumArrayU8;
    /// let v = NumArrayU8::new(vec![0x80, 0x00]); // bit 7 of byte 0 set
    /// let p = v.permute(1); // rotate left by 1
    /// // bit 7 moves to bit 8 (byte 1, bit 0)
    /// assert_eq!(p.get_data(), &[0x00, 0x01]);
    /// ```
    pub fn permute(&self, k: usize) -> Self {
        let data = &self.data;
        let total_bits = data.len() * 8;
        if total_bits == 0 {
            return self.clone();
        }
        let k = k % total_bits;
        if k == 0 {
            return self.clone();
        }

        let len = data.len();
        let mut out = vec![0u8; len];

        let byte_shift = k / 8;
        let bit_shift = k % 8;

        if bit_shift == 0 {
            for i in 0..len {
                let src_idx = (i + len - byte_shift) % len;
                out[i] = data[src_idx];
            }
        } else {
            for i in 0..len {
                let src_hi = (i + len - byte_shift) % len;
                let src_lo = (i + len - byte_shift + len - 1) % len;
                out[i] = (data[src_hi] << bit_shift) | (data[src_lo] >> (8 - bit_shift));
            }
        }

        NumArrayU8::new_with_shape(out, self.shape.clone())
    }

    /// BUNDLE: Majority vote across multiple bitpacked hypervectors.
    ///
    /// For each bit position, if more than half the input vectors have that bit set,
    /// the output bit is 1; otherwise 0. Ties (even count) are broken toward 0.
    ///
    /// Uses a hybrid strategy:
    /// - **n ≤ 16**: per-byte counting, compiler auto-vectorizes to AVX-512
    /// - **n > 16**: ripple-carry bit-parallel counter with explicit u64x8 SIMD
    /// - **large workloads**: blackboard parallelization (split_at_mut, no Mutex)
    ///
    /// # Panics
    /// Panics if the input slice is empty or vectors have different lengths.
    ///
    /// # Example
    /// ```
    /// use numrus_rs::NumArrayU8;
    /// let a = NumArrayU8::new(vec![0xFF; 8]);
    /// let b = NumArrayU8::new(vec![0xFF; 8]);
    /// let c = NumArrayU8::new(vec![0x00; 8]);
    /// let result = NumArrayU8::bundle(&[&a, &b, &c]);
    /// assert_eq!(result.get_data(), &[0xFF; 8]); // 2 out of 3 = majority
    /// ```
    pub fn bundle(vectors: &[&NumArrayU8]) -> NumArrayU8 {
        assert!(!vectors.is_empty(), "Bundle requires at least one vector");
        let len = vectors[0].data.len();
        for v in vectors.iter() {
            assert_eq!(v.data.len(), len, "All vectors must have the same length");
        }

        let n = vectors.len();
        let threshold = n / 2;

        if n <= BUNDLE_RIPPLE_THRESHOLD {
            // ── Fast path: per-byte counting ──
            // The compiler auto-vectorizes this to AVX-512 byte-level ops.
            // For small n, this beats the ripple-carry due to zero overhead.
            return bundle_naive(vectors, len, threshold);
        }

        // ── Ripple-carry path for large n ──
        let u64_lanes = len / 8;
        let has_tail = !len.is_multiple_of(8);
        let counter_bits = (usize::BITS - n.leading_zeros()) as usize;

        let mut out = vec![0u8; len];

        if u64_lanes > 0 {
            let total_work = n * counter_bits * u64_lanes;
            let n_threads = std::thread::available_parallelism()
                .map(|t| t.get())
                .unwrap_or(1);

            if total_work > BUNDLE_PARALLEL_WORK && n_threads > 1 && u64_lanes >= n_threads * 8 {
                // ── Blackboard borrow-mut scheme ──
                // Split output into disjoint mutable regions. Each thread
                // writes exclusively to its own slice — no locks.
                let lanes_per_thread = u64_lanes.div_ceil(n_threads);
                let lanes_per_thread = (lanes_per_thread + 7) & !7; // align to u64x8

                std::thread::scope(|s| {
                    let mut remaining = &mut out[..u64_lanes * 8];
                    let mut lane_offset = 0usize;

                    while lane_offset < u64_lanes && !remaining.is_empty() {
                        let lanes_this = lanes_per_thread.min(u64_lanes - lane_offset);
                        let byte_count = lanes_this * 8;
                        let (chunk, rest) = remaining.split_at_mut(byte_count);
                        remaining = rest;

                        let byte_off = lane_offset * 8;
                        s.spawn(move || {
                            bundle_ripple_into(
                                vectors,
                                byte_off,
                                lanes_this,
                                threshold,
                                counter_bits,
                                chunk,
                            );
                        });

                        lane_offset += lanes_per_thread;
                    }
                });
            } else {
                bundle_ripple_into(
                    vectors,
                    0,
                    u64_lanes,
                    threshold,
                    counter_bits,
                    &mut out[..u64_lanes * 8],
                );
            }

            if has_tail {
                bundle_tail_bytes(vectors, u64_lanes * 8, len, threshold, &mut out);
            }
        } else {
            bundle_tail_bytes(vectors, 0, len, threshold, &mut out);
        }

        NumArrayU8::new_with_shape(out, vectors[0].shape.clone())
    }

    /// Compute dot product interpreting bytes as signed int8 values.
    ///
    /// With `-C target-cpu=native`, the compiler emits AVX-512 VNNI instructions
    /// (VPDPBUSD) for hardware-accelerated int8 multiply-accumulate.
    ///
    /// For CogRecord Container 3 embeddings:
    /// - 1024D int8: 1024 bytes, dot product in ~2 VNNI passes (512 bits each)
    /// - 2048D int8: 2048 bytes = full 16384-bit container
    ///
    /// Cosine similarity: `dot_i8(a,b) / (norm_i8(a) * norm_i8(b))`
    ///
    /// # Example
    /// ```
    /// use numrus_rs::NumArrayU8;
    /// // Two vectors: [1, 2, 3, 127] as unsigned bytes (interpreted as i8)
    /// let a = NumArrayU8::new(vec![1, 2, 3, 127]);
    /// let b = NumArrayU8::new(vec![1, 2, 3, 127]);
    /// // dot = 1*1 + 2*2 + 3*3 + 127*127 = 1 + 4 + 9 + 16129 = 16143
    /// assert_eq!(a.dot_i8(&b), 16143);
    /// ```
    pub fn dot_i8(&self, other: &Self) -> i64 {
        assert_eq!(
            self.data.len(),
            other.data.len(),
            "Vectors must have the same length"
        );

        let a = &self.data;
        let b = &other.data;
        let len = a.len();

        // Process in chunks of 32 to enable VNNI auto-vectorization.
        // VPDPBUSD processes 64 bytes per instruction (16 groups of 4 bytes).
        // Using i32 accumulators to avoid overflow, then widen to i64.
        let chunks = len / 32;
        let mut total: i64 = 0;

        for c in 0..chunks {
            let base = c * 32;
            let mut acc: i32 = 0;
            for i in 0..32 {
                acc += (a[base + i] as i8 as i32) * (b[base + i] as i8 as i32);
            }
            total += acc as i64;
        }

        // Scalar tail
        for i in (chunks * 32)..len {
            total += (a[i] as i8 as i64) * (b[i] as i8 as i64);
        }

        total
    }

    /// Compute squared L2 norm interpreting bytes as signed int8 values.
    /// Returns ‖v‖² = Σ(v[i]²) as i64.
    ///
    /// Use with `dot_i8` for cosine similarity:
    /// `cos(a,b) = dot_i8(a,b) as f64 / ((norm_sq_i8(a) as f64).sqrt() * (norm_sq_i8(b) as f64).sqrt())`
    pub fn norm_sq_i8(&self) -> i64 {
        let a = &self.data;
        let len = a.len();
        let chunks = len / 32;
        let mut total: i64 = 0;

        for c in 0..chunks {
            let base = c * 32;
            let mut acc: i32 = 0;
            for i in 0..32 {
                let v = a[base + i] as i8 as i32;
                acc += v * v;
            }
            total += acc as i64;
        }

        for i in (chunks * 32)..len {
            let v = a[i] as i8 as i64;
            total += v * v;
        }

        total
    }

    /// Cosine similarity interpreting bytes as signed int8 values.
    /// Returns a value in [-1.0, 1.0].
    ///
    /// Uses VNNI-accelerated dot product and norm computation.
    pub fn cosine_i8(&self, other: &Self) -> f64 {
        let dot = self.dot_i8(other) as f64;
        let norm_a = (self.norm_sq_i8() as f64).sqrt();
        let norm_b = (other.norm_sq_i8() as f64).sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }

    /// Adaptive Hamming distance with multi-stage early-exit cascade.
    ///
    /// Returns `Some(exact_distance)` if the distance is within `threshold`,
    /// `None` if the candidate is rejected early (saving ~97-99.7% of compute).
    ///
    /// ## Cascade stages
    ///
    /// | Stage | Sample | Reject condition | Eliminates |
    /// |-------|--------|------------------|------------|
    /// | 1     | 1/16   | estimate > threshold + 3σ | ~99.7% |
    /// | 2     | 1/4    | estimate > threshold + 2σ | ~95%   |
    /// | 3     | full   | exact > threshold | exact |
    ///
    /// ## Statistical basis
    ///
    /// For k sampled bytes out of N total, the scaled estimate d̂ = popcount(sample) × (N/k).
    /// Under independent bits with p = d̂/(N×8):
    ///   σ(d̂) = N × √(8p(1-p)/k)
    ///
    /// ## Example
    /// ```
    /// use numrus_rs::NumArrayU8;
    /// let a = NumArrayU8::new(vec![0xFF; 2048]);
    /// let b = NumArrayU8::new(vec![0x00; 2048]);
    /// // Distance is 2048*8 = 16384, threshold 1000 → rejected early
    /// assert!(a.hamming_distance_adaptive(&b, 1000).is_none());
    /// // Same vector → distance 0
    /// assert_eq!(a.hamming_distance_adaptive(&a, 1000), Some(0));
    /// ```
    pub fn hamming_distance_adaptive(&self, other: &Self, threshold: u64) -> Option<u64> {
        assert_eq!(
            self.data.len(),
            other.data.len(),
            "Arrays must have the same length"
        );

        let a = &self.data;
        let b = &other.data;
        let n = a.len();

        if n == 0 {
            return Some(0);
        }

        // For small vectors (< 128 bytes), just compute directly
        if n < 128 {
            let d = self.hamming_distance(other);
            return if d <= threshold { Some(d) } else { None };
        }

        let total_bits = (n * 8) as f64;

        // ── Stage 1: 1/16 sample ──
        let s1 = n / 16;
        let d1 = hamming_chunk_inline(&a[..s1], &b[..s1]);
        let estimate1 = d1 * 16;
        let p1 = (estimate1 as f64) / total_bits;
        let p1 = p1.clamp(0.001, 0.999); // avoid degenerate sigma
        let sigma1 = (n as f64) * (8.0 * p1 * (1.0 - p1) / s1 as f64).sqrt();

        if estimate1 as f64 > threshold as f64 + 3.0 * sigma1 {
            return None; // 3σ rejection
        }

        // ── Stage 2: 1/4 sample (incremental) ──
        let s2 = n / 4;
        let d2 = d1 + hamming_chunk_inline(&a[s1..s2], &b[s1..s2]);
        let estimate2 = d2 * 4;
        let p2 = (estimate2 as f64) / total_bits;
        let p2 = p2.clamp(0.001, 0.999);
        let sigma2 = (n as f64) * (8.0 * p2 * (1.0 - p2) / s2 as f64).sqrt();

        if estimate2 as f64 > threshold as f64 + 2.0 * sigma2 {
            return None; // 2σ rejection
        }

        // ── Stage 3: full precision (incremental) ──
        let d3 = d2 + hamming_chunk_inline(&a[s2..], &b[s2..]);

        if d3 <= threshold {
            Some(d3)
        } else {
            None
        }
    }

    /// Adaptive Hamming search with 3-stroke HDR cascade.
    ///
    /// Returns `(index, exact_hamming_distance)` for all candidates within threshold.
    /// Uses statistical warmup (128 samples) to set rejection thresholds.
    /// CPUID is checked once (not per-candidate).
    ///
    /// For ranked results with cosine precision, use `hdr_search` instead.
    ///
    /// ## Performance model
    ///
    /// | Stroke | Candidates | Operation | Eliminates |
    /// |--------|-----------|-----------|------------|
    /// | 1      | ALL       | Partial VPOPCNTDQ + 3σ reject | ~98% |
    /// | 2      | survivors | Full VPOPCNTDQ (incremental) | ~90% of survivors |
    ///
    /// ## Example
    /// ```
    /// use numrus_rs::NumArrayU8;
    /// let query = NumArrayU8::new(vec![0xAA; 2048]);
    /// // Database: 4 vectors of 2048 bytes each
    /// let mut db_data = vec![0xAA; 2048]; // vec 0: identical to query
    /// db_data.extend(vec![0x55; 2048]);   // vec 1: maximally different
    /// db_data.extend(vec![0xAA; 2048]);   // vec 2: identical to query
    /// db_data.extend(vec![0x00; 2048]);   // vec 3: different
    /// let db = NumArrayU8::new(db_data);
    /// let results = query.hamming_search_adaptive(&db, 2048, 4, 100);
    /// // Only vectors 0 and 2 should match (distance = 0)
    /// assert_eq!(results.len(), 2);
    /// assert_eq!(results[0], (0, 0));
    /// assert_eq!(results[1], (2, 0));
    /// ```
    pub fn hamming_search_adaptive(
        &self,
        database: &NumArrayU8,
        vec_len: usize,
        count: usize,
        threshold: u64,
    ) -> Vec<(usize, u64)> {
        let results = numrus_core::simd::hdr_cascade_search(
            &self.data,
            &database.data,
            vec_len,
            count,
            threshold,
            numrus_core::simd::PreciseMode::Off,
        );
        results.iter().map(|r| (r.index, r.hamming)).collect()
    }

    /// HDR (High Dynamic Range) search: 3-stroke cascade with cosine precision tier.
    ///
    /// Same as `hamming_search_adaptive` but Stroke 3 computes cosine similarity
    /// via VNNI dot_i8 for the ~0.2% finalists.
    ///
    /// Returns `(index, hamming_distance, cosine_similarity)` sorted by cosine (best first).
    ///
    /// ## Cascade
    ///
    /// | Stroke | Sample | Operation | Eliminates |
    /// |--------|--------|-----------|------------|
    /// | 1 | 1/16 | Partial VPOPCNTDQ + 3σ warmup reject | ~98% |
    /// | 2 | Full | Full VPOPCNTDQ (incremental) | ~90% of survivors |
    /// | 3 | Full | VNNI dot_i8 → cosine similarity | precision ranking |
    pub fn hdr_search(
        &self,
        database: &NumArrayU8,
        vec_len: usize,
        count: usize,
        threshold: u64,
    ) -> Vec<(usize, u64, f64)> {
        let results = numrus_core::simd::hdr_cascade_search(
            &self.data,
            &database.data,
            vec_len,
            count,
            threshold,
            numrus_core::simd::PreciseMode::Vnni,
        );
        results
            .iter()
            .map(|r| (r.index, r.hamming, r.precise))
            .collect()
    }

    /// HDR search with f32 dequantization precision tier.
    ///
    /// For quantized embeddings (e.g. Jina f32 → u8). Dequantizes the ~0.2% finalists
    /// back to f32 using scale/zero_point, then computes SIMD dot_f32 cosine.
    /// Returns `(index, hamming_distance, cosine_similarity)` sorted by cosine (best first).
    pub fn hdr_search_f32(
        &self,
        database: &NumArrayU8,
        vec_len: usize,
        count: usize,
        threshold: u64,
        scale: f32,
        zero_point: i32,
    ) -> Vec<(usize, u64, f64)> {
        let results = numrus_core::simd::hdr_cascade_search(
            &self.data,
            &database.data,
            vec_len,
            count,
            threshold,
            numrus_core::simd::PreciseMode::F32 { scale, zero_point },
        );
        results
            .iter()
            .map(|r| (r.index, r.hamming, r.precise))
            .collect()
    }

    /// HDR search with XOR Delta + INT8 residual precision tier (Case 5).
    ///
    /// For 3D bitpacked vectors with organic INT8 delta XOR.
    /// Tier 1-2: Hamming on XOR delta bits.
    /// Tier 3: Blended distance — hamming_norm * (1-w) + INT8 cosine * w.
    /// `delta_weight` controls blend (0.0 = pure Hamming, 1.0 = pure INT8, typical 0.3).
    pub fn hdr_search_delta(
        &self,
        database: &NumArrayU8,
        vec_len: usize,
        count: usize,
        threshold: u64,
        delta_weight: f32,
    ) -> Vec<(usize, u64, f64)> {
        let results = numrus_core::simd::hdr_cascade_search(
            &self.data,
            &database.data,
            vec_len,
            count,
            threshold,
            numrus_core::simd::PreciseMode::DeltaXor { delta_weight },
        );
        results
            .iter()
            .map(|r| (r.index, r.hamming, r.precise))
            .collect()
    }

    /// HDR search with BF16-structured Hamming precision tier.
    ///
    /// For native BF16 embeddings (2 bytes per dimension). Stroke 3 computes
    /// weighted Hamming distance respecting IEEE 754 BF16 field structure:
    /// sign (1 bit), exponent (8 bits), mantissa (7 bits) with configurable weights.
    /// Returns `(index, hamming_distance, similarity)` sorted by similarity (best first).
    pub fn hdr_search_bf16(
        &self,
        database: &NumArrayU8,
        vec_len: usize,
        count: usize,
        threshold: u64,
        weights: numrus_core::bf16_hamming::BF16Weights,
    ) -> Vec<(usize, u64, f64)> {
        let results = numrus_core::simd::hdr_cascade_search(
            &self.data,
            &database.data,
            vec_len,
            count,
            threshold,
            numrus_core::simd::PreciseMode::BF16Hamming { weights },
        );
        results
            .iter()
            .map(|r| (r.index, r.hamming, r.precise))
            .collect()
    }

    /// Adaptive cosine similarity search for int8 embeddings.
    ///
    /// Same cascade principle as `hamming_search_adaptive`, but for int8 dot product.
    /// Uses progressive dot product computation with early exit based on estimated
    /// cosine similarity bounds.
    ///
    /// Returns `(index, cosine_similarity)` pairs for candidates above `min_similarity`.
    ///
    /// ## Cascade stages
    ///
    /// | Stage | Sample | Reject condition |
    /// |-------|--------|------------------|
    /// | 1     | 1/16   | upper_bound(cos) < min_similarity - margin |
    /// | 2     | 1/4    | upper_bound(cos) < min_similarity |
    /// | 3     | full   | exact cos ≥ min_similarity |
    ///
    /// This enables **FP64 cosine at ~3-6% of full cost** for typical workloads
    /// (99.7% of candidates rejected at stage 1).
    ///
    /// ## Example
    /// ```
    /// use numrus_rs::NumArrayU8;
    /// let query = NumArrayU8::new(vec![100u8; 1024]); // 1024D int8 embedding
    /// let mut db_data = vec![100u8; 1024]; // vec 0: identical
    /// db_data.extend(vec![0u8; 1024]);     // vec 1: orthogonal
    /// db_data.extend(vec![100u8; 1024]);   // vec 2: identical
    /// let db = NumArrayU8::new(db_data);
    /// let results = query.cosine_search_adaptive(&db, 1024, 3, 0.9);
    /// assert_eq!(results.len(), 2); // Only vecs 0 and 2 match
    /// ```
    pub fn cosine_search_adaptive(
        &self,
        database: &NumArrayU8,
        vec_len: usize,
        count: usize,
        min_similarity: f64,
    ) -> Vec<(usize, f64)> {
        assert_eq!(
            database.data.len(),
            vec_len * count,
            "Database length must be vec_len * count"
        );
        assert_eq!(self.data.len(), vec_len, "Query must have length vec_len");

        let query = &self.data;
        let db = &database.data;

        // Hoist CPUID dispatch — ONE check, used for all candidates.
        let dot_fn = numrus_core::simd::select_dot_i8_fn();

        // Pre-compute query norm (amortized across all candidates)
        let query_norm_sq = dot_fn(query, query);
        let query_norm = (query_norm_sq as f64).sqrt();

        if query_norm == 0.0 {
            return Vec::new();
        }

        let mut results = Vec::new();

        // For small vectors, compute directly
        if vec_len < 128 {
            for i in 0..count {
                let candidate = &db[i * vec_len..(i + 1) * vec_len];
                let dot = dot_fn(query, candidate);
                let cand_norm = (dot_fn(candidate, candidate) as f64).sqrt();
                if cand_norm > 0.0 {
                    let cos = dot as f64 / (query_norm * cand_norm);
                    if cos >= min_similarity {
                        results.push((i, cos));
                    }
                }
            }
            return results;
        }

        let s1 = vec_len / 16;
        let s2 = vec_len / 4;
        let scale_1 = 16.0_f64;
        let scale_2 = 4.0_f64;

        for i in 0..count {
            let base = i * vec_len;
            let candidate = &db[base..base + vec_len];

            // ── Stage 1: 1/16 sample ──
            let dot_s1 = dot_fn(&query[..s1], &candidate[..s1]);
            let cand_norm_sq_s1 = dot_fn(&candidate[..s1], &candidate[..s1]);
            let cand_norm_s1 = (cand_norm_sq_s1 as f64 * scale_1).sqrt();

            if cand_norm_s1 > 0.0 {
                let cos_est = (dot_s1 as f64 * scale_1) / (query_norm * cand_norm_s1);
                if cos_est < min_similarity - 0.3 {
                    continue;
                }
            }

            // ── Stage 2: 1/4 sample (incremental) ──
            let dot_s2 = dot_s1 + dot_fn(&query[s1..s2], &candidate[s1..s2]);
            let cand_norm_sq_s2 = cand_norm_sq_s1 + dot_fn(&candidate[s1..s2], &candidate[s1..s2]);
            let cand_norm_s2 = (cand_norm_sq_s2 as f64 * scale_2).sqrt();

            if cand_norm_s2 > 0.0 {
                let cos_est = (dot_s2 as f64 * scale_2) / (query_norm * cand_norm_s2);
                if cos_est < min_similarity - 0.1 {
                    continue;
                }
            }

            // ── Stage 3: full precision ──
            let dot_full = dot_s2 + dot_fn(&query[s2..], &candidate[s2..]);
            let cand_norm_sq_full = cand_norm_sq_s2 + dot_fn(&candidate[s2..], &candidate[s2..]);
            let cand_norm = (cand_norm_sq_full as f64).sqrt();

            if cand_norm > 0.0 {
                let cos = dot_full as f64 / (query_norm * cand_norm);
                if cos >= min_similarity {
                    results.push((i, cos));
                }
            }
        }

        results
    }
}

// ── Adaptive search helpers ──

/// Hamming distance with 3-tier SIMD dispatch via numrus_core.
///
/// Dispatch chain: VPOPCNTDQ (AVX-512) → Harley-Seal (AVX2) → scalar POPCNT.
/// Replaces the previous scalar-only u64 POPCNT implementation.
#[inline(always)]
fn hamming_chunk_inline(a: &[u8], b: &[u8]) -> u64 {
    numrus_core::simd::hamming_distance(a, b)
}

// ── Bundle implementations ──

/// Per-byte majority vote. The compiler auto-vectorizes the inner loop to AVX-512.
/// Fast for small n (≤ 16) due to zero overhead from counter allocation.
#[inline]
fn bundle_naive(vectors: &[&NumArrayU8], len: usize, threshold: usize) -> NumArrayU8 {
    let mut out = vec![0u8; len];

    for byte_idx in 0..len {
        let mut count = [0u16; 8];
        for v in vectors.iter() {
            let byte = v.data[byte_idx];
            count[0] += ((byte) & 1) as u16;
            count[1] += ((byte >> 1) & 1) as u16;
            count[2] += ((byte >> 2) & 1) as u16;
            count[3] += ((byte >> 3) & 1) as u16;
            count[4] += ((byte >> 4) & 1) as u16;
            count[5] += ((byte >> 5) & 1) as u16;
            count[6] += ((byte >> 6) & 1) as u16;
            count[7] += ((byte >> 7) & 1) as u16;
        }
        let mut result_byte = 0u8;
        if count[0] as usize > threshold {
            result_byte |= 1;
        }
        if count[1] as usize > threshold {
            result_byte |= 2;
        }
        if count[2] as usize > threshold {
            result_byte |= 4;
        }
        if count[3] as usize > threshold {
            result_byte |= 8;
        }
        if count[4] as usize > threshold {
            result_byte |= 16;
        }
        if count[5] as usize > threshold {
            result_byte |= 32;
        }
        if count[6] as usize > threshold {
            result_byte |= 64;
        }
        if count[7] as usize > threshold {
            result_byte |= 128;
        }
        out[byte_idx] = result_byte;
    }

    NumArrayU8::new_with_shape(out, vectors[0].shape.clone())
}

/// Load 64 bytes into a u64x8 SIMD vector.
#[inline(always)]
fn load_u64x8(bytes: &[u8]) -> u64x8 {
    u64x8::from_array([
        u64::from_ne_bytes(bytes[0..8].try_into().unwrap()),
        u64::from_ne_bytes(bytes[8..16].try_into().unwrap()),
        u64::from_ne_bytes(bytes[16..24].try_into().unwrap()),
        u64::from_ne_bytes(bytes[24..32].try_into().unwrap()),
        u64::from_ne_bytes(bytes[32..40].try_into().unwrap()),
        u64::from_ne_bytes(bytes[40..48].try_into().unwrap()),
        u64::from_ne_bytes(bytes[48..56].try_into().unwrap()),
        u64::from_ne_bytes(bytes[56..64].try_into().unwrap()),
    ])
}

/// Store a u64x8 SIMD vector as 64 bytes.
#[inline(always)]
fn store_u64x8(val: u64x8, bytes: &mut [u8]) {
    let arr = val.to_array();
    bytes[0..8].copy_from_slice(&arr[0].to_ne_bytes());
    bytes[8..16].copy_from_slice(&arr[1].to_ne_bytes());
    bytes[16..24].copy_from_slice(&arr[2].to_ne_bytes());
    bytes[24..32].copy_from_slice(&arr[3].to_ne_bytes());
    bytes[32..40].copy_from_slice(&arr[4].to_ne_bytes());
    bytes[40..48].copy_from_slice(&arr[5].to_ne_bytes());
    bytes[48..56].copy_from_slice(&arr[6].to_ne_bytes());
    bytes[56..64].copy_from_slice(&arr[7].to_ne_bytes());
}

/// Ripple-carry bit-parallel bundle using explicit u64x8 SIMD (AVX-512).
/// Writes directly into `out` — blackboard pattern, no intermediate allocation.
#[inline]
fn bundle_ripple_into(
    vectors: &[&NumArrayU8],
    byte_offset: usize,
    num_lanes: usize,
    threshold: usize,
    counter_bits: usize,
    out: &mut [u8],
) {
    const SIMD_W: usize = 8;
    let simd_groups = num_lanes / SIMD_W;
    let scalar_tail = num_lanes % SIMD_W;

    if simd_groups > 0 {
        let mut digits = vec![u64x8::splat(0); counter_bits * simd_groups];
        let mut carry_buf = vec![u64x8::splat(0); simd_groups];

        for v in vectors.iter() {
            let v_data = &v.data;
            for sg in 0..simd_groups {
                let base = byte_offset + sg * SIMD_W * 8;
                carry_buf[sg] = load_u64x8(&v_data[base..]);
            }
            for d in 0..counter_bits {
                let doff = d * simd_groups;
                for sg in 0..simd_groups {
                    let new_carry = digits[doff + sg] & carry_buf[sg];
                    digits[doff + sg] ^= carry_buf[sg];
                    carry_buf[sg] = new_carry;
                }
            }
        }

        let check_val = (threshold + 1) as u64;
        let mut borrow = vec![u64x8::splat(0); simd_groups];

        for d in 0..counter_bits {
            let check_bit = if (check_val >> d) & 1 == 1 {
                u64x8::splat(u64::MAX)
            } else {
                u64x8::splat(0)
            };
            let doff = d * simd_groups;
            for sg in 0..simd_groups {
                let digit = digits[doff + sg];
                let new_borrow =
                    (!digit & check_bit) | (!digit & borrow[sg]) | (check_bit & borrow[sg]);
                borrow[sg] = new_borrow;
            }
        }

        for sg in 0..simd_groups {
            let result = !borrow[sg];
            store_u64x8(result, &mut out[sg * SIMD_W * 8..]);
        }
    }

    if scalar_tail > 0 {
        let tail_out_start = simd_groups * SIMD_W * 8;
        let tail_byte_offset = byte_offset + tail_out_start;

        let mut digits = vec![0u64; counter_bits * scalar_tail];
        let mut carry = vec![0u64; scalar_tail];

        for v in vectors.iter() {
            let v_data = &v.data;
            for lane in 0..scalar_tail {
                let base = tail_byte_offset + lane * 8;
                carry[lane] = u64::from_ne_bytes(v_data[base..base + 8].try_into().unwrap());
            }
            for d in 0..counter_bits {
                let doff = d * scalar_tail;
                for lane in 0..scalar_tail {
                    let new_carry = digits[doff + lane] & carry[lane];
                    digits[doff + lane] ^= carry[lane];
                    carry[lane] = new_carry;
                }
            }
        }

        let check_val = (threshold + 1) as u64;
        let mut borrow = vec![0u64; scalar_tail];
        for d in 0..counter_bits {
            let check_bit = if (check_val >> d) & 1 == 1 {
                u64::MAX
            } else {
                0
            };
            let doff = d * scalar_tail;
            for lane in 0..scalar_tail {
                let digit = digits[doff + lane];
                let new_borrow =
                    (!digit & check_bit) | (!digit & borrow[lane]) | (check_bit & borrow[lane]);
                borrow[lane] = new_borrow;
            }
        }

        for lane in 0..scalar_tail {
            let result_word = !borrow[lane];
            let base = tail_out_start + lane * 8;
            out[base..base + 8].copy_from_slice(&result_word.to_ne_bytes());
        }
    }
}

/// Handle tail bytes (< 8) that don't fit into u64 lanes.
#[inline]
fn bundle_tail_bytes(
    vectors: &[&NumArrayU8],
    start: usize,
    end: usize,
    threshold: usize,
    out: &mut [u8],
) {
    for byte_idx in start..end {
        let mut count = [0u32; 8];
        for v in vectors.iter() {
            let byte = v.data[byte_idx];
            for bit in 0..8 {
                count[bit] += ((byte >> bit) & 1) as u32;
            }
        }
        let mut result_byte = 0u8;
        for bit in 0..8 {
            if count[bit] as usize > threshold {
                result_byte |= 1 << bit;
            }
        }
        out[byte_idx] = result_byte;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- PERMUTE tests ----

    #[test]
    fn test_permute_zero() {
        let v = NumArrayU8::new(vec![0xAA; 8192]);
        let p = v.permute(0);
        assert_eq!(p.get_data(), v.get_data());
    }

    #[test]
    fn test_permute_full_rotation() {
        let v = NumArrayU8::new(vec![0xAA; 16]);
        let total_bits = 16 * 8;
        let p = v.permute(total_bits);
        assert_eq!(p.get_data(), v.get_data());
    }

    #[test]
    fn test_permute_single_bit() {
        let mut data = vec![0u8; 16];
        data[0] = 0x80;
        let v = NumArrayU8::new(data);
        let p = v.permute(1);
        let mut expected = vec![0u8; 16];
        expected[1] = 0x01;
        assert_eq!(p.get_data(), &expected);
    }

    #[test]
    fn test_permute_byte_aligned() {
        let mut data = vec![0u8; 16];
        data[0] = 0xFF;
        let v = NumArrayU8::new(data);
        let p = v.permute(8);
        let mut expected = vec![0u8; 16];
        expected[1] = 0xFF;
        assert_eq!(p.get_data(), &expected);
    }

    #[test]
    fn test_permute_inverse() {
        let v = NumArrayU8::new((0..8192).map(|i| (i % 256) as u8).collect());
        let total_bits = 8192 * 8;
        let k = 42;
        let p1 = v.permute(k);
        let p2 = p1.permute(total_bits - k);
        assert_eq!(p2.get_data(), v.get_data());
    }

    #[test]
    fn test_permute_orthogonality() {
        let v = NumArrayU8::new(vec![0xAB; 8192]);
        let p1 = v.permute(1);
        let p2 = v.permute(2);
        let hamming = p1.hamming_distance(&p2);
        assert!(hamming > 0);
    }

    // ---- BUNDLE tests ----

    #[test]
    fn test_bundle_unanimous() {
        let a = NumArrayU8::new(vec![0xFF; 8192]);
        let b = NumArrayU8::new(vec![0xFF; 8192]);
        let c = NumArrayU8::new(vec![0xFF; 8192]);
        let result = NumArrayU8::bundle(&[&a, &b, &c]);
        assert_eq!(result.get_data(), &vec![0xFF; 8192]);
    }

    #[test]
    fn test_bundle_all_zero() {
        let a = NumArrayU8::new(vec![0x00; 8192]);
        let b = NumArrayU8::new(vec![0x00; 8192]);
        let c = NumArrayU8::new(vec![0x00; 8192]);
        let result = NumArrayU8::bundle(&[&a, &b, &c]);
        assert_eq!(result.get_data(), &vec![0x00; 8192]);
    }

    #[test]
    fn test_bundle_majority_2_of_3() {
        let a = NumArrayU8::new(vec![0xFF; 8]);
        let b = NumArrayU8::new(vec![0xFF; 8]);
        let c = NumArrayU8::new(vec![0x00; 8]);
        let result = NumArrayU8::bundle(&[&a, &b, &c]);
        assert_eq!(result.get_data(), &vec![0xFF; 8]);
    }

    #[test]
    fn test_bundle_minority_1_of_3() {
        let a = NumArrayU8::new(vec![0xFF; 8]);
        let b = NumArrayU8::new(vec![0x00; 8]);
        let c = NumArrayU8::new(vec![0x00; 8]);
        let result = NumArrayU8::bundle(&[&a, &b, &c]);
        assert_eq!(result.get_data(), &vec![0x00; 8]);
    }

    #[test]
    fn test_bundle_tie_even() {
        let a = NumArrayU8::new(vec![0xFF; 8]);
        let b = NumArrayU8::new(vec![0xFF; 8]);
        let c = NumArrayU8::new(vec![0x00; 8]);
        let d = NumArrayU8::new(vec![0x00; 8]);
        let result = NumArrayU8::bundle(&[&a, &b, &c, &d]);
        assert_eq!(result.get_data(), &vec![0x00; 8]);
    }

    #[test]
    fn test_bundle_majority_3_of_4() {
        let a = NumArrayU8::new(vec![0xFF; 8]);
        let b = NumArrayU8::new(vec![0xFF; 8]);
        let c = NumArrayU8::new(vec![0xFF; 8]);
        let d = NumArrayU8::new(vec![0x00; 8]);
        let result = NumArrayU8::bundle(&[&a, &b, &c, &d]);
        assert_eq!(result.get_data(), &vec![0xFF; 8]);
    }

    #[test]
    fn test_bundle_single_vector() {
        let a = NumArrayU8::new(vec![0xAB; 8192]);
        let result = NumArrayU8::bundle(&[&a]);
        assert_eq!(result.get_data(), a.get_data());
    }

    #[test]
    fn test_bundle_large_8192() {
        let ones = NumArrayU8::new(vec![0xFF; 8192]);
        let zeros = NumArrayU8::new(vec![0x00; 8192]);
        let result = NumArrayU8::bundle(&[&ones, &ones, &ones, &zeros, &zeros]);
        assert_eq!(result.get_data(), &vec![0xFF; 8192]);
    }

    #[test]
    fn test_bundle_mixed_pattern() {
        let a = NumArrayU8::new(vec![0b10101010; 8]);
        let b = NumArrayU8::new(vec![0b11001100; 8]);
        let c = NumArrayU8::new(vec![0b11110000; 8]);
        let result = NumArrayU8::bundle(&[&a, &b, &c]);
        assert_eq!(result.get_data(), &vec![0b11101000; 8]);
    }

    #[test]
    fn test_bundle_large_count() {
        let ones = NumArrayU8::new(vec![0xFF; 8192]);
        let zeros = NumArrayU8::new(vec![0x00; 8192]);
        let mut vecs: Vec<&NumArrayU8> = Vec::new();
        for _ in 0..40 {
            vecs.push(&ones);
        }
        for _ in 0..24 {
            vecs.push(&zeros);
        }
        let result = NumArrayU8::bundle(&vecs);
        assert_eq!(result.get_data(), &vec![0xFF; 8192]);
    }

    #[test]
    fn test_bundle_1024_vectors() {
        let ones = NumArrayU8::new(vec![0xFF; 8192]);
        let zeros = NumArrayU8::new(vec![0x00; 8192]);
        let mut vecs: Vec<&NumArrayU8> = Vec::new();
        for _ in 0..600 {
            vecs.push(&ones);
        }
        for _ in 0..424 {
            vecs.push(&zeros);
        }
        let result = NumArrayU8::bundle(&vecs);
        assert_eq!(result.get_data(), &vec![0xFF; 8192]);
    }

    // ---- Larger vector sizes: 2048, 16384, 65536 bytes ----

    #[test]
    fn test_bundle_2048_bytes() {
        // CogRecord single container: 16384 bits = 2048 bytes
        let ones = NumArrayU8::new(vec![0xFF; 2048]);
        let zeros = NumArrayU8::new(vec![0x00; 2048]);
        let result = NumArrayU8::bundle(&[&ones, &ones, &ones, &zeros, &zeros]);
        assert_eq!(result.get_data(), &vec![0xFF; 2048]);
    }

    #[test]
    fn test_bundle_16384_bytes() {
        let ones = NumArrayU8::new(vec![0xFF; 16384]);
        let zeros = NumArrayU8::new(vec![0x00; 16384]);
        let result = NumArrayU8::bundle(&[&ones, &ones, &ones, &zeros, &zeros]);
        assert_eq!(result.get_data(), &vec![0xFF; 16384]);
    }

    #[test]
    fn test_bundle_65536_bytes() {
        let ones = NumArrayU8::new(vec![0xFF; 65536]);
        let zeros = NumArrayU8::new(vec![0x00; 65536]);
        let result = NumArrayU8::bundle(&[&ones, &ones, &ones, &zeros, &zeros]);
        assert_eq!(result.get_data(), &vec![0xFF; 65536]);
    }

    #[test]
    fn test_bundle_at_threshold_boundary() {
        // Test right at BUNDLE_RIPPLE_THRESHOLD (n=16) and n=17
        let ones = NumArrayU8::new(vec![0xFF; 8192]);
        let zeros = NumArrayU8::new(vec![0x00; 8192]);

        // n=16: should use naive path (9 ones, 7 zeros → majority 1)
        let mut vecs16: Vec<&NumArrayU8> = Vec::new();
        for _ in 0..9 {
            vecs16.push(&ones);
        }
        for _ in 0..7 {
            vecs16.push(&zeros);
        }
        let result16 = NumArrayU8::bundle(&vecs16);
        assert_eq!(result16.get_data(), &vec![0xFF; 8192]);

        // n=17: should use ripple path (10 ones, 7 zeros → majority 1)
        let mut vecs17: Vec<&NumArrayU8> = Vec::new();
        for _ in 0..10 {
            vecs17.push(&ones);
        }
        for _ in 0..7 {
            vecs17.push(&zeros);
        }
        let result17 = NumArrayU8::bundle(&vecs17);
        assert_eq!(result17.get_data(), &vec![0xFF; 8192]);
    }

    #[test]
    fn test_permute_16384() {
        let v = NumArrayU8::new((0..16384).map(|i| (i % 256) as u8).collect());
        let total_bits = 16384 * 8;
        let k = 100;
        let p1 = v.permute(k);
        let p2 = p1.permute(total_bits - k);
        assert_eq!(p2.get_data(), v.get_data());
    }

    #[test]
    fn test_permute_65536() {
        let v = NumArrayU8::new((0..65536).map(|i| (i % 256) as u8).collect());
        let total_bits = 65536 * 8;
        let k = 257;
        let p1 = v.permute(k);
        let p2 = p1.permute(total_bits - k);
        assert_eq!(p2.get_data(), v.get_data());
    }

    #[test]
    fn test_bundle_mixed_pattern_65536() {
        let a = NumArrayU8::new(vec![0b10101010; 65536]);
        let b = NumArrayU8::new(vec![0b11001100; 65536]);
        let c = NumArrayU8::new(vec![0b11110000; 65536]);
        let result = NumArrayU8::bundle(&[&a, &b, &c]);
        assert_eq!(result.get_data(), &vec![0b11101000; 65536]);
    }

    // ---- DOT_I8 tests ----

    #[test]
    fn test_dot_i8_simple() {
        let a = NumArrayU8::new(vec![1, 2, 3, 4]);
        let b = NumArrayU8::new(vec![1, 2, 3, 4]);
        // 1*1 + 2*2 + 3*3 + 4*4 = 1 + 4 + 9 + 16 = 30
        assert_eq!(a.dot_i8(&b), 30);
    }

    #[test]
    fn test_dot_i8_negative() {
        // 0xFF as i8 = -1, 0xFE as i8 = -2
        let a = NumArrayU8::new(vec![0xFF, 0xFE, 0x01, 0x02]);
        let b = NumArrayU8::new(vec![0xFF, 0xFE, 0x01, 0x02]);
        // (-1)(-1) + (-2)(-2) + 1*1 + 2*2 = 1 + 4 + 1 + 4 = 10
        assert_eq!(a.dot_i8(&b), 10);
    }

    #[test]
    fn test_dot_i8_orthogonal() {
        // Approximation of orthogonal int8 vectors
        let a = NumArrayU8::new(vec![1, 0, 1, 0]);
        let b = NumArrayU8::new(vec![0, 1, 0, 1]);
        assert_eq!(a.dot_i8(&b), 0);
    }

    #[test]
    fn test_dot_i8_large() {
        // 2048 bytes (full 16384-bit container), all 1s
        let a = NumArrayU8::new(vec![1; 2048]);
        let b = NumArrayU8::new(vec![1; 2048]);
        assert_eq!(a.dot_i8(&b), 2048);
    }

    #[test]
    fn test_dot_i8_max_values() {
        // 127 × 127 × 1024 dimensions
        let a = NumArrayU8::new(vec![127; 1024]);
        let b = NumArrayU8::new(vec![127; 1024]);
        assert_eq!(a.dot_i8(&b), 127i64 * 127 * 1024);
    }

    #[test]
    fn test_norm_sq_i8() {
        let a = NumArrayU8::new(vec![3, 4]); // 3²+4² = 25
        assert_eq!(a.norm_sq_i8(), 25);
    }

    #[test]
    fn test_cosine_i8_identical() {
        let a = NumArrayU8::new(vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let cos = a.cosine_i8(&a);
        assert!((cos - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_i8_opposite() {
        // a = [1,1,1,1], b = [-1,-1,-1,-1] (0xFF)
        let a = NumArrayU8::new(vec![1, 1, 1, 1]);
        let b = NumArrayU8::new(vec![0xFF, 0xFF, 0xFF, 0xFF]);
        let cos = a.cosine_i8(&b);
        assert!((cos - (-1.0)).abs() < 1e-10);
    }

    // ---- Adaptive Hamming distance tests ----

    #[test]
    fn test_adaptive_hamming_identical() {
        let a = NumArrayU8::new(vec![0xAA; 2048]);
        let result = a.hamming_distance_adaptive(&a, 100);
        assert_eq!(result, Some(0));
    }

    #[test]
    fn test_adaptive_hamming_reject_far() {
        let a = NumArrayU8::new(vec![0xFF; 2048]);
        let b = NumArrayU8::new(vec![0x00; 2048]);
        // Distance is 2048*8 = 16384, threshold 100 → rejected early
        assert!(a.hamming_distance_adaptive(&b, 100).is_none());
    }

    #[test]
    fn test_adaptive_hamming_accept_close() {
        let mut data = vec![0xAA; 2048];
        let query = NumArrayU8::new(data.clone());
        // Scatter flipped bytes uniformly across the vector so the sample
        // is representative (every ~200 bytes flip one byte = ~10 flips)
        let stride = 2048 / 10;
        for i in 0..10 {
            data[i * stride] = !data[i * stride];
        }
        let target = NumArrayU8::new(data);
        let d = query.hamming_distance(&target);
        // d = 80 bits, threshold = 90 → should accept
        let result = query.hamming_distance_adaptive(&target, d + 10);
        assert_eq!(result, Some(d));
    }

    #[test]
    fn test_adaptive_hamming_exact_threshold() {
        let mut data = vec![0xAA; 2048];
        let query = NumArrayU8::new(data.clone());
        // Scatter flips uniformly
        let stride = 2048 / 5;
        for i in 0..5 {
            data[i * stride] = !data[i * stride];
        }
        let target = NumArrayU8::new(data);
        let d = query.hamming_distance(&target);
        // At exact threshold should accept
        assert_eq!(query.hamming_distance_adaptive(&target, d), Some(d));
        // Below threshold should reject (at stage 3)
        assert!(query.hamming_distance_adaptive(&target, d - 1).is_none());
    }

    #[test]
    fn test_adaptive_search_batch() {
        let query = NumArrayU8::new(vec![0xAA; 2048]);
        let mut db_data = vec![0xAA; 2048]; // vec 0: identical (d=0)
        db_data.extend(vec![0x55; 2048]); // vec 1: maximally different
        db_data.extend(vec![0xAA; 2048]); // vec 2: identical (d=0)
        db_data.extend(vec![0x00; 2048]); // vec 3: very different
        let db = NumArrayU8::new(db_data);

        let results = query.hamming_search_adaptive(&db, 2048, 4, 100);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], (0, 0));
        assert_eq!(results[1], (2, 0));
    }

    #[test]
    fn test_adaptive_search_8192() {
        let query = NumArrayU8::new(vec![0xAA; 8192]);
        let mut db_data = Vec::new();
        // 100 random-ish vectors, only vec 0 and 50 match
        for i in 0..100 {
            if i == 0 || i == 50 {
                db_data.extend(vec![0xAA; 8192]);
            } else {
                db_data.extend((0..8192).map(|j| ((i * 37 + j * 13) % 256) as u8));
            }
        }
        let db = NumArrayU8::new(db_data);
        let results = query.hamming_search_adaptive(&db, 8192, 100, 100);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0);
        assert_eq!(results[1].0, 50);
    }

    // ---- Adaptive cosine search tests ----

    #[test]
    fn test_cosine_search_identical() {
        let query = NumArrayU8::new(vec![100u8; 1024]);
        let mut db_data = vec![100u8; 1024]; // identical
        db_data.extend(vec![0u8; 1024]); // different
        db_data.extend(vec![100u8; 1024]); // identical
        let db = NumArrayU8::new(db_data);

        let results = query.cosine_search_adaptive(&db, 1024, 3, 0.9);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0);
        assert!((results[0].1 - 1.0).abs() < 1e-5);
        assert_eq!(results[1].0, 2);
    }

    #[test]
    fn test_cosine_search_2048d() {
        let query = NumArrayU8::new(vec![50u8; 2048]);
        let mut db_data = Vec::new();
        for i in 0..50 {
            if i == 10 || i == 30 {
                db_data.extend(vec![50u8; 2048]); // match
            } else {
                db_data.extend((0..2048).map(|j| ((i * 71 + j * 13) % 256) as u8));
            }
        }
        let db = NumArrayU8::new(db_data);
        let results = query.cosine_search_adaptive(&db, 2048, 50, 0.95);
        assert!(results.len() >= 2);
        let match_indices: Vec<usize> = results.iter().map(|r| r.0).collect();
        assert!(match_indices.contains(&10));
        assert!(match_indices.contains(&30));
    }

    // ---- BIND tests ----

    #[test]
    fn test_bind_involution() {
        let a = NumArrayU8::new((0..8192).map(|i| (i % 256) as u8).collect());
        let b = NumArrayU8::new((0..8192).map(|i| ((i * 7 + 13) % 256) as u8).collect());
        let bound = a.bind(&b);
        let recovered = bound.bind(&b);
        assert_eq!(recovered.get_data(), a.get_data());
    }

    #[test]
    fn test_bind_commutative() {
        let a = NumArrayU8::new(vec![0xAA; 8192]);
        let b = NumArrayU8::new(vec![0x55; 8192]);
        let ab = a.bind(&b);
        let ba = b.bind(&a);
        assert_eq!(ab.get_data(), ba.get_data());
    }

    // ---- Integration: edge encoding and recovery ----

    #[test]
    fn test_edge_encode_decode() {
        let src = NumArrayU8::new((0..8192).map(|i| (i % 256) as u8).collect());
        let rel = NumArrayU8::new((0..8192).map(|i| ((i * 3) % 256) as u8).collect());
        let tgt = NumArrayU8::new((0..8192).map(|i| ((i * 7 + 42) % 256) as u8).collect());

        let total_bits = 8192 * 8;
        let perm_rel = rel.permute(1);
        let perm_tgt = tgt.permute(2);
        let edge = &(&src ^ &perm_rel) ^ &perm_tgt;

        let recovered_perm_tgt = &(&edge ^ &src) ^ &perm_rel;
        let recovered_tgt = recovered_perm_tgt.permute(total_bits - 2);
        assert_eq!(recovered_tgt.get_data(), tgt.get_data());
    }

    #[test]
    fn test_edge_encode_decode_65536() {
        let src = NumArrayU8::new((0..65536).map(|i| (i % 256) as u8).collect());
        let rel = NumArrayU8::new((0..65536).map(|i| ((i * 3) % 256) as u8).collect());
        let tgt = NumArrayU8::new((0..65536).map(|i| ((i * 7 + 42) % 256) as u8).collect());

        let total_bits = 65536 * 8;
        let perm_rel = rel.permute(1);
        let perm_tgt = tgt.permute(2);
        let edge = &(&src ^ &perm_rel) ^ &perm_tgt;

        let recovered_perm_tgt = &(&edge ^ &src) ^ &perm_rel;
        let recovered_tgt = recovered_perm_tgt.permute(total_bits - 2);
        assert_eq!(recovered_tgt.get_data(), tgt.get_data());
    }

    // ---- HDR search F32 dequantize tests ----

    #[test]
    fn test_hdr_search_f32_identical() {
        let query = NumArrayU8::new(vec![200u8; 2048]);
        let mut db_data = vec![200u8; 2048]; // identical
        db_data.extend(vec![50u8; 2048]); // very different
        let db = NumArrayU8::new(db_data);

        let results = query.hdr_search_f32(&db, 2048, 2, 20000, 1.0, 128);
        assert!(!results.is_empty());
        // Identical vector should have cosine ~1.0
        let ident = results.iter().find(|r| r.0 == 0).unwrap();
        assert!(
            (ident.2 - 1.0).abs() < 0.01,
            "Expected ~1.0, got {}",
            ident.2
        );
    }

    // ---- HDR search delta tests ----

    #[test]
    fn test_hdr_search_delta_basic() {
        let query = NumArrayU8::new(vec![0xAA; 2048]);
        let mut db_data = vec![0xAA; 2048]; // identical
        db_data.extend(vec![0x55; 2048]); // maximally different
        let db = NumArrayU8::new(db_data);

        let results = query.hdr_search_delta(&db, 2048, 2, 100, 0.3);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
        assert!(results[0].2.is_finite());
    }

    // ---- Backward compatibility ----

    #[test]
    fn test_hamming_search_adaptive_backward_compat() {
        let query = NumArrayU8::new(vec![0xAA; 2048]);
        let mut db_data = vec![0xAA; 2048]; // identical
        db_data.extend(vec![0x55; 2048]); // maximally different
        db_data.extend(vec![0xAA; 2048]); // identical
        db_data.extend(vec![0x00; 2048]); // different
        let db = NumArrayU8::new(db_data);

        let results = query.hamming_search_adaptive(&db, 2048, 4, 100);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], (0, 0));
        assert_eq!(results[1], (2, 0));
    }
}
