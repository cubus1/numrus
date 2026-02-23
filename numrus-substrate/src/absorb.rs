//! Organic X-Trans Holograph: Write = Clean = Learn in one operation.
//!
//! Instead of 30+ operations (carriers, focus, gabor, spectral cleaning, etc.),
//! this module uses SPATIAL SAMPLING TOPOLOGY to prevent interference:
//!
//! - **X-Trans pattern**: Fibonacci lattice assigns each concept its own positions.
//!   Cross-channel interference is structurally impossible.
//! - **Organic WAL**: Write-ahead re-referencing orthogonalizes before writing.
//!   The projections ARE the learning signal. No separate cleaning pass.
//! - **Absorption**: Each byte self-regulates via receptivity. No separate
//!   overexposure check needed.
//!
//! Inspired by Fujifilm X-Trans (irregular sampling eliminates moire),
//! Fibonacci carrier spacing (frequency gaps prevent resonance), and
//! Write-Ahead Logging (re-reference IS cleaning).

// ---------------------------------------------------------------------------
// Part 1: X-Trans Sampling Pattern
// ---------------------------------------------------------------------------

/// X-Trans inspired sampling pattern for holographic containers.
///
/// Maps D positions to C channels using a Fibonacci lattice.
/// Each channel gets D/C positions, maximally uniformly distributed.
/// No two adjacent positions share a channel.
///
/// This is the anti-aliasing filter BUILT INTO the sampling topology.
pub struct XTransPattern {
    /// Total number of positions (container size in bytes).
    pub d: usize,
    /// Number of channels (concept slots).
    pub channels: usize,
    /// Which channel owns each position: position_channel[j] -> channel_id.
    pub position_channel: Vec<u8>,
    /// Positions owned by each channel: channel_positions[c] -> sorted positions.
    pub channel_positions: Vec<Vec<usize>>,
    /// Positions per channel (= D / C, approximately).
    pub positions_per_channel: usize,
    /// Golden step used for lattice generation.
    pub golden_step: usize,
}

impl XTransPattern {
    /// Create a new Fibonacci-lattice sampling pattern.
    ///
    /// D positions distributed across C channels.
    /// Each channel gets approximately D/C positions.
    pub fn new(d: usize, channels: usize) -> Self {
        assert!(channels > 0 && channels <= d);

        // Use continuous golden ratio for channel assignment.
        // frac(j * phi^{-1}) maps each position to [0,1) quasi-uniformly,
        // then multiply by C and floor to get channel index.
        // This avoids GCD issues with integer modular arithmetic.
        let phi_inv = 0.618_033_988_749_895; // 1/phi = phi - 1
        let golden_step = (d as f64 / (phi_inv * phi_inv + 1.0)).round() as usize;

        let mut position_channel = vec![0u8; d];
        let mut channel_positions = vec![Vec::new(); channels];

        for j in 0..d {
            let frac = (j as f64 * phi_inv) - (j as f64 * phi_inv).floor();
            let channel = (frac * channels as f64).floor() as usize;
            let channel = channel.min(channels - 1); // safety clamp
            position_channel[j] = channel as u8;
            channel_positions[channel].push(j);
        }

        let positions_per_channel = d / channels;

        Self {
            d,
            channels,
            position_channel,
            channel_positions,
            positions_per_channel,
            golden_step,
        }
    }

    /// Verify the pattern quality: measure minimum distance between
    /// same-channel positions. Should be >= channels for good separation.
    pub fn min_same_channel_distance(&self) -> usize {
        let mut min_dist = self.d;
        for ch in 0..self.channels {
            let positions = &self.channel_positions[ch];
            for w in positions.windows(2) {
                let dist = w[1] - w[0];
                min_dist = min_dist.min(dist);
            }
        }
        min_dist
    }

    /// Measure the uniformity: standard deviation of channel sizes.
    /// Perfect uniformity = 0.0.
    pub fn size_uniformity(&self) -> f32 {
        let mean = self.d as f32 / self.channels as f32;
        let variance: f32 = self
            .channel_positions
            .iter()
            .map(|ch| {
                let diff = ch.len() as f32 - mean;
                diff * diff
            })
            .sum::<f32>()
            / self.channels as f32;
        variance.sqrt()
    }
}

/// Multi-resolution X-Trans patterns for three-temperature operation.
///
/// Hot:  all positions available (no channeling, full resolution thinking)
/// Warm: C channels x D/C positions (structured, interference-free)
/// Cold: C channels x 1 bit per position (binary crystal for search)
pub struct MultiResPattern {
    /// Warm pattern: full X-Trans layout.
    pub warm: XTransPattern,
    /// Channel masks: warm_channel_masks[c][j] = true if position j belongs to channel c.
    pub warm_channel_masks: Vec<Vec<bool>>,
}

impl MultiResPattern {
    pub fn new(d: usize, channels: usize) -> Self {
        let warm = XTransPattern::new(d, channels);

        let warm_channel_masks = (0..channels)
            .map(|ch| {
                let mut mask = vec![false; d];
                for &pos in &warm.channel_positions[ch] {
                    mask[pos] = true;
                }
                mask
            })
            .collect();

        Self {
            warm,
            warm_channel_masks,
        }
    }
}

// ---------------------------------------------------------------------------
// Part 2: Organic Write -- Receptivity-Gated Absorption
// ---------------------------------------------------------------------------

/// Compute receptivity at each position: how much room remains.
///
/// Full (|value| = 127): receptivity = 0.0 -> concept bounces off
/// Empty (value = 0):    receptivity = 1.0 -> fully absorbs
/// Partial:              linear interpolation
#[inline]
pub fn receptivity(current: i8) -> f32 {
    1.0 - (current as f32).abs() / 127.0
}

/// Organic write: concept soaks into container proportional to
/// available capacity at each position.
///
/// Returns the absorption ratio:
///   1.0 = fully absorbed (container had room everywhere)
///   0.0 = fully rejected (container saturated)
pub fn organic_write(
    container: &mut [i8],
    template: &[i8],
    amplitude: f32,
    positions: &[usize],
) -> f32 {
    let mut absorbed = 0.0f32;
    let mut total = 0.0f32;

    for &j in positions {
        let recv = receptivity(container[j]);
        let full_write = amplitude * template[j] as f32;
        let actual_write = full_write * recv;

        let new_val = container[j] as f32 + actual_write;
        container[j] = new_val.round().clamp(-128.0, 127.0) as i8;

        absorbed += actual_write.abs();
        total += full_write.abs();
    }

    if total > 1e-10 {
        absorbed / total
    } else {
        1.0
    }
}

/// Organic write from f32 template (after orthogonalization).
pub fn organic_write_f32(
    container: &mut [i8],
    template: &[f32],
    amplitude: f32,
    positions: &[usize],
) -> f32 {
    let mut absorbed = 0.0f32;
    let mut total = 0.0f32;

    for &j in positions {
        let recv = receptivity(container[j]);
        let full_write = amplitude * template[j];
        let actual_write = full_write * recv;

        let new_val = container[j] as f32 + actual_write;
        container[j] = new_val.round().clamp(-128.0, 127.0) as i8;

        absorbed += actual_write.abs();
        total += full_write.abs();
    }

    if total > 1e-10 {
        absorbed / total
    } else {
        1.0
    }
}

/// Read a concept from the container using only its channel positions.
///
/// Returns (cosine_similarity, projection_amplitude).
pub fn organic_read(container: &[i8], template: &[i8], positions: &[usize]) -> (f32, f32) {
    let mut dot = 0.0f64;
    let mut energy = 0.0f64;
    let mut template_energy = 0.0f64;

    for &j in positions {
        dot += container[j] as f64 * template[j] as f64;
        energy += container[j] as f64 * container[j] as f64;
        template_energy += template[j] as f64 * template[j] as f64;
    }

    let norm = (energy.sqrt() * template_energy.sqrt()).max(1e-10);
    let similarity = (dot / norm) as f32;
    let amplitude = (dot / template_energy.max(1e-10)) as f32;

    (similarity, amplitude)
}

// ---------------------------------------------------------------------------
// Private helper: Cholesky solve (extracted from linalg.rs)
// ---------------------------------------------------------------------------

/// Cholesky decomposition and solve for symmetric positive definite matrix.
fn cholesky_solve(gram: &[f64], rhs: &[f64], k: usize) -> Vec<f64> {
    // Cholesky: G = L * L^T
    let mut l = vec![0.0f64; k * k];
    for i in 0..k {
        for j in 0..=i {
            let mut sum = gram[i * k + j];
            for p in 0..j {
                sum -= l[i * k + p] * l[j * k + p];
            }
            if i == j {
                // Diagonal: handle near-zero for numerical stability
                l[i * k + j] = if sum > 1e-12 { sum.sqrt() } else { 1e-6 };
            } else {
                l[i * k + j] = sum / l[j * k + j];
            }
        }
    }

    // Forward: L * y = rhs
    let mut y = vec![0.0f64; k];
    for i in 0..k {
        let mut sum = rhs[i];
        for j in 0..i {
            sum -= l[i * k + j] * y[j];
        }
        y[i] = sum / l[i * k + i];
    }

    // Backward: L^T * alpha = y
    let mut alpha = vec![0.0f64; k];
    for i in (0..k).rev() {
        let mut sum = y[i];
        for j in (i + 1)..k {
            sum -= l[j * k + i] * alpha[j];
        }
        alpha[i] = sum / l[i * k + i];
    }

    alpha
}

// ---------------------------------------------------------------------------
// Part 3: Organic WAL -- Write-Ahead Re-Referencing
// ---------------------------------------------------------------------------

/// Result of an organic WAL write.
#[derive(Clone, Debug)]
pub struct WriteResult {
    /// How much of the concept was absorbed (0.0 = rejected, 1.0 = full).
    pub absorption: f32,
    /// Projection of the concept onto each known concept (learning signal).
    pub projections: Vec<f32>,
    /// Which channel was used for the write.
    pub channel: usize,
    /// Number of positions written to.
    pub positions_written: usize,
}

/// Write-Ahead Log with orthogonal re-referencing.
///
/// Before writing concept C:
///   1. Compute projections of C onto each known concept template
///   2. Subtract these projections from C (Gram-Schmidt)
///   3. Write the orthogonalized C via organic absorption
///   4. Return the projections as learning signal
pub struct OrganicWAL {
    /// The X-Trans pattern (shared across all writes).
    pub pattern: XTransPattern,
    /// Known concept templates: [concept_idx][position] -> i8
    known_templates: Vec<Vec<i8>>,
    /// Precomputed template norms for each known concept.
    template_norms: Vec<f64>,
    /// Current concept coefficients (updated by projections).
    pub coefficients: Vec<f32>,
    /// Concept IDs (parallel to coefficients and known_templates).
    pub concept_ids: Vec<u32>,
}

impl OrganicWAL {
    /// Create a new OrganicWAL with the given sampling pattern.
    pub fn new(pattern: XTransPattern) -> Self {
        Self {
            pattern,
            known_templates: Vec::new(),
            template_norms: Vec::new(),
            coefficients: Vec::new(),
            concept_ids: Vec::new(),
        }
    }

    /// Register a concept template. Must be done before writing.
    pub fn register_concept(&mut self, concept_id: u32, template: Vec<i8>) {
        let norm: f64 = template.iter().map(|&v| (v as f64) * (v as f64)).sum();
        self.known_templates.push(template);
        self.template_norms.push(norm);
        self.coefficients.push(0.0);
        self.concept_ids.push(concept_id);
    }

    /// Number of registered concepts.
    pub fn k(&self) -> usize {
        self.known_templates.len()
    }

    /// Write a concept into the container with full re-referencing.
    ///
    /// Returns absorption, projections (learning signal), and channel info.
    pub fn write(
        &mut self,
        container: &mut [i8],
        concept_idx: usize,
        amplitude: f32,
        learning_rate: f32,
    ) -> WriteResult {
        let d = self.pattern.d;
        let template = &self.known_templates[concept_idx];

        // Step 1: Gram-Schmidt orthogonalization against all OTHER concepts
        let mut orthogonal = vec![0.0f32; d];
        for j in 0..d {
            orthogonal[j] = template[j] as f32;
        }

        let mut projections = vec![0.0f32; self.k()];

        for i in 0..self.k() {
            if i == concept_idx {
                continue;
            }

            let known = &self.known_templates[i];
            let norm = self.template_norms[i];
            if norm < 1e-10 {
                continue;
            }

            let mut dot = 0.0f64;
            for j in 0..d {
                dot += orthogonal[j] as f64 * known[j] as f64;
            }
            let proj_coeff = (dot / norm) as f32;
            projections[i] = proj_coeff;

            // Subtract projection
            for j in 0..d {
                orthogonal[j] -= proj_coeff * known[j] as f32;
            }
        }

        // Step 2: Write to this concept's channel positions
        let channel = concept_idx % self.pattern.channels;
        let positions = &self.pattern.channel_positions[channel];

        // Step 3: Organic write of the orthogonalized template
        let absorption = organic_write_f32(container, &orthogonal, amplitude, positions);

        // Step 4: Update coefficients from projections (learning)
        for i in 0..self.k() {
            if i == concept_idx {
                self.coefficients[i] += amplitude * absorption;
            } else {
                self.coefficients[i] += learning_rate * projections[i];
            }
        }

        WriteResult {
            absorption,
            projections,
            channel,
            positions_written: positions.len(),
        }
    }

    /// Write with plasticity modulation.
    ///
    /// The learning rate for coefficient updates is multiplied by
    /// each concept's plasticity. Rigid concepts resist change.
    pub fn write_plastic(
        &mut self,
        container: &mut [i8],
        concept_idx: usize,
        amplitude: f32,
        learning_rate: f32,
        plasticity: &mut PlasticityTracker,
    ) -> WriteResult {
        let self_plasticity = plasticity.record_write(concept_idx);

        let d = self.pattern.d;
        let template = &self.known_templates[concept_idx];

        // Gram-Schmidt orthogonalization
        let mut orthogonal = vec![0.0f32; d];
        for j in 0..d {
            orthogonal[j] = template[j] as f32;
        }

        let mut projections = vec![0.0f32; self.k()];

        for i in 0..self.k() {
            if i == concept_idx {
                continue;
            }

            let known = &self.known_templates[i];
            let norm = self.template_norms[i];
            if norm < 1e-10 {
                continue;
            }

            let mut dot = 0.0f64;
            for j in 0..d {
                dot += orthogonal[j] as f64 * known[j] as f64;
            }
            let proj_coeff = (dot / norm) as f32;
            projections[i] = proj_coeff;

            for j in 0..d {
                orthogonal[j] -= proj_coeff * known[j] as f32;
            }
        }

        // Organic write with self-plasticity modulating amplitude
        let effective_amplitude = amplitude * self_plasticity;
        let channel = concept_idx % self.pattern.channels;
        let positions = &self.pattern.channel_positions[channel];

        let absorption = organic_write_f32(container, &orthogonal, effective_amplitude, positions);

        // Update coefficients with per-concept plasticity
        for i in 0..self.k() {
            let target_plasticity = plasticity.plasticity(i);
            if i == concept_idx {
                self.coefficients[i] += effective_amplitude * absorption;
            } else {
                self.coefficients[i] += learning_rate * target_plasticity * projections[i];
            }
        }

        WriteResult {
            absorption,
            projections,
            channel,
            positions_written: positions.len(),
        }
    }

    /// Read a concept from the container using its channel positions.
    pub fn read(&self, container: &[i8], concept_idx: usize) -> (f32, f32) {
        let channel = concept_idx % self.pattern.channels;
        let positions = &self.pattern.channel_positions[channel];
        let template = &self.known_templates[concept_idx];
        organic_read(container, template, positions)
    }

    /// Batch read: read all registered concepts.
    pub fn read_all(&self, container: &[i8]) -> Vec<(u32, f32, f32)> {
        (0..self.k())
            .map(|i| {
                let (sim, amp) = self.read(container, i);
                (self.concept_ids[i], sim, amp)
            })
            .collect()
    }

    /// Full orthogonal projection: extract exact coefficients.
    ///
    /// Channel-aware: concepts on different channels have zero Gram
    /// off-diagonal blocks by X-Trans construction.
    pub fn surgical_extract(&self, container: &[i8]) -> Vec<f32> {
        let k = self.k();
        if k == 0 {
            return vec![];
        }

        // Build Gram matrix using channel-aware dot products
        let mut gram = vec![0.0f64; k * k];
        for i in 0..k {
            let ch_i = i % self.pattern.channels;
            let pos_i = &self.pattern.channel_positions[ch_i];
            for j in i..k {
                let ch_j = j % self.pattern.channels;
                if ch_i == ch_j {
                    // Same channel: dot product over shared positions
                    let mut dot = 0.0f64;
                    for &p in pos_i {
                        dot +=
                            self.known_templates[i][p] as f64 * self.known_templates[j][p] as f64;
                    }
                    gram[i * k + j] = dot;
                    gram[j * k + i] = dot;
                } else {
                    // Different channels: zero by X-Trans construction
                    gram[i * k + j] = 0.0;
                    gram[j * k + i] = 0.0;
                }
            }
        }

        // W * b: template dot products against container
        let mut wb = vec![0.0f64; k];
        for i in 0..k {
            let ch = i % self.pattern.channels;
            let positions = &self.pattern.channel_positions[ch];
            for &p in positions {
                wb[i] += self.known_templates[i][p] as f64 * container[p] as f64;
            }
        }

        // Solve via Cholesky
        let recovered = cholesky_solve(&gram, &wb, k);
        recovered.iter().map(|&v| v as f32).collect()
    }

    /// Get a reference to a registered concept's template.
    pub fn template(&self, concept_idx: usize) -> &[i8] {
        &self.known_templates[concept_idx]
    }

    /// Get the precomputed squared norm of a concept's template.
    pub fn template_norm(&self, concept_idx: usize) -> f64 {
        self.template_norms[concept_idx]
    }

    /// Replace a concept's template and recompute its norm.
    ///
    /// Used by the recognizer's incremental learning: running average is
    /// quantized back to i8, then the WAL template is updated in-place.
    pub fn update_template(&mut self, concept_idx: usize, template: &[i8]) {
        let d = self.known_templates[concept_idx].len();
        assert_eq!(template.len(), d);
        self.known_templates[concept_idx].copy_from_slice(template);
        self.template_norms[concept_idx] = template.iter().map(|&v| (v as f64) * (v as f64)).sum();
    }

    /// Compute the residual: container minus reconstructed known-concept contributions.
    ///
    /// For each concept i with coefficient c_i, subtracts c_i * template_i
    /// from the container at that concept's channel positions only.
    /// Returns f32 residual (not quantized, to preserve precision).
    pub fn compute_residual(&self, container: &[i8], coefficients: &[f32]) -> Vec<f32> {
        let d = self.pattern.d;
        let mut residual = vec![0.0f32; d];

        for j in 0..d {
            residual[j] = container[j] as f32;
        }

        for i in 0..self.k().min(coefficients.len()) {
            let channel = i % self.pattern.channels;
            let positions = &self.pattern.channel_positions[channel];
            for &j in positions {
                residual[j] -= coefficients[i] * self.known_templates[i][j] as f32;
            }
        }

        residual
    }

    /// Residual energy: sum of squares of the residual.
    ///
    /// Low residual energy means the known concepts explain the container well.
    pub fn residual_energy(&self, container: &[i8], coefficients: &[f32]) -> f32 {
        let residual = self.compute_residual(container, coefficients);
        residual.iter().map(|&v| v * v).sum::<f32>() / residual.len() as f32
    }
}

// ---------------------------------------------------------------------------
// Part 4: Plasticity -- Per-Concept Adaptive Learning Rate
// ---------------------------------------------------------------------------

/// Per-concept plasticity: controls how quickly each concept adapts.
///
/// Bathtub curve: high early (exploring), low middle (consolidating),
/// medium late (allowing revision).
///
/// Base-aware: the initial caution phase scales inversely with base cardinality.
/// Narrow bases (cardinality 5, range [-2,2]) get HIGHER initial plasticity
/// because their small values are destroyed by aggressive amplitude attenuation.
/// Wide bases (cardinality 9, range [-4,4]) can afford MORE caution.
///
/// Formula: base_scale = reference_cardinality / cardinality
/// where reference = 8 (the midpoint of our base range).
/// cardinality 5: 8/5 = 1.6 -> clamped to 1.5 -> new=0.5*1.5=0.75 (less caution)
/// cardinality 7: 8/7 = 1.14 -> new=0.5*1.14=0.57
/// cardinality 9: 8/9 = 0.89 -> new=0.5*0.89=0.44 (more caution, can afford it)
/// cardinality 2: 8/2 = 4.0 -> clamped to 1.5 (binary is hopeless anyway)
pub struct PlasticityTracker {
    /// Write count per concept.
    pub write_counts: Vec<u32>,
    /// Last write timestamp per concept (for decay detection).
    pub last_write: Vec<u64>,
    /// Current global timestamp (incremented per write operation).
    pub clock: u64,
    /// Decay window: concepts not written within this many ticks are "decaying".
    pub decay_window: u64,
    /// Base-aware scaling factor for initial plasticity phases.
    /// Computed as clamp(8.0 / cardinality, 0.5, 1.5).
    base_scale: f32,
}

impl PlasticityTracker {
    pub fn new(k: usize, decay_window: u64) -> Self {
        Self {
            write_counts: vec![0; k],
            last_write: vec![0; k],
            clock: 0,
            decay_window,
            base_scale: 1.0, // default: no scaling (backwards compatible)
        }
    }

    /// Create with cardinality-aware plasticity scaling.
    ///
    /// The initial caution phases (new, young) are scaled by
    /// 8 / cardinality, so narrow bases get LESS attenuation
    /// (higher initial plasticity) and wide bases get MORE.
    pub fn new_with_cardinality(k: usize, decay_window: u64, cardinality: usize) -> Self {
        let scale = (8.0 / cardinality as f32).clamp(0.5, 1.5);
        Self {
            write_counts: vec![0; k],
            last_write: vec![0; k],
            clock: 0,
            decay_window,
            base_scale: scale,
        }
    }

    /// Record a write to concept i and return its plasticity multiplier.
    pub fn record_write(&mut self, concept_idx: usize) -> f32 {
        self.clock += 1;
        self.write_counts[concept_idx] += 1;
        self.last_write[concept_idx] = self.clock;
        self.plasticity(concept_idx)
    }

    /// Get current plasticity for concept i.
    ///
    /// The bathtub curve's early phases are scaled by base_scale (= 8/cardinality):
    ///   New (0-2 writes):    0.5 * base_scale  -> cardinality 5: 0.75, cardinality 9: 0.44
    ///   Young (3-5 writes):  0.8 * base_scale  -> cardinality 5: 1.0+, cardinality 9: 0.71
    ///   Active (6-20):       1.0 (always full)
    ///   Consolidating/Stable/Ancient: unchanged (rigidity doesn't need scaling)
    pub fn plasticity(&self, concept_idx: usize) -> f32 {
        let count = self.write_counts[concept_idx];
        let age = self.clock.saturating_sub(self.last_write[concept_idx]);

        // Decaying: not written recently
        if age > self.decay_window {
            return 0.3;
        }

        // Bathtub curve with base-aware initial scaling
        match count {
            0..=2 => (0.5 * self.base_scale).clamp(0.15, 1.0), // New: cautious (base-scaled)
            3..=5 => (0.8 * self.base_scale).clamp(0.3, 1.0),  // Young: absorbing (base-scaled)
            6..=20 => 1.0,                                     // Active: full plasticity (always)
            21..=50 => 0.5,                                    // Consolidating: stabilizing
            51..=200 => 0.2,                                   // Stable: rigid
            _ => 0.1,                                          // Ancient: very rigid
        }
    }

    /// Add a new concept slot.
    pub fn add_concept(&mut self) {
        self.write_counts.push(0);
        self.last_write.push(self.clock);
    }
}

// ---------------------------------------------------------------------------
// Part 5: Absorption Tracker -- Overexposure as Absorption Decay
// ---------------------------------------------------------------------------

/// Overexposure tracking via absorption history.
///
/// Instead of scanning the container, track absorption ratios
/// across recent writes. When the running average drops below
/// threshold, the container is getting full.
pub struct AbsorptionTracker {
    /// Ring buffer of recent absorption ratios.
    history: Vec<f32>,
    /// Write pointer into the ring buffer.
    head: usize,
    /// Number of entries filled.
    count: usize,
}

impl AbsorptionTracker {
    /// Create with a window size (how many recent writes to track).
    pub fn new(window: usize) -> Self {
        Self {
            history: vec![1.0; window],
            head: 0,
            count: 0,
        }
    }

    /// Record an absorption ratio from a write.
    pub fn record(&mut self, absorption: f32) {
        self.history[self.head] = absorption;
        self.head = (self.head + 1) % self.history.len();
        self.count = self.count.saturating_add(1).min(self.history.len());
    }

    /// Running average absorption (1.0 = clean, 0.0 = saturated).
    pub fn average(&self) -> f32 {
        if self.count == 0 {
            return 1.0;
        }
        self.history[..self.count].iter().sum::<f32>() / self.count as f32
    }

    /// Minimum absorption in the window (worst single write).
    pub fn minimum(&self) -> f32 {
        if self.count == 0 {
            return 1.0;
        }
        self.history[..self.count]
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min)
    }

    /// Flush decision based on absorption history.
    pub fn flush_decision(&self) -> crate::flush::FlushAction {
        let avg = self.average();
        let min = self.minimum();

        if min < 0.1 {
            return crate::flush::FlushAction::Emergency;
        }
        if avg < 0.4 {
            return crate::flush::FlushAction::HardFlush;
        }
        if avg < 0.7 {
            return crate::flush::FlushAction::SoftFlush;
        }
        crate::flush::FlushAction::None
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flush::FlushAction;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn seeded_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    /// Local test helper: generate random i8 templates in [-2, 2] range.
    fn make_test_templates(k: usize, d: usize, rng: &mut StdRng) -> Vec<Vec<i8>> {
        (0..k)
            .map(|_| (0..d).map(|_| rng.gen_range(-2i8..=2i8)).collect())
            .collect()
    }

    // --- X-Trans Pattern Tests ---

    #[test]
    fn test_xtrans_positions_per_channel() {
        let pat = XTransPattern::new(2048, 16);
        assert_eq!(pat.positions_per_channel, 128);
        // Every position assigned exactly once
        let total: usize = pat.channel_positions.iter().map(|ch| ch.len()).sum();
        assert_eq!(total, 2048);
    }

    #[test]
    fn test_xtrans_no_adjacent_same_channel() {
        let pat = XTransPattern::new(2048, 16);
        // Min distance between same-channel positions should be >= channels
        // (or at least > 1, meaning no two adjacent positions share a channel)
        let min_dist = pat.min_same_channel_distance();
        assert!(
            min_dist >= 2,
            "min_same_channel_distance = {}, expected >= 2",
            min_dist
        );
    }

    #[test]
    fn test_xtrans_size_uniformity() {
        let pat = XTransPattern::new(2048, 16);
        let uniformity = pat.size_uniformity();
        assert!(
            uniformity < 2.0,
            "size_uniformity = {}, expected < 2.0",
            uniformity
        );
    }

    #[test]
    fn test_xtrans_complete_coverage() {
        let pat = XTransPattern::new(2048, 16);
        // Every position must be assigned to exactly one channel
        let mut covered = vec![false; 2048];
        for ch in 0..16 {
            for &pos in &pat.channel_positions[ch] {
                assert!(
                    !covered[pos],
                    "Position {} assigned to multiple channels",
                    pos
                );
                covered[pos] = true;
            }
        }
        assert!(covered.iter().all(|&c| c), "Not all positions covered");
    }

    #[test]
    fn test_xtrans_golden_step_irregular() {
        let pat = XTransPattern::new(2048, 16);
        // Golden step should NOT be D/C (that would be uniform modulo)
        let uniform_step = 2048 / 16;
        assert_ne!(
            pat.golden_step, uniform_step,
            "Golden step should differ from uniform step"
        );
    }

    // --- Organic Write Tests ---

    #[test]
    fn test_receptivity_values() {
        assert!((receptivity(0) - 1.0).abs() < 1e-6);
        assert!(receptivity(127) < 0.01);
        assert!(receptivity(-127) < 0.01);
        // Partial value
        let r = receptivity(64);
        assert!(r > 0.4 && r < 0.6, "receptivity(64) = {}", r);
    }

    #[test]
    fn test_organic_write_empty_container() {
        let mut container = vec![0i8; 256];
        let template: Vec<i8> = (0..256).map(|i| (i % 5) as i8 - 2).collect();
        let positions: Vec<usize> = (0..128).collect();
        let absorption = organic_write(&mut container, &template, 1.0, &positions);
        assert!(
            absorption > 0.95,
            "Empty container absorption = {}, expected > 0.95",
            absorption
        );
    }

    #[test]
    fn test_organic_write_saturated_container() {
        let mut container = vec![127i8; 256];
        let template: Vec<i8> = vec![4; 256];
        let positions: Vec<usize> = (0..128).collect();
        let absorption = organic_write(&mut container, &template, 1.0, &positions);
        assert!(
            absorption < 0.05,
            "Saturated container absorption = {}, expected < 0.05",
            absorption
        );
    }

    #[test]
    fn test_organic_write_half_full_container() {
        let mut container = vec![64i8; 256];
        let template: Vec<i8> = vec![4; 256];
        let positions: Vec<usize> = (0..128).collect();
        let absorption = organic_write(&mut container, &template, 1.0, &positions);
        assert!(
            absorption > 0.3 && absorption < 0.7,
            "Half-full absorption = {}, expected in (0.3, 0.7)",
            absorption
        );
    }

    #[test]
    fn test_organic_write_channel_isolation() {
        let pat = XTransPattern::new(256, 8);
        let mut container = vec![0i8; 256];
        let template: Vec<i8> = (0..256).map(|i| (i % 9) as i8 - 4).collect();

        // Write to channel 3
        let ch3_positions = &pat.channel_positions[3];
        organic_write(&mut container, &template, 1.0, ch3_positions);

        // Channel 5 positions should be unchanged (all zero)
        for &pos in &pat.channel_positions[5] {
            assert_eq!(
                container[pos], 0,
                "Channel 5 position {} modified by channel 3 write",
                pos
            );
        }
    }

    // --- Organic WAL Tests ---

    #[test]
    fn test_wal_single_concept_recovery() {
        let mut rng = seeded_rng();
        let d = 2048;
        let channels = 16;
        let pattern = XTransPattern::new(d, channels);
        let mut wal = OrganicWAL::new(pattern);
        let mut container = vec![0i8; d];

        let templates = make_test_templates(1, d, &mut rng);
        wal.register_concept(0, templates[0].clone());

        let amplitude = 0.8f32;
        let result = wal.write(&mut container, 0, amplitude, 0.1);

        // Extract should recover a non-zero coefficient.
        // Due to i8 quantization of small signed values (range [-2,2]),
        // rounding amplifies values toward the nearest integer, so the
        // extracted coefficient may differ from amplitude*absorption.
        let extracted = wal.surgical_extract(&container);
        assert!(
            extracted[0].abs() > 0.1,
            "Single concept not recovered (extracted={:.4})",
            extracted[0]
        );
        // Absorption should be near 1.0 for an empty container
        assert!(
            result.absorption > 0.9,
            "Empty container absorption = {:.4}, expected > 0.9",
            result.absorption
        );
    }

    #[test]
    fn test_wal_orthogonal_projections_near_zero() {
        let mut rng = seeded_rng();
        let d = 4096;
        let channels = 16;
        let pattern = XTransPattern::new(d, channels);
        let mut wal = OrganicWAL::new(pattern);
        let mut container = vec![0i8; d];

        // Two concepts on DIFFERENT channels -> structurally orthogonal
        let templates = make_test_templates(2, d, &mut rng);
        wal.register_concept(0, templates[0].clone());
        wal.register_concept(1, templates[1].clone());

        let r0 = wal.write(&mut container, 0, 0.8, 0.1);
        // Concept 1 is on channel 1, concept 0 is on channel 0
        // Projection of concept 1 onto concept 0's template should be small
        // (random templates in high D are approximately orthogonal)
        assert!(
            r0.projections[1].abs() < 0.3,
            "Orthogonal projection = {}, expected < 0.3",
            r0.projections[1]
        );
    }

    #[test]
    fn test_wal_correlated_concepts_positive_projection() {
        let d = 2048;
        let channels = 16;
        let pattern = XTransPattern::new(d, channels);
        let mut wal = OrganicWAL::new(pattern);
        let mut container = vec![0i8; d];

        // Create two correlated templates (same direction, different magnitude)
        let t0: Vec<i8> = (0..d).map(|i| (i % 5) as i8 - 2).collect();
        let t1: Vec<i8> = t0.iter().map(|&v| v.saturating_mul(1)).collect(); // identical

        wal.register_concept(0, t0);
        wal.register_concept(1, t1);

        wal.write(&mut container, 0, 0.5, 0.1);
        let r1 = wal.write(&mut container, 1, 0.5, 0.1);

        // Projection onto concept 0 should be positive (correlated)
        assert!(
            r1.projections[0] > 0.0,
            "Correlated projection = {}, expected > 0.0",
            r1.projections[0]
        );
    }

    #[test]
    fn test_wal_anti_correlated_negative_projection() {
        let d = 2048;
        let channels = 16;
        let pattern = XTransPattern::new(d, channels);
        let mut wal = OrganicWAL::new(pattern);
        let mut container = vec![0i8; d];

        let t0: Vec<i8> = (0..d).map(|i| (i % 5) as i8 - 2).collect();
        let t1: Vec<i8> = t0.iter().map(|&v| v.wrapping_neg()).collect(); // anti-correlated

        wal.register_concept(0, t0);
        wal.register_concept(1, t1);

        wal.write(&mut container, 0, 0.5, 0.1);
        let r1 = wal.write(&mut container, 1, 0.5, 0.1);

        assert!(
            r1.projections[0] < 0.0,
            "Anti-correlated projection = {}, expected < 0.0",
            r1.projections[0]
        );
    }

    #[test]
    fn test_wal_gram_schmidt_residual_orthogonal() {
        let mut rng = seeded_rng();
        let d = 2048;
        let channels = 16;
        let pattern = XTransPattern::new(d, channels);
        let mut wal = OrganicWAL::new(pattern);
        let mut container = vec![0i8; d];

        let templates = make_test_templates(3, d, &mut rng);
        for (i, t) in templates.iter().enumerate() {
            wal.register_concept(i as u32, t.clone());
        }

        // Write concept 0
        wal.write(&mut container, 0, 0.8, 0.1);

        // The orthogonalized residual of concept 1 (after removing its
        // projection onto concept 0) should be approximately orthogonal to concept 0.
        // We verify indirectly: after writing both, surgical_extract should
        // recover both independently.
        wal.write(&mut container, 1, 0.8, 0.1);

        let extracted = wal.surgical_extract(&container);
        // Both coefficients should be non-zero (both concepts are present)
        assert!(
            extracted[0].abs() > 0.01,
            "Concept 0 lost after orthogonalization"
        );
        assert!(
            extracted[1].abs() > 0.01,
            "Concept 1 lost after orthogonalization"
        );
    }

    #[test]
    fn test_wal_16_concepts_recovery() {
        let mut rng = seeded_rng();
        let d = 2048;
        let channels = 16;
        let k = 16;
        let pattern = XTransPattern::new(d, channels);
        let mut wal = OrganicWAL::new(pattern);
        let mut container = vec![0i8; d];

        let templates = make_test_templates(k, d, &mut rng);
        for (i, t) in templates.iter().enumerate() {
            wal.register_concept(i as u32, t.clone());
        }

        for i in 0..k {
            wal.write(&mut container, i, 0.5, 0.1);
        }

        let extracted = wal.surgical_extract(&container);
        // All 16 should be recoverable -- error should be modest
        let mut total_error = 0.0f32;
        for i in 0..k {
            total_error += (wal.coefficients[i] - extracted[i]).abs();
        }
        let mean_error = total_error / k as f32;
        assert!(
            mean_error < 0.3,
            "16-concept mean recovery error = {:.4}, expected < 0.3",
            mean_error
        );
    }

    #[test]
    fn test_wal_hebbian_coefficient_increase() {
        let d = 2048;
        let channels = 16;
        let pattern = XTransPattern::new(d, channels);
        let mut wal = OrganicWAL::new(pattern);
        let mut container = vec![0i8; d];

        // Two correlated concepts
        let t0: Vec<i8> = (0..d).map(|i| (i % 5) as i8 - 2).collect();
        let t1 = t0.clone();

        wal.register_concept(0, t0);
        wal.register_concept(1, t1);

        wal.write(&mut container, 0, 0.5, 0.1);
        let coeff_before = wal.coefficients[0];

        // Writing correlated concept 1 should increase concept 0's coefficient
        wal.write(&mut container, 1, 0.5, 0.1);
        let coeff_after = wal.coefficients[0];

        assert!(
            coeff_after > coeff_before,
            "Hebbian: coeff should increase ({} -> {})",
            coeff_before,
            coeff_after
        );
    }

    #[test]
    fn test_wal_antihebbian_coefficient_decrease() {
        let d = 2048;
        let channels = 16;
        let pattern = XTransPattern::new(d, channels);
        let mut wal = OrganicWAL::new(pattern);
        let mut container = vec![0i8; d];

        let t0: Vec<i8> = (0..d).map(|i| (i % 5) as i8 - 2).collect();
        let t1: Vec<i8> = t0.iter().map(|&v| v.wrapping_neg()).collect();

        wal.register_concept(0, t0);
        wal.register_concept(1, t1);

        wal.write(&mut container, 0, 0.5, 0.1);
        let coeff_before = wal.coefficients[0];

        // Writing anti-correlated concept 1 should decrease concept 0's coefficient
        wal.write(&mut container, 1, 0.5, 0.1);
        let coeff_after = wal.coefficients[0];

        assert!(
            coeff_after < coeff_before,
            "Anti-Hebbian: coeff should decrease ({} -> {})",
            coeff_before,
            coeff_after
        );
    }

    // --- Plasticity Tests ---

    #[test]
    fn test_plasticity_new_concept() {
        // Default (no base scaling): base_scale = 1.0, so 0.5 * 1.0 = 0.5
        let tracker = PlasticityTracker::new(3, 50);
        assert!(
            (tracker.plasticity(0) - 0.5).abs() < 1e-6,
            "New concept plasticity = {}, expected 0.5",
            tracker.plasticity(0)
        );
    }

    #[test]
    fn test_plasticity_active_concept() {
        let mut tracker = PlasticityTracker::new(3, 50);
        for _ in 0..10 {
            tracker.record_write(0);
        }
        assert!(
            (tracker.plasticity(0) - 1.0).abs() < 1e-6,
            "Active concept plasticity = {}, expected 1.0",
            tracker.plasticity(0)
        );
    }

    #[test]
    fn test_plasticity_stable_concept() {
        let mut tracker = PlasticityTracker::new(3, 50);
        for _ in 0..100 {
            tracker.record_write(0);
        }
        assert!(
            (tracker.plasticity(0) - 0.2).abs() < 1e-6,
            "Stable concept plasticity = {}, expected 0.2",
            tracker.plasticity(0)
        );
    }

    #[test]
    fn test_plasticity_decaying_concept() {
        let mut tracker = PlasticityTracker::new(3, 50);
        tracker.record_write(0); // write once
        // Advance clock past decay window by writing to other concepts
        for _ in 0..60 {
            tracker.record_write(1);
        }
        assert!(
            (tracker.plasticity(0) - 0.3).abs() < 1e-6,
            "Decaying concept plasticity = {}, expected 0.3",
            tracker.plasticity(0)
        );
    }

    #[test]
    fn test_plasticity_cardinality_5() {
        // cardinality=5, scale=8/5=1.6 clamped to 1.5
        let tracker = PlasticityTracker::new_with_cardinality(3, 50, 5);
        let p = tracker.plasticity(0);
        let expected = 0.5 * 1.5; // = 0.75
        assert!(
            (p - expected).abs() < 1e-4,
            "Cardinality 5 new plasticity = {}, expected {}",
            p,
            expected
        );
    }

    #[test]
    fn test_plasticity_cardinality_9() {
        // cardinality=9, scale=8/9=0.889
        let tracker = PlasticityTracker::new_with_cardinality(3, 50, 9);
        let p = tracker.plasticity(0);
        let expected = 0.5 * (8.0 / 9.0); // ~ 0.444
        assert!(
            (p - expected).abs() < 1e-3,
            "Cardinality 9 new plasticity = {:.4}, expected {:.4}",
            p,
            expected
        );
    }

    #[test]
    fn test_plasticity_cardinality_active_unaffected() {
        // Active phase (6-20 writes) should always be 1.0 regardless of cardinality
        let mut tracker = PlasticityTracker::new_with_cardinality(3, 50, 5);
        for _ in 0..10 {
            tracker.record_write(0);
        }
        assert!(
            (tracker.plasticity(0) - 1.0).abs() < 1e-6,
            "Active phase should be 1.0 regardless of cardinality, got {}",
            tracker.plasticity(0)
        );
    }

    #[test]
    fn test_plasticity_narrow_less_cautious() {
        // Narrow base should have HIGHER initial plasticity than wide base
        // cardinality 5: 8/5 = 1.6 -> clamped 1.5 -> 0.5 * 1.5 = 0.75
        // cardinality 9: 8/9 = 0.889 -> 0.5 * 0.889 = 0.444
        let narrow = PlasticityTracker::new_with_cardinality(3, 50, 5);
        let wide = PlasticityTracker::new_with_cardinality(3, 50, 9);

        let p_narrow = narrow.plasticity(0); // 0.75
        let p_wide = wide.plasticity(0); // 0.444
        assert!(
            p_narrow > p_wide,
            "Narrow base ({:.4}) should be LESS cautious than wide ({:.4})",
            p_narrow,
            p_wide
        );
    }

    #[test]
    fn test_plasticity_rigid_absorbs_less() {
        let mut rng = seeded_rng();
        let d = 2048;
        let channels = 16;

        // Set up two identical WALs
        let templates = make_test_templates(2, d, &mut rng);

        // WAL 1: concept 0 is plastic (few writes)
        let pattern1 = XTransPattern::new(d, channels);
        let mut wal1 = OrganicWAL::new(pattern1);
        let mut container1 = vec![0i8; d];
        let mut plasticity1 = PlasticityTracker::new(2, 50);
        plasticity1.add_concept();
        plasticity1.add_concept();
        wal1.register_concept(0, templates[0].clone());
        wal1.register_concept(1, templates[1].clone());

        // Write concept 0 a few times -> plasticity 0.5 (new)
        let r1 = wal1.write_plastic(&mut container1, 0, 0.5, 0.1, &mut plasticity1);

        // WAL 2: concept 0 is rigid (many writes)
        let pattern2 = XTransPattern::new(d, channels);
        let mut wal2 = OrganicWAL::new(pattern2);
        let mut container2 = vec![0i8; d];
        let mut plasticity2 = PlasticityTracker::new(2, 50);
        plasticity2.add_concept();
        plasticity2.add_concept();
        wal2.register_concept(0, templates[0].clone());
        wal2.register_concept(1, templates[1].clone());
        // Make concept 0 rigid: 100+ writes
        for _ in 0..100 {
            plasticity2.record_write(0);
        }
        let r2 = wal2.write_plastic(&mut container2, 0, 0.5, 0.1, &mut plasticity2);

        // Rigid concept gets effective_amplitude = 0.5 * 0.2 = 0.1
        // Plastic concept gets effective_amplitude = 0.5 * 0.5 = 0.25
        // Both writing to empty containers, so absorption ~ 1.0 for both.
        // The difference is in the coefficient update.
        assert!(
            wal1.coefficients[0] > wal2.coefficients[0],
            "Plastic concept coefficient ({}) should be > rigid ({})",
            wal1.coefficients[0],
            wal2.coefficients[0],
        );
        // Verify both writes succeeded (just with different effective amplitudes)
        assert!(r1.absorption > 0.9);
        assert!(r2.absorption > 0.9);
    }

    // --- Absorption Tracker Tests ---

    #[test]
    fn test_absorption_tracker_fresh() {
        let tracker = AbsorptionTracker::new(10);
        assert!((tracker.average() - 1.0).abs() < 1e-6);
        assert_eq!(tracker.flush_decision(), FlushAction::None);
    }

    #[test]
    fn test_absorption_tracker_good_writes() {
        let mut tracker = AbsorptionTracker::new(10);
        for _ in 0..10 {
            tracker.record(0.95);
        }
        assert!(tracker.average() > 0.9);
        assert_eq!(tracker.flush_decision(), FlushAction::None);
    }

    #[test]
    fn test_absorption_tracker_hard_flush() {
        let mut tracker = AbsorptionTracker::new(10);
        for _ in 0..10 {
            tracker.record(0.3);
        }
        assert!(tracker.average() < 0.4);
        assert_eq!(tracker.flush_decision(), FlushAction::HardFlush);
    }

    #[test]
    fn test_absorption_tracker_emergency() {
        let mut tracker = AbsorptionTracker::new(10);
        for _ in 0..9 {
            tracker.record(0.95);
        }
        tracker.record(0.05); // single bad write
        assert_eq!(tracker.flush_decision(), FlushAction::Emergency);
    }

    #[test]
    fn test_absorption_tracker_running_mean() {
        let mut tracker = AbsorptionTracker::new(4);
        tracker.record(1.0);
        tracker.record(0.8);
        tracker.record(0.6);
        tracker.record(0.4);
        let avg = tracker.average();
        let expected = (1.0 + 0.8 + 0.6 + 0.4) / 4.0;
        assert!(
            (avg - expected).abs() < 1e-6,
            "Average = {}, expected {}",
            avg,
            expected
        );
    }

    // --- MultiResPattern Test ---

    #[test]
    fn test_multires_pattern_masks() {
        let mr = MultiResPattern::new(256, 8);
        // Each mask should have exactly positions_per_channel true values
        for ch in 0..8 {
            let true_count = mr.warm_channel_masks[ch].iter().filter(|&&v| v).count();
            let expected = mr.warm.channel_positions[ch].len();
            assert_eq!(
                true_count, expected,
                "Channel {} mask has {} trues but {} positions",
                ch, true_count, expected
            );
        }
        // No position should be true in multiple masks
        for j in 0..256 {
            let count: usize = (0..8).map(|ch| mr.warm_channel_masks[ch][j] as usize).sum();
            assert_eq!(count, 1, "Position {} in {} masks, expected 1", j, count);
        }
    }
}
