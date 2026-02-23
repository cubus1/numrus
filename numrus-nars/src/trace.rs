//! Causal trace — multi-hop reverse reasoning.
//!
//! Contains `TraceStep`, `CausalTrace`, `reverse_trace` (i8 version),
//! and `BF16TraceStep`, `BF16CausalTrace`, `bf16_reverse_trace` (BF16 version).

use crate::bind::{symbol_distance, Entity, Role, Base, reverse_unbind};
use crate::granger::BF16Entity;
use numrus_core::bf16_hamming::{
    select_bf16_hamming_fn, structural_diff, BF16StructuralDiff, BF16Weights,
};

// ---------------------------------------------------------------------------
// Causal Trace — multi-hop reverse reasoning (i8)
// ---------------------------------------------------------------------------

/// One step in a causal trace: the recovered entity and its confidence.
#[derive(Clone, Debug)]
pub struct TraceStep {
    /// The entity ID that was recovered at this step.
    pub entity_id: u32,
    /// Symbol distance from the unbound candidate to the recovered entity.
    pub distance: u64,
    /// Distance as a fraction of total dimensions (0.0 = identical, 0.5 = random).
    pub normalized_distance: f64,
    /// Whether this step is confident (distance < threshold).
    pub confident: bool,
}

/// Result of a multi-hop reverse causal trace.
#[derive(Clone, Debug)]
pub struct CausalTrace {
    /// The outcome entity we started from.
    pub outcome_id: u32,
    /// The role (verb) used for each unbinding hop.
    pub role_name: String,
    /// Each step in the reverse chain, from outcome back to root cause.
    pub steps: Vec<TraceStep>,
    /// How many steps were confident before the chain broke.
    pub confident_depth: usize,
}

/// Brute-force nearest entity search on i8 vectors.
///
/// Returns (entity_id, hamming_distance) of the closest entity.
/// For production use, replace with CAKES DFS Sieve on a CLAM tree.
fn nearest_entity(candidate: &[i8], entities: &[Entity]) -> (u32, u64) {
    let mut best_id = 0u32;
    let mut best_dist = u64::MAX;

    for e in entities {
        let dist = symbol_distance(candidate, &e.vector);
        if dist < best_dist {
            best_dist = dist;
            best_id = e.id;
        }
    }

    (best_id, best_dist)
}

/// Reverse causal trace: starting from `outcome`, repeatedly unbind with
/// `role` and search for the nearest real entity, up to `max_depth` hops.
///
/// Each hop: `candidate = unbind(current, role)` -> nearest entity.
/// Stops when confidence drops below threshold or max_depth is reached.
///
/// The `confidence_threshold` is a normalized distance: steps with
/// `normalized_distance > confidence_threshold` are marked not confident.
/// A typical value is 0.35 (since 0.5 = random for binary vectors).
pub fn reverse_trace(
    outcome: &Entity,
    role: &Role,
    entities: &[Entity],
    base: Base,
    max_depth: usize,
    confidence_threshold: f64,
) -> CausalTrace {
    let total_dims = outcome.vector.len() as f64;
    let mut steps = Vec::with_capacity(max_depth);
    let mut current = outcome.vector.clone();
    let mut confident_depth = 0;

    for _ in 0..max_depth {
        let candidate = reverse_unbind(&current, role, base);
        let (entity_id, distance) = nearest_entity(&candidate, entities);
        let normalized = distance as f64 / total_dims;
        let confident = normalized < confidence_threshold;

        steps.push(TraceStep {
            entity_id,
            distance,
            normalized_distance: normalized,
            confident,
        });

        if confident {
            confident_depth += 1;
        } else {
            // Chain is broken — stop tracing.
            break;
        }

        // Next hop: use the recovered entity as the new current.
        if let Some(e) = entities.iter().find(|e| e.id == entity_id) {
            current = e.vector.clone();
        } else {
            break;
        }
    }

    CausalTrace {
        outcome_id: outcome.id,
        role_name: role.name.clone(),
        steps,
        confident_depth,
    }
}

// ---------------------------------------------------------------------------
// BF16 Causal Trace — reverse reasoning with per-dim attribution
// ---------------------------------------------------------------------------

/// One step in a BF16 causal trace, with structural diff information.
#[derive(Clone, Debug)]
pub struct BF16TraceStep {
    pub entity_id: u32,
    /// BF16-structured weighted distance.
    pub weighted_distance: u64,
    /// Normalized weighted distance (0.0 = identical).
    pub normalized_distance: f64,
    /// Whether this step is confident.
    pub confident: bool,
    /// Structural diff between unbound candidate and recovered entity.
    pub diff: BF16StructuralDiff,
}

/// BF16 reverse causal trace with per-dimension attribution.
#[derive(Clone, Debug)]
pub struct BF16CausalTrace {
    pub outcome_id: u32,
    pub role_name: String,
    pub steps: Vec<BF16TraceStep>,
    pub confident_depth: usize,
    /// Dimensions that consistently carry causal signal across all hops.
    pub causal_backbone_dims: Vec<usize>,
}

/// BF16 reverse trace: unbind outcome with role, search for nearest
/// BF16 entity, report structural diff at each hop.
pub fn bf16_reverse_trace(
    outcome: &BF16Entity,
    role_bf16: &[u8],
    entities: &[BF16Entity],
    max_depth: usize,
    confidence_threshold: f64,
    weights: &BF16Weights,
) -> BF16CausalTrace {
    let bf16_fn = select_bf16_hamming_fn();

    // For BF16 binary base: unbind = XOR (same as bind)
    let xor_unbind = |bound: &[u8], role: &[u8]| -> Vec<u8> {
        bound.iter().zip(role.iter()).map(|(a, b)| a ^ b).collect()
    };

    // Returns (entity_index, entity_id, distance)
    let nearest_bf16 = |candidate: &[u8]| -> (usize, u32, u64) {
        let mut best_idx = 0usize;
        let mut best_id = 0u32;
        let mut best_dist = u64::MAX;
        for (i, e) in entities.iter().enumerate() {
            let d = bf16_fn(candidate, &e.bf16_bytes, weights);
            if d < best_dist {
                best_dist = d;
                best_id = e.id;
                best_idx = i;
            }
        }
        (best_idx, best_id, best_dist)
    };

    // Max possible BF16 distance for normalization
    let max_dist_per_dim =
        (weights.sign as u64) + 8 * (weights.exponent as u64) + 7 * (weights.mantissa as u64);
    let max_total = max_dist_per_dim * (outcome.n_dims as u64);

    let mut steps = Vec::with_capacity(max_depth);
    // Track current source by index into entities slice (avoids cloning bf16_bytes)
    let mut current_entity_idx: Option<usize> = None;
    let mut confident_depth = 0;
    let mut all_sign_flip_dims: Vec<Vec<usize>> = Vec::new();

    for hop in 0..max_depth {
        let current: &[u8] = if hop == 0 {
            &outcome.bf16_bytes
        } else if let Some(idx) = current_entity_idx {
            &entities[idx].bf16_bytes
        } else {
            break;
        };

        let candidate = xor_unbind(current, role_bf16);
        let (entity_idx, entity_id, distance) = nearest_bf16(&candidate);
        let normalized = if max_total > 0 {
            distance as f64 / max_total as f64
        } else {
            1.0
        };
        let confident = normalized < confidence_threshold;

        // Structural diff between candidate and recovered entity
        let diff = structural_diff(&candidate, &entities[entity_idx].bf16_bytes);

        all_sign_flip_dims.push(diff.sign_flip_dims.to_vec());

        steps.push(BF16TraceStep {
            entity_id,
            weighted_distance: distance,
            normalized_distance: normalized,
            confident,
            diff,
        });

        if confident {
            confident_depth += 1;
            current_entity_idx = Some(entity_idx);
        } else {
            break;
        }
    }

    // Causal backbone: dimensions that appear in sign_flip_dims
    // across most confident hops.
    let causal_backbone_dims = if confident_depth >= 2 {
        let confident_dims: Vec<&Vec<usize>> =
            all_sign_flip_dims[..confident_depth].iter().collect();
        let mut dim_counts = std::collections::HashMap::new();
        for dims in &confident_dims {
            for &d in dims.iter() {
                *dim_counts.entry(d).or_insert(0u32) += 1;
            }
        }
        let threshold = (confident_depth as u32).saturating_sub(1).max(1);
        let mut backbone: Vec<usize> = dim_counts
            .iter()
            .filter(|(_, &count)| count >= threshold)
            .map(|(&dim, _)| dim)
            .collect();
        backbone.sort();
        backbone
    } else {
        Vec::new()
    };

    BF16CausalTrace {
        outcome_id: outcome.id,
        role_name: String::new(),
        steps,
        confident_depth,
        causal_backbone_dims,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bind::{forward_bind, Base};
    use numrus_core::bf16_hamming::fp32_to_bf16_bytes;
    use numrus_core::SplitMix64;

    fn make_random_entity(id: u32, d: usize, rng: &mut SplitMix64, base: Base) -> Entity {
        let vector: Vec<i8> = (0..d)
            .map(|_| rng.gen_range_i8(base.min_val(), base.max_val()))
            .collect();
        Entity {
            id,
            name: format!("entity_{}", id),
            vector,
        }
    }

    fn make_role(name: &str, d: usize, rng: &mut SplitMix64, base: Base) -> Role {
        let vector: Vec<i8> = (0..d)
            .map(|_| rng.gen_range_i8(base.min_val(), base.max_val()))
            .collect();
        Role {
            name: name.to_string(),
            vector,
        }
    }

    fn make_jina_like_embedding(seed: u64, n_dims: usize) -> Vec<f32> {
        let mut rng = SplitMix64::new(seed);
        (0..n_dims)
            .map(|_| (rng.next_u64() as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32 * 0.1)
            .collect()
    }

    #[test]
    fn test_reverse_trace_single_hop() {
        let mut rng = SplitMix64::new(42);
        let base = Base::Binary;
        let d = 1024;

        let cause = make_random_entity(1, d, &mut rng, base);
        let role = make_role("CAUSES", d, &mut rng, base);

        // Effect = bind(cause, CAUSES)
        let effect_vec = forward_bind(&cause.vector, &role, base);
        let effect = Entity {
            id: 2,
            name: "effect".to_string(),
            vector: effect_vec,
        };

        let entities = vec![cause.clone(), effect.clone()];

        let trace = reverse_trace(&effect, &role, &entities, base, 3, 0.4);
        assert!(!trace.steps.is_empty());
        assert_eq!(
            trace.steps[0].entity_id, cause.id,
            "Reverse trace should recover the cause"
        );
        assert_eq!(
            trace.steps[0].distance, 0,
            "Binary unbind should give exact recovery"
        );
        assert!(trace.steps[0].confident);
        assert!(trace.confident_depth >= 1);
    }

    #[test]
    fn test_reverse_trace_chain() {
        let mut rng = SplitMix64::new(42);
        let base = Base::Binary;
        let d = 2048;

        // Build a causal chain: A -> B -> C
        let a = make_random_entity(1, d, &mut rng, base);
        let role = make_role("CAUSES", d, &mut rng, base);

        let b_vec = forward_bind(&a.vector, &role, base);
        let b = Entity {
            id: 2,
            name: "B".to_string(),
            vector: b_vec,
        };

        let c_vec = forward_bind(&b.vector, &role, base);
        let c = Entity {
            id: 3,
            name: "C".to_string(),
            vector: c_vec,
        };

        let entities = vec![a.clone(), b.clone(), c.clone()];

        // Reverse trace from C should recover: C -> B -> A
        let trace = reverse_trace(&c, &role, &entities, base, 5, 0.4);
        assert!(
            trace.confident_depth >= 2,
            "Should trace back at least 2 hops: got {}",
            trace.confident_depth
        );
        assert_eq!(trace.steps[0].entity_id, b.id, "First hop should find B");
        assert_eq!(trace.steps[1].entity_id, a.id, "Second hop should find A");
    }

    #[test]
    fn test_reverse_trace_stops_at_noise() {
        let mut rng = SplitMix64::new(42);
        let base = Base::Binary;
        let d = 2048;

        // Entities: A is a root cause, B = bind(A, CAUSES_1),
        // we trace using a DIFFERENT role (CAUSES_2).
        // Unbinding B with CAUSES_2 gives a random vector (not A).
        let a = make_random_entity(1, d, &mut rng, base);
        let role1 = make_role("CAUSES_1", d, &mut rng, base);
        let role2 = make_role("CAUSES_2", d, &mut rng, base);

        let b_vec = forward_bind(&a.vector, &role1, base);
        let b = Entity {
            id: 2,
            name: "B".to_string(),
            vector: b_vec,
        };

        // Only A and B in the entity set.
        // unbind(B, CAUSES_2) = B XOR CAUSES_2, which is random noise
        // (since B was created with CAUSES_1, not CAUSES_2).
        let entities = vec![a.clone(), b.clone()];

        let trace = reverse_trace(&b, &role2, &entities, base, 5, 0.05);
        // First hop: unbind(B, CAUSES_2) = random -> nearest entity has
        // normalized distance ~0.5, so it should NOT be confident.
        assert_eq!(
            trace.confident_depth, 0,
            "Wrong role should fail on first hop: confident_depth = {}",
            trace.confident_depth
        );
    }

    #[test]
    fn test_bf16_reverse_trace_single_hop() {
        let n_dims = 64;
        let cause = BF16Entity::from_f32(1, "cause", &make_jina_like_embedding(1, n_dims));

        let role_f32 = make_jina_like_embedding(99, n_dims);
        let role_bf16 = fp32_to_bf16_bytes(&role_f32);

        // Effect = XOR(cause, role) in BF16 byte space
        let effect_bytes: Vec<u8> = cause
            .bf16_bytes
            .iter()
            .zip(role_bf16.iter())
            .map(|(a, b)| a ^ b)
            .collect();
        let effect = BF16Entity {
            id: 2,
            name: "effect".into(),
            bf16_bytes: effect_bytes,
            n_dims,
        };

        let entities = vec![cause.clone(), effect.clone()];
        let trace = bf16_reverse_trace(
            &effect,
            &role_bf16,
            &entities,
            3,
            0.3,
            &BF16Weights::default(),
        );

        assert!(!trace.steps.is_empty());
        assert_eq!(trace.steps[0].entity_id, 1, "Should recover cause");
        assert_eq!(
            trace.steps[0].weighted_distance, 0,
            "XOR unbind should be exact"
        );
        assert!(trace.steps[0].confident);
    }
}
