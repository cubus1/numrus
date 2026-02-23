//! Organic Flush -- Surgical Cool with Plasticity Preservation
//!
//! Complete organic flush cycle:
//! 1. Extract coefficients via surgical orthogonal projection
//! 2. Blend with running coefficients (80/20)
//! 3. Optionally prune to top-K
//! 4. Clear container and re-materialize survivors

use crate::absorb::{organic_write, OrganicWAL, PlasticityTracker};

/// Flush action levels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FlushAction {
    None,
    SoftFlush,
    HardFlush,
    Emergency,
}

/// Result of an organic flush.
#[derive(Clone, Debug)]
pub struct FlushResult {
    pub coefficients_extracted: Vec<f32>,
    pub concepts_pruned: Vec<u32>,
    pub concepts_rewritten: usize,
    pub average_absorption: f32,
}

/// Complete organic flush cycle.
///
/// 1. Extract coefficients via surgical orthogonal projection
/// 2. Blend with running coefficients (80/20)
/// 3. Optionally prune to top-K
/// 4. Clear container and re-materialize survivors
pub fn organic_flush(
    wal: &mut OrganicWAL,
    container: &mut [i8],
    _plasticity: &PlasticityTracker,
    keep_top_k: Option<usize>,
) -> FlushResult {
    // Step 1: Extract exact coefficients
    let extracted = wal.surgical_extract(container);

    // Step 2: Blend extracted with running coefficients
    for i in 0..wal.k() {
        if i < extracted.len() {
            let blend = 0.8;
            wal.coefficients[i] = blend * extracted[i] + (1.0 - blend) * wal.coefficients[i];
        }
    }

    // Step 3: Optionally prune to top-K
    let pruned = if let Some(top_k) = keep_top_k {
        prune_to_top_k(wal, top_k)
    } else {
        Vec::new()
    };

    // Step 4: Clear the container
    container.fill(0);

    // Step 5: Re-materialize the surviving concepts
    let mut total_absorption = 0.0f32;
    let mut writes = 0usize;
    for i in 0..wal.k() {
        if wal.coefficients[i].abs() < 1e-6 {
            continue;
        }

        let channel = i % wal.pattern.channels;
        let positions = &wal.pattern.channel_positions[channel];

        let template = wal.template(i).to_vec();
        let absorption = organic_write(
            container,
            &template,
            wal.coefficients[i],
            positions,
        );
        total_absorption += absorption;
        writes += 1;
    }

    FlushResult {
        coefficients_extracted: extracted,
        concepts_pruned: pruned,
        concepts_rewritten: writes,
        average_absorption: if writes > 0 {
            total_absorption / writes as f32
        } else {
            1.0
        },
    }
}

/// Prune the WAL to keep only the top-K concepts by |coefficient|.
/// Returns the pruned concept IDs.
fn prune_to_top_k(wal: &mut OrganicWAL, top_k: usize) -> Vec<u32> {
    if wal.k() <= top_k {
        return vec![];
    }

    let mut indexed: Vec<(usize, f32)> = wal
        .coefficients
        .iter()
        .enumerate()
        .map(|(i, &c)| (i, c.abs()))
        .collect();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1));

    let remove_set: Vec<usize> = indexed[top_k..].iter().map(|&(i, _)| i).collect();

    let pruned_ids: Vec<u32> = remove_set.iter().map(|&i| wal.concept_ids[i]).collect();

    // Zero out pruned coefficients
    for &i in &remove_set {
        wal.coefficients[i] = 0.0;
    }

    pruned_ids
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::absorb::{OrganicWAL, PlasticityTracker, XTransPattern};
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

    #[test]
    fn test_flush_clean_container_preserves_coefficients() {
        let mut rng = seeded_rng();
        let d = 2048;
        let channels = 16;
        let k = 4;
        let pattern = XTransPattern::new(d, channels);
        let mut wal = OrganicWAL::new(pattern);
        let mut container = vec![0i8; d];
        let plasticity = PlasticityTracker::new(k, 50);

        let templates = make_test_templates(k, d, &mut rng);
        for (i, t) in templates.iter().enumerate() {
            wal.register_concept(i as u32, t.clone());
        }

        for i in 0..k {
            wal.write(&mut container, i, 0.5, 0.1);
        }

        let coeffs_before: Vec<f32> = wal.coefficients.clone();
        let result = organic_flush(&mut wal, &mut container, &plasticity, None);

        // Coefficients should be close to before (blended 80/20 with extracted)
        for i in 0..k {
            let error = (wal.coefficients[i] - coeffs_before[i]).abs();
            assert!(
                error < 0.5,
                "Flush coefficient drift for concept {}: {:.4}",
                i,
                error
            );
        }
        assert_eq!(result.concepts_rewritten, k);
    }

    #[test]
    fn test_flush_rematerialization_readable() {
        let mut rng = seeded_rng();
        let d = 2048;
        let channels = 16;
        let k = 4;
        let pattern = XTransPattern::new(d, channels);
        let mut wal = OrganicWAL::new(pattern);
        let mut container = vec![0i8; d];
        let plasticity = PlasticityTracker::new(k, 50);

        let templates = make_test_templates(k, d, &mut rng);
        for (i, t) in templates.iter().enumerate() {
            wal.register_concept(i as u32, t.clone());
        }

        for i in 0..k {
            wal.write(&mut container, i, 0.8, 0.1);
        }

        organic_flush(&mut wal, &mut container, &plasticity, None);

        // After flush + re-materialization, all concepts should be readable
        let readbacks = wal.read_all(&container);
        for (id, sim, _amp) in &readbacks {
            assert!(
                *sim > 0.3,
                "Concept {} not readable after flush (sim = {:.4})",
                id,
                sim
            );
        }
    }

    #[test]
    fn test_flush_pruning_removes_weakest() {
        let mut rng = seeded_rng();
        let d = 2048;
        let channels = 16;
        let k = 8;
        let pattern = XTransPattern::new(d, channels);
        let mut wal = OrganicWAL::new(pattern);
        let mut container = vec![0i8; d];
        let plasticity = PlasticityTracker::new(k, 50);

        let templates = make_test_templates(k, d, &mut rng);
        for (i, t) in templates.iter().enumerate() {
            wal.register_concept(i as u32, t.clone());
        }

        // Write with varying amplitudes so some concepts are stronger
        for i in 0..k {
            let amp = if i < 4 { 0.8 } else { 0.1 }; // first 4 strong, last 4 weak
            wal.write(&mut container, i, amp, 0.1);
        }

        let result = organic_flush(&mut wal, &mut container, &plasticity, Some(4));

        // Should have pruned 4 concepts
        assert_eq!(
            result.concepts_pruned.len(),
            4,
            "Expected 4 pruned, got {}",
            result.concepts_pruned.len()
        );

        // Pruned concepts should have zero coefficients
        for &pruned_id in &result.concepts_pruned {
            let idx = wal
                .concept_ids
                .iter()
                .position(|&id| id == pruned_id)
                .unwrap();
            assert!(
                wal.coefficients[idx].abs() < 1e-6,
                "Pruned concept {} still has coefficient {:.4}",
                pruned_id,
                wal.coefficients[idx]
            );
        }
    }
}
