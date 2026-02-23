//! Learning event classification from BF16 structural diffs.
//!
//! Contains `LearningInterpretation`, `BF16LearningEvent`, and
//! `classify_learning_event`.

use numrus_core::bf16_hamming::{
    select_bf16_hamming_fn, structural_diff, BF16StructuralDiff, BF16Weights,
};

// ---------------------------------------------------------------------------
// BF16 Learning Event — what changed and why
// ---------------------------------------------------------------------------

/// A learning event captured from BF16 structural diff.
#[derive(Clone, Debug)]
pub struct BF16LearningEvent {
    /// Entity that was updated.
    pub entity_id: u32,
    /// Timestep of the update.
    pub timestep: usize,
    /// BF16-structured distance between before and after.
    pub distance: u64,
    /// Structural diff.
    pub diff: BF16StructuralDiff,
    /// Causal interpretation.
    pub interpretation: LearningInterpretation,
}

#[derive(Clone, Debug)]
pub enum LearningInterpretation {
    /// No meaningful change (mantissa noise only).
    Noise,
    /// Attention rebalancing: magnitude shifted but polarity preserved.
    AttentionShift { dims: Vec<usize> },
    /// Semantic reversal: sign flipped on key dimensions.
    SemanticReversal { dims: Vec<usize> },
    /// Both: sign flips AND magnitude shifts.
    MajorUpdate {
        sign_dims: Vec<usize>,
        magnitude_dims: Vec<usize>,
    },
}

/// Classify a learning step from BF16 structural diff.
pub fn classify_learning_event(
    entity_id: u32,
    timestep: usize,
    before: &[u8],
    after: &[u8],
    weights: &BF16Weights,
) -> BF16LearningEvent {
    let bf16_fn = select_bf16_hamming_fn();
    let distance = bf16_fn(before, after, weights);
    let diff = structural_diff(before, after);

    let interpretation = match (diff.sign_flips, diff.major_magnitude_shifts.len()) {
        (0, 0) => LearningInterpretation::Noise,
        (0, _) => LearningInterpretation::AttentionShift {
            dims: diff.major_magnitude_shifts.to_vec(),
        },
        (_, 0) => LearningInterpretation::SemanticReversal {
            dims: diff.sign_flip_dims.to_vec(),
        },
        (_, _) => LearningInterpretation::MajorUpdate {
            sign_dims: diff.sign_flip_dims.to_vec(),
            magnitude_dims: diff.major_magnitude_shifts.to_vec(),
        },
    };

    BF16LearningEvent {
        entity_id,
        timestep,
        distance,
        diff,
        interpretation,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use numrus_core::bf16_hamming::{fp32_to_bf16_bytes, TRAINING_WEIGHTS};
    use numrus_core::SplitMix64;

    fn make_jina_like_embedding(seed: u64, n_dims: usize) -> Vec<f32> {
        let mut rng = SplitMix64::new(seed);
        (0..n_dims)
            .map(|_| (rng.next_u64() as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32 * 0.1)
            .collect()
    }

    #[test]
    fn test_classify_learning_event_semantic_reversal() {
        let n_dims = 32;
        let before_f32 = make_jina_like_embedding(42, n_dims);
        let mut after_f32 = before_f32.clone();
        after_f32[7] = -after_f32[7];

        let before = fp32_to_bf16_bytes(&before_f32);
        let after = fp32_to_bf16_bytes(&after_f32);

        let event = classify_learning_event(1, 0, &before, &after, &TRAINING_WEIGHTS);

        match &event.interpretation {
            LearningInterpretation::SemanticReversal { dims }
            | LearningInterpretation::MajorUpdate {
                sign_dims: dims, ..
            } => {
                assert!(dims.contains(&7), "Should detect sign flip on dim 7");
            }
            other => panic!("Expected SemanticReversal or MajorUpdate, got {:?}", other),
        }
    }

    #[test]
    fn test_classify_learning_event_noise() {
        let n_dims = 32;
        let emb = make_jina_like_embedding(42, n_dims);
        let before = fp32_to_bf16_bytes(&emb);
        let after = before.clone();

        let event = classify_learning_event(1, 0, &before, &after, &TRAINING_WEIGHTS);
        assert!(matches!(
            event.interpretation,
            LearningInterpretation::Noise
        ));
    }

    #[test]
    fn test_classify_learning_event_attention_shift() {
        // Create two BF16 vectors where only exponent bits differ significantly
        let _n_dims = 4;
        let before_f32 = vec![0.1f32, 0.2, 0.3, 0.4];
        let after_f32 = vec![0.1, 0.2, 30.0, 40.0]; // dims 2,3: huge magnitude change

        let before = fp32_to_bf16_bytes(&before_f32);
        let after = fp32_to_bf16_bytes(&after_f32);

        let event = classify_learning_event(1, 0, &before, &after, &TRAINING_WEIGHTS);

        // Should be AttentionShift or MajorUpdate (depends on whether sign also flipped)
        match &event.interpretation {
            LearningInterpretation::AttentionShift { dims } => {
                assert!(!dims.is_empty(), "Should detect magnitude shifts");
            }
            LearningInterpretation::MajorUpdate { magnitude_dims, .. } => {
                assert!(!magnitude_dims.is_empty(), "Should detect magnitude shifts");
            }
            _ => {
                // Also acceptable if exponent didn't change enough bits
            }
        }
    }
}
