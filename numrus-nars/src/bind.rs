//! Binding algebra for holographic vectors.
//!
//! Contains the `Base` enum, bind/unbind operations, template generation,
//! entity types, and similar pair detection.
//!
//! Extracted from sweep.rs (Base, bind, bind_deep, bundle, generate_template,
//! generate_templates) and nars.rs (unbind, Entity, Role, forward_bind,
//! reverse_unbind, find_similar_pairs, SimilarPair).

use rand::Rng;

// ---------------------------------------------------------------------------
// Parameter constants
// ---------------------------------------------------------------------------

/// Dimensions to test. Total: 7 values.
pub const DIMS: &[usize] = &[1024, 2048, 4096, 8192, 16384, 32768, 65536];

/// All base types. Total: 8 values.
pub const BASES: &[Base] = &[
    // Unsigned
    Base::Binary,      // B=2, {0, 1}
    Base::Unsigned(3), // B=3, {0, 1, 2}
    Base::Unsigned(5), // B=5, {0, 1, 2, 3, 4}
    Base::Unsigned(7), // B=7, {0, 1, 2, 3, 4, 5, 6}
    // Signed (with Auslöschung / cancellation at zero)
    Base::Signed(3), // B=3, {-1, 0, +1}
    Base::Signed(5), // B=5, {-2, -1, 0, +1, +2}
    Base::Signed(7), // B=7, {-3, -2, -1, 0, +1, +2, +3}
    Base::Signed(9), // B=9, {-4, -3, -2, -1, 0, +1, +2, +3, +4}
];

/// Number of axes. Total: 3 values.
pub const AXES: &[usize] = &[1, 2, 3];

/// Bundle sizes (number of concepts K). Total: 9 values.
pub const BUNDLE_SIZES: &[usize] = &[1, 3, 5, 8, 13, 21, 34, 55, 89];

/// Bind depths to test. Total: 4 values.
pub const BIND_DEPTHS: &[usize] = &[1, 2, 3, 4];

// ---------------------------------------------------------------------------
// Base type
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Base {
    Binary,       // 1 bit per dimension, {0, 1}
    Unsigned(u8), // B values per dimension, {0, 1, ..., B-1}
    Signed(u8),   // B values per dimension, {-(B/2), ..., 0, ..., +(B/2)}
}

impl Base {
    /// Number of distinct values per dimension.
    pub fn cardinality(&self) -> u8 {
        match self {
            Base::Binary => 2,
            Base::Unsigned(b) | Base::Signed(b) => *b,
        }
    }

    /// Bits required per dimension.
    pub fn bits_per_dim(&self) -> f32 {
        (self.cardinality() as f32).log2()
    }

    /// Minimum value per dimension.
    pub fn min_val(&self) -> i8 {
        match self {
            Base::Binary => 0,
            Base::Unsigned(_) => 0,
            Base::Signed(b) => -((*b as i8) / 2),
        }
    }

    /// Maximum value per dimension.
    pub fn max_val(&self) -> i8 {
        match self {
            Base::Binary => 1,
            Base::Unsigned(b) => *b as i8 - 1,
            Base::Signed(b) => (*b as i8) / 2,
        }
    }

    /// Does this base support Auslöschung (cancellation at zero)?
    pub fn has_cancellation(&self) -> bool {
        matches!(self, Base::Signed(_))
    }

    /// Total bits per packed storage for D dimensions.
    pub fn storage_bits(&self, d: usize) -> usize {
        (d as f32 * self.bits_per_dim()).ceil() as usize
    }

    /// Storage in bytes for D dimensions x N axes.
    pub fn storage_bytes(&self, d: usize, axes: usize) -> usize {
        (self.storage_bits(d) * axes).div_ceil(8)
    }

    /// String name for CSV output.
    pub fn name(&self) -> String {
        match self {
            Base::Binary => "binary".to_string(),
            Base::Unsigned(b) => format!("unsigned({})", b),
            Base::Signed(b) => format!("signed({})", b),
        }
    }

    /// Whether this base is signed.
    pub fn is_signed(&self) -> bool {
        matches!(self, Base::Signed(_))
    }
}

// ---------------------------------------------------------------------------
// Template generation
// ---------------------------------------------------------------------------

/// Generate a random template vector at the given base.
pub fn generate_template(d: usize, base: Base, rng: &mut impl Rng) -> Vec<i8> {
    let mut template = vec![0i8; d];
    match base {
        Base::Binary => {
            for v in template.iter_mut() {
                *v = rng.gen_range(0..2);
            }
        }
        Base::Unsigned(b) => {
            for v in template.iter_mut() {
                *v = rng.gen_range(0..b as i8);
            }
        }
        Base::Signed(b) => {
            let half = (b / 2) as i8;
            for v in template.iter_mut() {
                *v = rng.gen_range(-half..=half);
            }
        }
    }
    template
}

/// Generate K random templates for one axis.
pub fn generate_templates(k: usize, d: usize, base: Base, rng: &mut impl Rng) -> Vec<Vec<i8>> {
    (0..k).map(|_| generate_template(d, base, rng)).collect()
}

// ---------------------------------------------------------------------------
// Binding
// ---------------------------------------------------------------------------

/// Bind two vectors (role-filler binding).
///
/// Binary: XOR
/// Unsigned: element-wise addition mod B
/// Signed: element-wise addition clamped to range
pub fn bind(a: &[i8], b: &[i8], base: Base) -> Vec<i8> {
    let d = a.len();
    let mut result = vec![0i8; d];
    match base {
        Base::Binary => {
            for i in 0..d {
                result[i] = a[i] ^ b[i];
            }
        }
        Base::Unsigned(bv) => {
            for i in 0..d {
                result[i] = ((a[i] as i16 + b[i] as i16) % bv as i16) as i8;
            }
        }
        Base::Signed(bv) => {
            let half = (bv / 2) as i8;
            for i in 0..d {
                result[i] = (a[i].saturating_add(b[i])).clamp(-half, half);
            }
        }
    }
    result
}

/// Bind at depth N: bind v with N role vectors sequentially.
pub fn bind_deep(v: &[i8], roles: &[Vec<i8>], base: Base) -> Vec<i8> {
    let mut result = v.to_vec();
    for role in roles {
        result = bind(&result, role, base);
    }
    result
}

// ---------------------------------------------------------------------------
// Bundling
// ---------------------------------------------------------------------------

/// Bundle K vectors into one via element-wise addition + threshold.
///
/// The Auslöschung property: for signed bases, opposing values cancel.
pub fn bundle(vectors: &[Vec<i8>], base: Base) -> Vec<i8> {
    let d = vectors[0].len();
    let k = vectors.len();
    let mut accum = vec![0i16; d];

    for v in vectors {
        for i in 0..d {
            accum[i] += v[i] as i16;
        }
    }

    let mut result = vec![0i8; d];
    match base {
        Base::Binary => {
            let threshold = k as i16 / 2;
            for i in 0..d {
                result[i] = if accum[i] > threshold { 1 } else { 0 };
            }
        }
        Base::Unsigned(bv) => {
            for i in 0..d {
                result[i] = (accum[i].max(0).min(bv as i16 - 1)) as i8;
            }
        }
        Base::Signed(bv) => {
            let half = (bv / 2) as i16;
            for i in 0..d {
                result[i] = accum[i].clamp(-half, half) as i8;
            }
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Unbind
// ---------------------------------------------------------------------------

/// Unbind: inverse of `bind`.
///
/// - Binary: exact inverse — XOR is self-inverse, so `unbind == bind`.
/// - Unsigned(B): exact inverse — element-wise subtraction mod B.
/// - Signed(B): **approximate** inverse — negates role, then binds (saturating add +
///   clamp). When the original bind saturated (e.g. bind(3, 2) in Signed(7) clamps
///   to 3), information is destroyed and unbind cannot recover the original value.
///   For values that don't hit the clamp boundary, the inverse is exact.
pub fn unbind(bound: &[i8], role: &[i8], base: Base) -> Vec<i8> {
    let d = bound.len();
    assert_eq!(d, role.len());

    match base {
        Base::Binary => {
            // XOR is its own inverse.
            bind(bound, role, base)
        }
        Base::Unsigned(bv) => {
            // Additive inverse mod B: (bound - role) mod B
            let mut result = vec![0i8; d];
            for i in 0..d {
                let diff = bound[i] as i16 - role[i] as i16;
                result[i] = diff.rem_euclid(bv as i16) as i8;
            }
            result
        }
        Base::Signed(_) => {
            // Negate role, then bind (saturating add + clamp).
            // Widen to i16 to handle i8::MIN correctly: -(-128) = 128 -> clamp to 127.
            let neg_role: Vec<i8> = role
                .iter()
                .map(|&v| (-(v as i16)).clamp(-128, 127) as i8)
                .collect();
            bind(bound, &neg_role, base)
        }
    }
}

// ---------------------------------------------------------------------------
// Bind Space — a named store of holographic entities
// ---------------------------------------------------------------------------

/// An entry in the bind space: named entity with its vector.
#[derive(Clone)]
pub struct Entity {
    pub id: u32,
    pub name: String,
    pub vector: Vec<i8>,
}

/// A named role (verb) vector.
#[derive(Clone)]
pub struct Role {
    pub name: String,
    pub vector: Vec<i8>,
}

/// Forward binding: `bind(subject, role) -> target`.
///
/// Returns the bound vector. To find the actual target, search
/// the bind space for the nearest entity to this result.
pub fn forward_bind(subject: &[i8], role: &Role, base: Base) -> Vec<i8> {
    bind(subject, &role.vector, base)
}

/// Reverse unbind: given an outcome and a role, recover the candidate cause.
///
/// Returns the unbound vector. To find the actual cause, search
/// the bind space for the nearest entity to this result.
pub fn reverse_unbind(outcome: &[i8], role: &Role, base: Base) -> Vec<i8> {
    unbind(outcome, &role.vector, base)
}

// ---------------------------------------------------------------------------
// Symbol distance
// ---------------------------------------------------------------------------

/// Symbol-level Hamming distance on i8 slices.
///
/// Counts positions where `a[i] != b[i]`. This is the correct distance
/// for holographic vectors where each i8 is a discrete symbol (not a byte
/// to be decomposed into bits).
pub(crate) fn symbol_distance(a: &[i8], b: &[i8]) -> u64 {
    assert_eq!(a.len(), b.len());
    let mut dist = 0u64;
    for i in 0..a.len() {
        if a[i] != b[i] {
            dist += 1;
        }
    }
    dist
}

// ---------------------------------------------------------------------------
// SimilarPair Detection
// ---------------------------------------------------------------------------

/// A pair of entities with high structural similarity (low symbol distance).
///
/// This only checks structural proximity — it does NOT verify truth values.
/// To detect actual contradictions, the caller must compare truth values
/// on the returned pairs.
#[derive(Clone, Debug)]
pub struct SimilarPair {
    pub entity_a: u32,
    pub entity_b: u32,
    pub distance: u64,
    pub normalized_distance: f64,
}

/// Find entity pairs that are structurally similar (low symbol distance)
/// within a given radius. For Binary base, random pairs average ~0.5
/// normalized distance; pairs below the threshold are structurally correlated.
///
/// This is O(n^2) brute force. Replace with CAKES rho-NN for O(n*log n).
pub fn find_similar_pairs(entities: &[Entity], radius_threshold: f64) -> Vec<SimilarPair> {
    let mut pairs = Vec::new();
    let n = entities.len();
    if n == 0 {
        return pairs;
    }

    let total_dims = entities[0].vector.len() as f64;

    for i in 0..n {
        for j in (i + 1)..n {
            let dist = symbol_distance(&entities[i].vector, &entities[j].vector);
            let norm = dist as f64 / total_dims;
            if norm < radius_threshold {
                pairs.push(SimilarPair {
                    entity_a: entities[i].id,
                    entity_b: entities[j].id,
                    distance: dist,
                    normalized_distance: norm,
                });
            }
        }
    }

    pairs.sort_by_key(|a| a.distance);
    pairs
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use numrus_core::SplitMix64;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn seeded_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

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

    // -- Base type tests --

    #[test]
    fn test_base_binary_properties() {
        let b = Base::Binary;
        assert_eq!(b.cardinality(), 2);
        assert_eq!(b.min_val(), 0);
        assert_eq!(b.max_val(), 1);
        assert!(!b.has_cancellation());
    }

    #[test]
    fn test_base_signed_properties() {
        let b = Base::Signed(5);
        assert_eq!(b.cardinality(), 5);
        assert_eq!(b.min_val(), -2);
        assert_eq!(b.max_val(), 2);
        assert!(b.has_cancellation());
    }

    #[test]
    fn test_base_unsigned_properties() {
        let b = Base::Unsigned(7);
        assert_eq!(b.cardinality(), 7);
        assert_eq!(b.min_val(), 0);
        assert_eq!(b.max_val(), 6);
        assert!(!b.has_cancellation());
    }

    #[test]
    fn test_base_storage_bytes() {
        // Binary: 1 bit per dim, 1024 dims = 128 bytes per axis
        let bytes = Base::Binary.storage_bytes(1024, 1);
        assert_eq!(bytes, 128); // 1024 bits / 8
    }

    // -- Template generation --

    #[test]
    fn test_generate_template_binary_range() {
        let mut rng = seeded_rng();
        let t = generate_template(1000, Base::Binary, &mut rng);
        assert!(t.iter().all(|&v| v == 0 || v == 1));
    }

    #[test]
    fn test_generate_template_signed_range() {
        let mut rng = seeded_rng();
        let t = generate_template(1000, Base::Signed(5), &mut rng);
        assert!(t.iter().all(|&v| (-2..=2).contains(&v)));
    }

    #[test]
    fn test_generate_template_unsigned_range() {
        let mut rng = seeded_rng();
        let t = generate_template(1000, Base::Unsigned(7), &mut rng);
        assert!(t.iter().all(|&v| (0..=6).contains(&v)));
    }

    // -- Binding --

    #[test]
    fn test_bind_binary_xor() {
        let a = vec![0i8, 1, 1, 0];
        let b = vec![1i8, 1, 0, 0];
        let c = bind(&a, &b, Base::Binary);
        assert_eq!(c, vec![1, 0, 1, 0]);
    }

    #[test]
    fn test_bind_unsigned_mod() {
        let a = vec![2i8, 3, 4];
        let b = vec![4i8, 3, 2];
        let c = bind(&a, &b, Base::Unsigned(5));
        // (2+4)%5=1, (3+3)%5=1, (4+2)%5=1
        assert_eq!(c, vec![1, 1, 1]);
    }

    #[test]
    fn test_bind_signed_clamp() {
        let a = vec![2i8, -2, 1];
        let b = vec![2i8, -2, 0];
        let c = bind(&a, &b, Base::Signed(5));
        // (2+2) clamped to 2, (-2+-2) clamped to -2, (1+0) = 1
        assert_eq!(c, vec![2, -2, 1]);
    }

    #[test]
    fn test_bind_deep_sequential() {
        let mut rng = seeded_rng();
        let v = generate_template(100, Base::Signed(5), &mut rng);
        let roles = generate_templates(3, 100, Base::Signed(5), &mut rng);
        let deep = bind_deep(&v, &roles, Base::Signed(5));
        // Should be different from original
        assert_ne!(deep, v);
        assert_eq!(deep.len(), 100);
    }

    // -- Bundling --

    #[test]
    fn test_bundle_binary_majority() {
        let vecs = vec![
            vec![1i8, 0, 1, 1, 0],
            vec![1, 1, 0, 1, 0],
            vec![0, 1, 1, 1, 0],
        ];
        let b = bundle(&vecs, Base::Binary);
        // majority: [1, 1, 1, 1, 0]
        assert_eq!(b, vec![1, 1, 1, 1, 0]);
    }

    #[test]
    fn test_bundle_signed_cancellation() {
        // Opposing values should cancel (Auslöschung)
        let vecs = vec![vec![2i8, -2, 1], vec![-2, 2, -1]];
        let b = bundle(&vecs, Base::Signed(5));
        // 2+(-2)=0, (-2)+2=0, 1+(-1)=0
        assert_eq!(b, vec![0, 0, 0]);
    }

    // --- Unbind tests ---

    #[test]
    fn test_unbind_binary_self_inverse() {
        let mut rng = SplitMix64::new(42);
        let base = Base::Binary;
        let a: Vec<i8> = (0..256).map(|_| (rng.next_u64() & 1) as i8).collect();
        let role: Vec<i8> = (0..256).map(|_| (rng.next_u64() & 1) as i8).collect();

        let bound = bind(&a, &role, base);
        let recovered = unbind(&bound, &role, base);
        assert_eq!(a, recovered, "Binary unbind must recover original");
    }

    #[test]
    fn test_unbind_unsigned_mod_inverse() {
        let mut rng = SplitMix64::new(42);
        let base = Base::Unsigned(7);
        let a: Vec<i8> = (0..256).map(|_| rng.gen_range_i8(0, 6)).collect();
        let role: Vec<i8> = (0..256).map(|_| rng.gen_range_i8(0, 6)).collect();

        let bound = bind(&a, &role, base);
        let recovered = unbind(&bound, &role, base);
        assert_eq!(a, recovered, "Unsigned unbind must recover original");
    }

    #[test]
    fn test_unbind_signed_approximate_inverse() {
        // Signed bind uses saturating add + clamp, so unbind is approximate
        // when values hit the clamp boundary. Test with small values that
        // won't saturate.
        let base = Base::Signed(7);
        let a: Vec<i8> = vec![0, 1, -1, 2, -2, 0, 1, -1];
        let role: Vec<i8> = vec![0, 0, 0, 0, 0, 1, 1, 1];

        let bound = bind(&a, &role, base);
        let recovered = unbind(&bound, &role, base);
        // For non-saturating values, this should be exact
        assert_eq!(
            a, recovered,
            "Signed unbind should recover when not saturating"
        );
    }

    #[test]
    fn test_unbind_signed_saturation_is_lossy() {
        // Demonstrate that Signed unbind is lossy at the clamp boundary.
        // bind(3, 2) in Signed(7): saturating_add(3,2)=5, clamp(-3,3)=3
        // unbind(3, 2): negate(2)=-2, bind(3,-2)=saturating_add(3,-2)=1
        // Original was 3, recovered is 1 — information was destroyed by clamping.
        let base = Base::Signed(7);
        let a: Vec<i8> = vec![3]; // at the boundary
        let role: Vec<i8> = vec![2]; // pushes past clamp

        let bound = bind(&a, &role, base);
        assert_eq!(bound, vec![3], "bind(3,2) clamps to 3");

        let recovered = unbind(&bound, &role, base);
        assert_ne!(
            recovered, a,
            "Signed unbind MUST be lossy when bind saturated: recovered {:?} vs original {:?}",
            recovered, a
        );
        assert_eq!(recovered, vec![1], "unbind(3,2) = bind(3,-2) = 1 (not 3)");
    }

    #[test]
    fn test_unbind_signed_min_value() {
        // i8::MIN (-128) should negate to 127 (clamped), not stay as -128
        let bound = vec![0i8; 4];
        let role = vec![i8::MIN; 4]; // [-128, -128, -128, -128]
        let result = unbind(&bound, &role, Base::Signed(7));
        // unbind(0, -128) should = bind(0, 127) since neg(-128) clamps to 127
        // bind(0, 127) with Signed(7) = (0 + 127).clamp(-3, 3) = 3
        let expected = bind(&bound, &[127i8; 4], Base::Signed(7));
        assert_eq!(
            result, expected,
            "unbind should negate i8::MIN to 127, not -128"
        );
    }

    // --- Forward / Reverse roundtrip ---

    #[test]
    fn test_forward_reverse_roundtrip_binary() {
        let mut rng = SplitMix64::new(42);
        let base = Base::Binary;
        let d = 1024;

        let subject = make_random_entity(0, d, &mut rng, base);
        let role = make_role("CAUSES", d, &mut rng, base);

        let target_vector = forward_bind(&subject.vector, &role, base);
        let candidate = reverse_unbind(&target_vector, &role, base);

        // Binary: exact recovery
        assert_eq!(candidate, subject.vector);
    }

    // --- SimilarPair detection ---

    #[test]
    fn test_find_similar_pairs_identical() {
        let mut rng = SplitMix64::new(42);
        let base = Base::Binary;
        let d = 256;

        let a = make_random_entity(1, d, &mut rng, base);
        let b = Entity {
            id: 2,
            name: "clone".to_string(),
            vector: a.vector.clone(),
        };

        let pairs = find_similar_pairs(&[a, b], 0.1);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].distance, 0);
    }

    #[test]
    fn test_find_similar_pairs_no_match() {
        let mut rng = SplitMix64::new(42);
        let base = Base::Binary;
        let d = 1024;

        // Random binary vectors in 1024-D: expected Hamming distance ~= 512
        // (normalized ~= 0.5). Threshold 0.3 should find no pairs.
        let entities: Vec<Entity> = (0..10)
            .map(|i| make_random_entity(i, d, &mut rng, base))
            .collect();

        let pairs = find_similar_pairs(&entities, 0.3);
        assert!(
            pairs.is_empty(),
            "Random 1024-D binary vectors should not be within 0.3: found {} pairs",
            pairs.len()
        );
    }
}
