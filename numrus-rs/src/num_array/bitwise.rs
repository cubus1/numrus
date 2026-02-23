//! Bitwise operations for NumArray integer types.
//!
//! Provides AVX-512 accelerated `BitAnd`, `BitXor`, `BitOr`, `Not` operators
//! and dedicated methods for bitpacked hamming distance computation.
//! Operations are implemented for `u8`, `i32`, and `i64` element types.

use super::{NumArrayI32, NumArrayI64, NumArrayU8};
use crate::simd_ops::{BitwiseSimdOps, HammingSimdOps};

use std::ops::{BitAnd, BitOr, BitXor, Not};

// ===========================================================================
// Macro to implement bitwise operators for a concrete NumArray type.
// Each operator delegates to BitwiseSimdOps for AVX-512 acceleration.
// ===========================================================================

macro_rules! impl_bitwise_ops {
    ($array_type:ty, $elem:ty, $simd_type:ty) => {
        // ---- BitAnd: array & array ----
        impl BitAnd for $array_type {
            type Output = $array_type;
            #[inline]
            fn bitand(self, rhs: Self) -> Self::Output {
                assert_eq!(self.shape, rhs.shape, "Shapes must match for bitwise AND");
                let mut out = vec![<$elem>::default(); self.data.len()];
                <$simd_type as BitwiseSimdOps<$elem>>::bitwise_and(&self.data, &rhs.data, &mut out);
                <$array_type>::new_with_shape(out, self.shape.clone())
            }
        }

        impl<'a, 'b> BitAnd<&'b $array_type> for &'a $array_type {
            type Output = $array_type;
            #[inline]
            fn bitand(self, rhs: &'b $array_type) -> Self::Output {
                assert_eq!(self.shape, rhs.shape, "Shapes must match for bitwise AND");
                let mut out = vec![<$elem>::default(); self.data.len()];
                <$simd_type as BitwiseSimdOps<$elem>>::bitwise_and(&self.data, &rhs.data, &mut out);
                <$array_type>::new_with_shape(out, self.shape.clone())
            }
        }

        // ---- BitAnd: array & scalar ----
        impl BitAnd<$elem> for $array_type {
            type Output = $array_type;
            #[inline]
            fn bitand(self, rhs: $elem) -> Self::Output {
                let mut out = vec![<$elem>::default(); self.data.len()];
                <$simd_type as BitwiseSimdOps<$elem>>::bitwise_and_scalar(
                    &self.data, rhs, &mut out,
                );
                <$array_type>::new_with_shape(out, self.shape.clone())
            }
        }

        impl<'a> BitAnd<$elem> for &'a $array_type {
            type Output = $array_type;
            #[inline]
            fn bitand(self, rhs: $elem) -> Self::Output {
                let mut out = vec![<$elem>::default(); self.data.len()];
                <$simd_type as BitwiseSimdOps<$elem>>::bitwise_and_scalar(
                    &self.data, rhs, &mut out,
                );
                <$array_type>::new_with_shape(out, self.shape.clone())
            }
        }

        // ---- BitXor: array ^ array ----
        impl BitXor for $array_type {
            type Output = $array_type;
            #[inline]
            fn bitxor(self, rhs: Self) -> Self::Output {
                assert_eq!(self.shape, rhs.shape, "Shapes must match for bitwise XOR");
                let mut out = vec![<$elem>::default(); self.data.len()];
                <$simd_type as BitwiseSimdOps<$elem>>::bitwise_xor(&self.data, &rhs.data, &mut out);
                <$array_type>::new_with_shape(out, self.shape.clone())
            }
        }

        impl<'a, 'b> BitXor<&'b $array_type> for &'a $array_type {
            type Output = $array_type;
            #[inline]
            fn bitxor(self, rhs: &'b $array_type) -> Self::Output {
                assert_eq!(self.shape, rhs.shape, "Shapes must match for bitwise XOR");
                let mut out = vec![<$elem>::default(); self.data.len()];
                <$simd_type as BitwiseSimdOps<$elem>>::bitwise_xor(&self.data, &rhs.data, &mut out);
                <$array_type>::new_with_shape(out, self.shape.clone())
            }
        }

        // ---- BitXor: array ^ scalar ----
        impl BitXor<$elem> for $array_type {
            type Output = $array_type;
            #[inline]
            fn bitxor(self, rhs: $elem) -> Self::Output {
                let mut out = vec![<$elem>::default(); self.data.len()];
                <$simd_type as BitwiseSimdOps<$elem>>::bitwise_xor_scalar(
                    &self.data, rhs, &mut out,
                );
                <$array_type>::new_with_shape(out, self.shape.clone())
            }
        }

        impl<'a> BitXor<$elem> for &'a $array_type {
            type Output = $array_type;
            #[inline]
            fn bitxor(self, rhs: $elem) -> Self::Output {
                let mut out = vec![<$elem>::default(); self.data.len()];
                <$simd_type as BitwiseSimdOps<$elem>>::bitwise_xor_scalar(
                    &self.data, rhs, &mut out,
                );
                <$array_type>::new_with_shape(out, self.shape.clone())
            }
        }

        // ---- BitOr: array | array ----
        impl BitOr for $array_type {
            type Output = $array_type;
            #[inline]
            fn bitor(self, rhs: Self) -> Self::Output {
                assert_eq!(self.shape, rhs.shape, "Shapes must match for bitwise OR");
                let mut out = vec![<$elem>::default(); self.data.len()];
                <$simd_type as BitwiseSimdOps<$elem>>::bitwise_or(&self.data, &rhs.data, &mut out);
                <$array_type>::new_with_shape(out, self.shape.clone())
            }
        }

        impl<'a, 'b> BitOr<&'b $array_type> for &'a $array_type {
            type Output = $array_type;
            #[inline]
            fn bitor(self, rhs: &'b $array_type) -> Self::Output {
                assert_eq!(self.shape, rhs.shape, "Shapes must match for bitwise OR");
                let mut out = vec![<$elem>::default(); self.data.len()];
                <$simd_type as BitwiseSimdOps<$elem>>::bitwise_or(&self.data, &rhs.data, &mut out);
                <$array_type>::new_with_shape(out, self.shape.clone())
            }
        }

        // ---- BitOr: array | scalar ----
        impl BitOr<$elem> for $array_type {
            type Output = $array_type;
            #[inline]
            fn bitor(self, rhs: $elem) -> Self::Output {
                let mut out = vec![<$elem>::default(); self.data.len()];
                <$simd_type as BitwiseSimdOps<$elem>>::bitwise_or_scalar(&self.data, rhs, &mut out);
                <$array_type>::new_with_shape(out, self.shape.clone())
            }
        }

        impl<'a> BitOr<$elem> for &'a $array_type {
            type Output = $array_type;
            #[inline]
            fn bitor(self, rhs: $elem) -> Self::Output {
                let mut out = vec![<$elem>::default(); self.data.len()];
                <$simd_type as BitwiseSimdOps<$elem>>::bitwise_or_scalar(&self.data, rhs, &mut out);
                <$array_type>::new_with_shape(out, self.shape.clone())
            }
        }

        // ---- Not: !array ----
        impl Not for $array_type {
            type Output = $array_type;
            #[inline]
            fn not(self) -> Self::Output {
                let mut out = vec![<$elem>::default(); self.data.len()];
                <$simd_type as BitwiseSimdOps<$elem>>::bitwise_not(&self.data, &mut out);
                <$array_type>::new_with_shape(out, self.shape.clone())
            }
        }

        impl<'a> Not for &'a $array_type {
            type Output = $array_type;
            #[inline]
            fn not(self) -> Self::Output {
                let mut out = vec![<$elem>::default(); self.data.len()];
                <$simd_type as BitwiseSimdOps<$elem>>::bitwise_not(&self.data, &mut out);
                <$array_type>::new_with_shape(out, self.shape.clone())
            }
        }
    };
}

impl_bitwise_ops!(NumArrayU8, u8, std::simd::u8x64);
impl_bitwise_ops!(NumArrayI32, i32, std::simd::i32x16);
impl_bitwise_ops!(NumArrayI64, i64, std::simd::i64x8);

// ===========================================================================
// Hamming distance methods on NumArrayU8
// ===========================================================================

impl NumArrayU8 {
    /// Compute the bitpacked hamming distance between two u8 arrays.
    ///
    /// Returns the number of bits that differ between `self` and `other`.
    /// Optimized for arrays whose length is a multiple of 8192 bytes.
    ///
    /// This fuses XOR + popcount into a single pass for maximum throughput:
    /// each AVX-512 iteration processes 512 bits with 4× unrolling.
    ///
    /// # Panics
    /// Panics if the arrays have different lengths.
    ///
    /// # Example
    /// ```
    /// use numrus_rs::NumArrayU8;
    /// let a = NumArrayU8::new(vec![0xAA; 8192]);
    /// let b = NumArrayU8::new(vec![0x55; 8192]);
    /// assert_eq!(a.hamming_distance(&b), 8192 * 8);
    /// ```
    #[inline]
    pub fn hamming_distance(&self, other: &Self) -> u64 {
        assert_eq!(
            self.data.len(),
            other.data.len(),
            "Arrays must have the same length for hamming distance"
        );
        <std::simd::u8x64 as HammingSimdOps>::hamming_distance(&self.data, &other.data)
    }

    /// Count the total number of set bits (popcount) in the array.
    ///
    /// # Example
    /// ```
    /// use numrus_rs::NumArrayU8;
    /// let a = NumArrayU8::new(vec![0xFF; 8192]);
    /// assert_eq!(a.popcount(), 8192 * 8);
    /// ```
    #[inline]
    pub fn popcount(&self) -> u64 {
        if self.data.is_empty() {
            return 0;
        }
        <std::simd::u8x64 as HammingSimdOps>::popcount(&self.data)
    }

    /// Batch hamming distance: compute hamming distances between corresponding
    /// pairs of bitpacked vectors stored contiguously.
    ///
    /// `self` and `other` each contain `count` vectors of `vec_len` bytes,
    /// stored end-to-end. Returns `count` hamming distances.
    ///
    /// Parallelizes across pairs when count >= 16.
    ///
    /// # Panics
    /// Panics if array lengths don't match `vec_len * count`.
    ///
    /// # Example
    /// ```
    /// use numrus_rs::NumArrayU8;
    /// let vec_len = 8192;
    /// let count = 4;
    /// let a = NumArrayU8::new(vec![0xAA; vec_len * count]);
    /// let b = NumArrayU8::new(vec![0x55; vec_len * count]);
    /// let distances = a.hamming_distance_batch(&b, vec_len, count);
    /// assert_eq!(distances.len(), 4);
    /// assert!(distances.iter().all(|&d| d == vec_len as u64 * 8));
    /// ```
    #[inline]
    pub fn hamming_distance_batch(&self, other: &Self, vec_len: usize, count: usize) -> Vec<u64> {
        assert_eq!(
            self.data.len(),
            vec_len * count,
            "self length must be vec_len * count"
        );
        assert_eq!(
            other.data.len(),
            vec_len * count,
            "other length must be vec_len * count"
        );
        <std::simd::u8x64 as HammingSimdOps>::hamming_distance_batch(
            &self.data,
            &other.data,
            vec_len,
            count,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- NumArrayU8 bitwise operator tests ----

    #[test]
    fn test_u8_bitand_operator() {
        let a = NumArrayU8::new(vec![0xFF, 0x0F, 0xF0, 0xAA]);
        let b = NumArrayU8::new(vec![0x0F, 0xFF, 0x0F, 0x55]);
        let result = &a & &b;
        assert_eq!(result.get_data(), &[0x0F, 0x0F, 0x00, 0x00]);
    }

    #[test]
    fn test_u8_bitxor_operator() {
        let a = NumArrayU8::new(vec![0xFF, 0x0F, 0xAA]);
        let b = NumArrayU8::new(vec![0x0F, 0xFF, 0x55]);
        let result = &a ^ &b;
        assert_eq!(result.get_data(), &[0xF0, 0xF0, 0xFF]);
    }

    #[test]
    fn test_u8_bitor_operator() {
        let a = NumArrayU8::new(vec![0xF0, 0x0F, 0xA0]);
        let b = NumArrayU8::new(vec![0x0F, 0xF0, 0x05]);
        let result = &a | &b;
        assert_eq!(result.get_data(), &[0xFF, 0xFF, 0xA5]);
    }

    #[test]
    fn test_u8_not_operator() {
        let a = NumArrayU8::new(vec![0x00, 0xFF, 0x0F, 0xF0]);
        let result = !&a;
        assert_eq!(result.get_data(), &[0xFF, 0x00, 0xF0, 0x0F]);
    }

    #[test]
    fn test_u8_bitand_scalar() {
        let a = NumArrayU8::new(vec![0xFF, 0x0F, 0xF0, 0xAA]);
        let result = &a & 0x0Fu8;
        assert_eq!(result.get_data(), &[0x0F, 0x0F, 0x00, 0x0A]);
    }

    #[test]
    fn test_u8_bitxor_scalar() {
        let a = NumArrayU8::new(vec![0xFF, 0x0F, 0xF0]);
        let result = &a ^ 0xFFu8;
        assert_eq!(result.get_data(), &[0x00, 0xF0, 0x0F]);
    }

    // ---- NumArrayU8 hamming distance tests ----

    #[test]
    fn test_hamming_distance_method() {
        let a = NumArrayU8::new(vec![0xAA; 8192]);
        let b = NumArrayU8::new(vec![0x55; 8192]);
        assert_eq!(a.hamming_distance(&b), 8192 * 8);
    }

    #[test]
    fn test_hamming_distance_identical() {
        let a = NumArrayU8::new(vec![0x42; 8192]);
        let b = NumArrayU8::new(vec![0x42; 8192]);
        assert_eq!(a.hamming_distance(&b), 0);
    }

    #[test]
    fn test_popcount_method() {
        let a = NumArrayU8::new(vec![0xFF; 8192]);
        assert_eq!(a.popcount(), 8192 * 8);
    }

    #[test]
    fn test_popcount_empty() {
        let a = NumArrayU8::new(vec![]);
        assert_eq!(a.popcount(), 0);
    }

    #[test]
    fn test_hamming_batch() {
        let vec_len = 8192;
        let count = 8;
        let a = NumArrayU8::new(vec![0xAA; vec_len * count]);
        let mut b_data = vec![0xAA; vec_len * count];
        // Make vectors 0, 2, 4, 6 identical, vectors 1, 3, 5, 7 all different
        for i in 0..count {
            if i % 2 == 1 {
                for j in 0..vec_len {
                    b_data[i * vec_len + j] = 0x55;
                }
            }
        }
        let b = NumArrayU8::new(b_data);
        let results = a.hamming_distance_batch(&b, vec_len, count);
        assert_eq!(results.len(), count);
        for i in 0..count {
            if i % 2 == 0 {
                assert_eq!(results[i], 0, "vector {} should match", i);
            } else {
                assert_eq!(results[i], vec_len as u64 * 8, "vector {} should differ", i);
            }
        }
    }

    // ---- NumArrayI32 bitwise operator tests ----

    #[test]
    fn test_i32_bitand_operator() {
        let a = NumArrayI32::new(vec![0x0F0F0F0F, -1, 0, 0x12345678]);
        let b = NumArrayI32::new(vec![0x00FF00FF, 0x0F0F0F0F, -1, 0x0000FFFF]);
        let result = &a & &b;
        for i in 0..4 {
            assert_eq!(result.get_data()[i], a.get_data()[i] & b.get_data()[i]);
        }
    }

    #[test]
    fn test_i32_bitxor_operator() {
        let a = NumArrayI32::new(vec![0, -1, 0x12345678]);
        let b = NumArrayI32::new(vec![0, -1, 0x12345678]);
        let result = &a ^ &b;
        assert_eq!(result.get_data(), &[0, 0, 0]);
    }

    #[test]
    fn test_i32_not_operator() {
        let a = NumArrayI32::new(vec![0, -1, 1]);
        let result = !a;
        assert_eq!(result.get_data(), &[-1, 0, -2]);
    }

    // ---- NumArrayI64 bitwise operator tests ----

    #[test]
    fn test_i64_bitand_operator() {
        let a = NumArrayI64::new(vec![0x0F0F0F0F0F0F0F0F, -1, 0]);
        let b = NumArrayI64::new(vec![0x00FF00FF00FF00FF, 0x0F0F0F0F0F0F0F0F, -1]);
        let result = &a & &b;
        for i in 0..3 {
            assert_eq!(result.get_data()[i], a.get_data()[i] & b.get_data()[i]);
        }
    }

    // ---- Large array tests (8192 elements) ----

    #[test]
    fn test_u8_bitwise_large_8192() {
        let n = 8192;
        let a = NumArrayU8::new((0..n).map(|i| (i % 256) as u8).collect());
        let b = NumArrayU8::new((0..n).map(|i| ((i * 7) % 256) as u8).collect());

        let and_result = &a & &b;
        let xor_result = &a ^ &b;
        let or_result = &a | &b;

        for i in 0..n {
            let av = (i % 256) as u8;
            let bv = ((i * 7) % 256) as u8;
            assert_eq!(and_result.get_data()[i], av & bv, "AND mismatch at {}", i);
            assert_eq!(xor_result.get_data()[i], av ^ bv, "XOR mismatch at {}", i);
            assert_eq!(or_result.get_data()[i], av | bv, "OR mismatch at {}", i);
        }
    }

    #[test]
    fn test_i32_bitwise_large_8192() {
        let n = 8192;
        let a = NumArrayI32::new((0..n).map(|i| i as i32).collect());
        let b = NumArrayI32::new((0..n).map(|i| (i * 3) as i32).collect());

        let and_result = &a & &b;
        let xor_result = &a ^ &b;

        for i in 0..n {
            assert_eq!(and_result.get_data()[i], (i as i32) & (i as i32 * 3));
            assert_eq!(xor_result.get_data()[i], (i as i32) ^ (i as i32 * 3));
        }
    }

    // ---- 2D shape preservation tests ----

    #[test]
    fn test_bitwise_preserves_shape() {
        let a = NumArrayU8::new_with_shape(vec![0xFF; 6], vec![2, 3]);
        let b = NumArrayU8::new_with_shape(vec![0x0F; 6], vec![2, 3]);
        let result = &a & &b;
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result.get_data(), &[0x0F; 6]);
    }
}
