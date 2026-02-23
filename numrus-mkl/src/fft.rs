//! Fast Fourier Transform (FFT) — pure Rust, SIMD-accelerated.
//!
//! Implements radix-2 Cooley-Tukey FFT with butterfly operations
//! using AVX-512 SIMD where applicable. Operates directly on
//! blackboard buffers — zero serialization.
//!
//! ## Supported transforms
//!
//! - `fft_f32` / `fft_f64`: Forward complex FFT (in-place)
//! - `ifft_f32` / `ifft_f64`: Inverse complex FFT (in-place)
//! - `rfft_f32`: Real-to-complex FFT
//!
//! Complex numbers are stored as interleaved (re, im, re, im, ...).
//!
//! ## SIMD Strategy
//!
//! - Twiddle factors are pre-computed once per butterfly stage (not per k-group).
//! - For later stages (half >= F32_LANES/2 for f32, half >= F64_LANES/2 for f64),
//!   butterfly pairs are processed in SIMD-width batches using complex multiply:
//!   twiddle = wr*odd + wi_sign*odd_swapped
//!   where odd_swapped has re/im swapped and wi_sign alternates [-sin, +sin].
//! - ifft conjugate and scale loops use SIMD sign-mask and broadcast multiply.

use std::simd::f32x16;
use std::simd::f64x8;

use numrus_core::simd::{F32_LANES, F64_LANES};

/// Number of complex values processed per f32 SIMD iteration.
/// Each complex number occupies 2 f32 lanes (re, im).
const F32_COMPLEX_PER_SIMD: usize = F32_LANES / 2; // 8

/// Number of complex values processed per f64 SIMD iteration.
const F64_COMPLEX_PER_SIMD: usize = F64_LANES / 2; // 4

// ============================================================================
// Complex FFT (radix-2 Cooley-Tukey, in-place, decimation-in-time)
// ============================================================================

/// In-place radix-2 FFT on interleaved complex f32 data.
///
/// `data` has length `2 * n` where `n` is the FFT size (must be power of 2).
/// Elements are stored as [re0, im0, re1, im1, ...].
///
/// # Panics
/// Panics if n is not a power of 2.
pub fn fft_f32(data: &mut [f32], n: usize) {
    #[cfg(feature = "mkl")]
    {
        use numrus_core::mkl_ffi::*;
        unsafe {
            let mut handle: DftiDescriptorHandle = std::ptr::null_mut();
            let status = DftiCreateDescriptor(&mut handle, DFTI_SINGLE, DFTI_COMPLEX, 1, n as i64);
            if status == 0 {
                DftiSetValue(handle, DFTI_PLACEMENT, DFTI_INPLACE);
                DftiCommitDescriptor(handle);
                DftiComputeForward(handle, data.as_mut_ptr() as *mut std::os::raw::c_void);
                DftiFreeDescriptor(&mut handle);
            }
        }
        return;
    }
    assert!(n.is_power_of_two(), "FFT size must be a power of 2");
    assert_eq!(data.len(), 2 * n);

    // Bit-reversal permutation
    bit_reverse_permute_f32(data, n);

    // Butterfly stages
    let mut stage_len = 2;
    while stage_len <= n {
        let half = stage_len / 2;
        let angle = -2.0 * std::f32::consts::PI / stage_len as f32;

        // Pre-compute twiddle factors once per stage
        let mut cos_tbl = vec![0.0f32; half];
        let mut sin_tbl = vec![0.0f32; half];
        for j in 0..half {
            let theta = angle * j as f32;
            cos_tbl[j] = theta.cos();
            sin_tbl[j] = theta.sin();
        }

        if half >= F32_COMPLEX_PER_SIMD {
            // SIMD butterfly path — process F32_COMPLEX_PER_SIMD j-values at once.
            //
            // Pre-build interleaved twiddle vectors for each SIMD chunk of j.
            // For complex multiply on interleaved [re, im, re, im, ...] data:
            //   wr_dup  = [cos_j, cos_j, cos_{j+1}, cos_{j+1}, ...]
            //   wi_sign = [-sin_j, sin_j, -sin_{j+1}, sin_{j+1}, ...]
            // Then: twiddle = wr_dup * odd + wi_sign * odd_swapped
            // where odd_swapped has re<->im swapped within each complex pair.
            let num_simd_chunks = half / F32_COMPLEX_PER_SIMD;
            let mut wr_vecs: Vec<f32x16> = Vec::with_capacity(num_simd_chunks);
            let mut wi_vecs: Vec<f32x16> = Vec::with_capacity(num_simd_chunks);

            for chunk in 0..num_simd_chunks {
                let base_j = chunk * F32_COMPLEX_PER_SIMD;
                let mut wr_arr = [0.0f32; F32_LANES];
                let mut wi_arr = [0.0f32; F32_LANES];
                for c in 0..F32_COMPLEX_PER_SIMD {
                    let cj = cos_tbl[base_j + c];
                    let sj = sin_tbl[base_j + c];
                    wr_arr[2 * c] = cj;
                    wr_arr[2 * c + 1] = cj;
                    wi_arr[2 * c] = -sj; // -sin for real part
                    wi_arr[2 * c + 1] = sj; // +sin for imaginary part
                }
                wr_vecs.push(f32x16::from_array(wr_arr));
                wi_vecs.push(f32x16::from_array(wi_arr));
            }

            let simd_end = num_simd_chunks * F32_COMPLEX_PER_SIMD;

            for k in (0..n).step_by(stage_len) {
                // SIMD portion
                for chunk in 0..num_simd_chunks {
                    let j = chunk * F32_COMPLEX_PER_SIMD;
                    let even_base = 2 * (k + j);
                    let odd_base = 2 * (k + j + half);

                    let even_v = f32x16::from_slice(&data[even_base..]);
                    let odd_v = f32x16::from_slice(&data[odd_base..]);

                    // Build odd_swapped: swap re<->im within each complex pair.
                    // odd = [re0, im0, re1, im1, ...] -> [im0, re0, im1, re1, ...]
                    let odd_arr = odd_v.to_array();
                    let mut swapped_arr = [0.0f32; F32_LANES];
                    let mut ci = 0;
                    while ci < F32_LANES {
                        swapped_arr[ci] = odd_arr[ci + 1];
                        swapped_arr[ci + 1] = odd_arr[ci];
                        ci += 2;
                    }
                    let odd_swapped = f32x16::from_array(swapped_arr);

                    // Complex twiddle multiply:
                    //   result[2c]   = cos * odd_re - sin * odd_im
                    //   result[2c+1] = cos * odd_im + sin * odd_re
                    // = wr_dup * odd + wi_sign * odd_swapped
                    let twiddle = wr_vecs[chunk] * odd_v + wi_vecs[chunk] * odd_swapped;

                    let result_even = even_v + twiddle;
                    let result_odd = even_v - twiddle;

                    result_even.copy_to_slice(&mut data[even_base..even_base + F32_LANES]);
                    result_odd.copy_to_slice(&mut data[odd_base..odd_base + F32_LANES]);
                }

                // Scalar tail for remaining j values
                for j in simd_end..half {
                    let wr = cos_tbl[j];
                    let wi = sin_tbl[j];

                    let even_re = data[2 * (k + j)];
                    let even_im = data[2 * (k + j) + 1];
                    let odd_re = data[2 * (k + j + half)];
                    let odd_im = data[2 * (k + j + half) + 1];

                    let tr = wr * odd_re - wi * odd_im;
                    let ti = wr * odd_im + wi * odd_re;

                    data[2 * (k + j)] = even_re + tr;
                    data[2 * (k + j) + 1] = even_im + ti;
                    data[2 * (k + j + half)] = even_re - tr;
                    data[2 * (k + j + half) + 1] = even_im - ti;
                }
            }
        } else {
            // Scalar butterfly for early stages (half < SIMD width)
            for k in (0..n).step_by(stage_len) {
                for j in 0..half {
                    let wr = cos_tbl[j];
                    let wi = sin_tbl[j];

                    let even_re = data[2 * (k + j)];
                    let even_im = data[2 * (k + j) + 1];
                    let odd_re = data[2 * (k + j + half)];
                    let odd_im = data[2 * (k + j + half) + 1];

                    let tr = wr * odd_re - wi * odd_im;
                    let ti = wr * odd_im + wi * odd_re;

                    data[2 * (k + j)] = even_re + tr;
                    data[2 * (k + j) + 1] = even_im + ti;
                    data[2 * (k + j + half)] = even_re - tr;
                    data[2 * (k + j + half) + 1] = even_im - ti;
                }
            }
        }

        stage_len *= 2;
    }
}

/// In-place inverse FFT on interleaved complex f32 data.
///
/// Conjugates, applies forward FFT, conjugates again, and scales by 1/n.
pub fn ifft_f32(data: &mut [f32], n: usize) {
    #[cfg(feature = "mkl")]
    {
        use numrus_core::mkl_ffi::*;
        unsafe {
            let mut handle: DftiDescriptorHandle = std::ptr::null_mut();
            let status = DftiCreateDescriptor(&mut handle, DFTI_SINGLE, DFTI_COMPLEX, 1, n as i64);
            if status == 0 {
                let scale = 1.0f32 / n as f32;
                DftiSetValue(handle, DFTI_BACKWARD_SCALE, scale);
                DftiSetValue(handle, DFTI_PLACEMENT, DFTI_INPLACE);
                DftiCommitDescriptor(handle);
                DftiComputeBackward(handle, data.as_mut_ptr() as *mut std::os::raw::c_void);
                DftiFreeDescriptor(&mut handle);
            }
        }
        return;
    }
    let len = 2 * n;

    // SIMD conjugate: negate every other element (imaginary parts).
    // Sign mask: [1, -1, 1, -1, ...] applied via element-wise multiply.
    let conj_mask = {
        let mut arr = [0.0f32; F32_LANES];
        for i in 0..F32_LANES {
            arr[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        f32x16::from_array(arr)
    };

    let chunks = len / F32_LANES;

    // Conjugate (negate imaginary parts)
    for i in 0..chunks {
        let base = i * F32_LANES;
        let v = f32x16::from_slice(&data[base..]);
        let result = v * conj_mask;
        result.copy_to_slice(&mut data[base..base + F32_LANES]);
    }
    // Scalar tail for conjugate
    for i in (chunks * F32_LANES)..len {
        if i % 2 == 1 {
            data[i] = -data[i];
        }
    }

    fft_f32(data, n);

    // Conjugate and scale by 1/n.
    // Combined: real parts *= scale, imaginary parts *= -scale.
    let scale = 1.0 / n as f32;
    let scale_mask = {
        let mut arr = [0.0f32; F32_LANES];
        for i in 0..F32_LANES {
            arr[i] = if i % 2 == 0 { scale } else { -scale };
        }
        f32x16::from_array(arr)
    };

    for i in 0..chunks {
        let base = i * F32_LANES;
        let v = f32x16::from_slice(&data[base..]);
        let result = v * scale_mask;
        result.copy_to_slice(&mut data[base..base + F32_LANES]);
    }
    // Scalar tail for conjugate+scale
    for i in (chunks * F32_LANES)..len {
        if i % 2 == 0 {
            data[i] *= scale;
        } else {
            data[i] *= -scale;
        }
    }
}

/// In-place radix-2 FFT on interleaved complex f64 data.
pub fn fft_f64(data: &mut [f64], n: usize) {
    #[cfg(feature = "mkl")]
    {
        use numrus_core::mkl_ffi::*;
        unsafe {
            let mut handle: DftiDescriptorHandle = std::ptr::null_mut();
            let status = DftiCreateDescriptor(&mut handle, DFTI_DOUBLE, DFTI_COMPLEX, 1, n as i64);
            if status == 0 {
                DftiSetValue(handle, DFTI_PLACEMENT, DFTI_INPLACE);
                DftiCommitDescriptor(handle);
                DftiComputeForward(handle, data.as_mut_ptr() as *mut std::os::raw::c_void);
                DftiFreeDescriptor(&mut handle);
            }
        }
        return;
    }
    assert!(n.is_power_of_two(), "FFT size must be a power of 2");
    assert_eq!(data.len(), 2 * n);

    bit_reverse_permute_f64(data, n);

    let mut stage_len = 2;
    while stage_len <= n {
        let half = stage_len / 2;
        let angle = -2.0 * std::f64::consts::PI / stage_len as f64;

        // Pre-compute twiddle factors once per stage
        let mut cos_tbl = vec![0.0f64; half];
        let mut sin_tbl = vec![0.0f64; half];
        for j in 0..half {
            let theta = angle * j as f64;
            cos_tbl[j] = theta.cos();
            sin_tbl[j] = theta.sin();
        }

        if half >= F64_COMPLEX_PER_SIMD {
            // SIMD butterfly path — process F64_COMPLEX_PER_SIMD j-values at once.
            let num_simd_chunks = half / F64_COMPLEX_PER_SIMD;
            let mut wr_vecs: Vec<f64x8> = Vec::with_capacity(num_simd_chunks);
            let mut wi_vecs: Vec<f64x8> = Vec::with_capacity(num_simd_chunks);

            for chunk in 0..num_simd_chunks {
                let base_j = chunk * F64_COMPLEX_PER_SIMD;
                let mut wr_arr = [0.0f64; F64_LANES];
                let mut wi_arr = [0.0f64; F64_LANES];
                for c in 0..F64_COMPLEX_PER_SIMD {
                    let cj = cos_tbl[base_j + c];
                    let sj = sin_tbl[base_j + c];
                    wr_arr[2 * c] = cj;
                    wr_arr[2 * c + 1] = cj;
                    wi_arr[2 * c] = -sj;
                    wi_arr[2 * c + 1] = sj;
                }
                wr_vecs.push(f64x8::from_array(wr_arr));
                wi_vecs.push(f64x8::from_array(wi_arr));
            }

            let simd_end = num_simd_chunks * F64_COMPLEX_PER_SIMD;

            for k in (0..n).step_by(stage_len) {
                for chunk in 0..num_simd_chunks {
                    let j = chunk * F64_COMPLEX_PER_SIMD;
                    let even_base = 2 * (k + j);
                    let odd_base = 2 * (k + j + half);

                    let even_v = f64x8::from_slice(&data[even_base..]);
                    let odd_v = f64x8::from_slice(&data[odd_base..]);

                    // Build odd_swapped: swap re<->im within each complex pair
                    let odd_arr = odd_v.to_array();
                    let mut swapped_arr = [0.0f64; F64_LANES];
                    let mut ci = 0;
                    while ci < F64_LANES {
                        swapped_arr[ci] = odd_arr[ci + 1];
                        swapped_arr[ci + 1] = odd_arr[ci];
                        ci += 2;
                    }
                    let odd_swapped = f64x8::from_array(swapped_arr);

                    let twiddle = wr_vecs[chunk] * odd_v + wi_vecs[chunk] * odd_swapped;

                    let result_even = even_v + twiddle;
                    let result_odd = even_v - twiddle;

                    result_even.copy_to_slice(&mut data[even_base..even_base + F64_LANES]);
                    result_odd.copy_to_slice(&mut data[odd_base..odd_base + F64_LANES]);
                }

                // Scalar tail
                for j in simd_end..half {
                    let wr = cos_tbl[j];
                    let wi = sin_tbl[j];

                    let even_re = data[2 * (k + j)];
                    let even_im = data[2 * (k + j) + 1];
                    let odd_re = data[2 * (k + j + half)];
                    let odd_im = data[2 * (k + j + half) + 1];

                    let tr = wr * odd_re - wi * odd_im;
                    let ti = wr * odd_im + wi * odd_re;

                    data[2 * (k + j)] = even_re + tr;
                    data[2 * (k + j) + 1] = even_im + ti;
                    data[2 * (k + j + half)] = even_re - tr;
                    data[2 * (k + j + half) + 1] = even_im - ti;
                }
            }
        } else {
            // Scalar butterfly for early stages (half < SIMD width)
            for k in (0..n).step_by(stage_len) {
                for j in 0..half {
                    let wr = cos_tbl[j];
                    let wi = sin_tbl[j];

                    let even_re = data[2 * (k + j)];
                    let even_im = data[2 * (k + j) + 1];
                    let odd_re = data[2 * (k + j + half)];
                    let odd_im = data[2 * (k + j + half) + 1];

                    let tr = wr * odd_re - wi * odd_im;
                    let ti = wr * odd_im + wi * odd_re;

                    data[2 * (k + j)] = even_re + tr;
                    data[2 * (k + j) + 1] = even_im + ti;
                    data[2 * (k + j + half)] = even_re - tr;
                    data[2 * (k + j + half) + 1] = even_im - ti;
                }
            }
        }

        stage_len *= 2;
    }
}

/// In-place inverse FFT for f64.
pub fn ifft_f64(data: &mut [f64], n: usize) {
    #[cfg(feature = "mkl")]
    {
        use numrus_core::mkl_ffi::*;
        unsafe {
            let mut handle: DftiDescriptorHandle = std::ptr::null_mut();
            let status = DftiCreateDescriptor(&mut handle, DFTI_DOUBLE, DFTI_COMPLEX, 1, n as i64);
            if status == 0 {
                DftiSetValue(handle, DFTI_PLACEMENT, DFTI_INPLACE);
                DftiSetValue(handle, DFTI_BACKWARD_SCALE, 1.0f64 / n as f64);
                DftiCommitDescriptor(handle);
                DftiComputeBackward(handle, data.as_mut_ptr() as *mut std::os::raw::c_void);
                DftiFreeDescriptor(&mut handle);
            }
        }
        return;
    }
    let len = 2 * n;

    // SIMD conjugate: negate every other element (imaginary parts)
    let conj_mask = {
        let mut arr = [0.0f64; F64_LANES];
        for i in 0..F64_LANES {
            arr[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        f64x8::from_array(arr)
    };

    let chunks = len / F64_LANES;

    // Conjugate
    for i in 0..chunks {
        let base = i * F64_LANES;
        let v = f64x8::from_slice(&data[base..]);
        let result = v * conj_mask;
        result.copy_to_slice(&mut data[base..base + F64_LANES]);
    }
    for i in (chunks * F64_LANES)..len {
        if i % 2 == 1 {
            data[i] = -data[i];
        }
    }

    fft_f64(data, n);

    // Conjugate and scale by 1/n
    let scale = 1.0 / n as f64;
    let scale_mask = {
        let mut arr = [0.0f64; F64_LANES];
        for i in 0..F64_LANES {
            arr[i] = if i % 2 == 0 { scale } else { -scale };
        }
        f64x8::from_array(arr)
    };

    for i in 0..chunks {
        let base = i * F64_LANES;
        let v = f64x8::from_slice(&data[base..]);
        let result = v * scale_mask;
        result.copy_to_slice(&mut data[base..base + F64_LANES]);
    }
    for i in (chunks * F64_LANES)..len {
        if i % 2 == 0 {
            data[i] *= scale;
        } else {
            data[i] *= -scale;
        }
    }
}

/// Real-to-complex FFT for f32.
///
/// Input: `n` real f32 values.
/// Output: `n + 2` f32 values (interleaved complex, n/2 + 1 complex numbers).
///
/// Returns a new Vec with the complex output.
pub fn rfft_f32(input: &[f32]) -> Vec<f32> {
    // MKL DFTI rfft has different output layout (CCS vs interleaved).
    // Use hand-rolled path for now; wire when output format is confirmed.
    let n = input.len();
    assert!(n.is_power_of_two(), "FFT size must be a power of 2");

    // Pack real data into complex (zero imaginary parts)
    let mut complex = vec![0.0f32; 2 * n];
    for i in 0..n {
        complex[2 * i] = input[i];
    }

    fft_f32(&mut complex, n);

    // Return only the first n/2 + 1 complex values (positive frequencies)
    complex[..2 * (n / 2 + 1)].to_vec()
}

// ============================================================================
// Bit-reversal permutation
// ============================================================================

fn bit_reverse_permute_f32(data: &mut [f32], n: usize) {
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = bit_reverse(i as u32, bits) as usize;
        if i < j {
            data.swap(2 * i, 2 * j);
            data.swap(2 * i + 1, 2 * j + 1);
        }
    }
}

fn bit_reverse_permute_f64(data: &mut [f64], n: usize) {
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = bit_reverse(i as u32, bits) as usize;
        if i < j {
            data.swap(2 * i, 2 * j);
            data.swap(2 * i + 1, 2 * j + 1);
        }
    }
}

#[inline(always)]
fn bit_reverse(mut x: u32, bits: u32) -> u32 {
    let mut result = 0u32;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_ifft_roundtrip_f32() {
        let n = 8;
        let original: Vec<f32> = (0..n).map(|i| i as f32).collect();

        // Pack into complex
        let mut data = vec![0.0f32; 2 * n];
        for i in 0..n {
            data[2 * i] = original[i];
        }

        // Forward FFT
        fft_f32(&mut data, n);

        // Inverse FFT
        ifft_f32(&mut data, n);

        // Check roundtrip
        for i in 0..n {
            assert!(
                (data[2 * i] - original[i]).abs() < 1e-5,
                "Roundtrip mismatch at {}: {} vs {}",
                i,
                data[2 * i],
                original[i]
            );
            assert!(
                data[2 * i + 1].abs() < 1e-5,
                "Imaginary part should be ~0 at {}: {}",
                i,
                data[2 * i + 1]
            );
        }
    }

    #[test]
    fn test_fft_ifft_roundtrip_f64() {
        let n = 16;
        let original: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();

        let mut data = vec![0.0f64; 2 * n];
        for i in 0..n {
            data[2 * i] = original[i];
        }

        fft_f64(&mut data, n);
        ifft_f64(&mut data, n);

        for i in 0..n {
            assert!(
                (data[2 * i] - original[i]).abs() < 1e-10,
                "f64 roundtrip mismatch at {}",
                i
            );
        }
    }

    #[test]
    fn test_fft_dc_component() {
        // FFT of [1, 1, 1, 1] should have DC = 4, all others = 0
        let n = 4;
        let mut data = vec![0.0f32; 8];
        for i in 0..n {
            data[2 * i] = 1.0;
        }
        fft_f32(&mut data, n);
        assert!((data[0] - 4.0).abs() < 1e-6, "DC component should be 4");
        assert!(data[1].abs() < 1e-6, "DC imaginary should be 0");
        // Non-DC components should be 0
        for i in 1..n {
            assert!(
                data[2 * i].abs() < 1e-6 && data[2 * i + 1].abs() < 1e-6,
                "Non-DC component {} should be 0",
                i
            );
        }
    }

    #[test]
    fn test_rfft() {
        let input = vec![1.0f32, 0.0, -1.0, 0.0];
        let output = rfft_f32(&input);
        // DC = 0, Nyquist = -2, middle = 2
        assert!((output[0] - 0.0).abs() < 1e-6); // DC real
        assert!((output[1] - 0.0).abs() < 1e-6); // DC imag
    }

    #[test]
    fn test_bit_reverse() {
        assert_eq!(bit_reverse(0, 3), 0);
        assert_eq!(bit_reverse(1, 3), 4);
        assert_eq!(bit_reverse(2, 3), 2);
        assert_eq!(bit_reverse(3, 3), 6);
    }
}
