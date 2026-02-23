use crate::num_array::NumArray;
use crate::simd_ops::SimdOps;
use crate::traits::{FromU32, FromUsize, NumOps};
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};

impl<T, Ops> NumArray<T, Ops>
where
    T: Clone
        + Mul<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Copy
        + Debug
        + Default
        + PartialOrd
        + FromU32
        + FromUsize,
    Ops: SimdOps<T>,
{
    /// Computes the mean of the array.
    ///
    /// # Returns
    /// A new `NumArray` instance containing the mean value.
    ///
    /// # Example
    /// ```
    /// use numrus_rs::NumArrayF32;
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let array = NumArrayF32::new(data);
    /// let mean_array = array.mean();
    /// println!("Mean array: {:?}", mean_array.get_data());
    /// ```
    pub fn mean(&self) -> NumArray<T, Ops> {
        let sum: T = Ops::sum(&self.data);
        let count = T::from_u32(self.data.len() as u32);
        let mean = sum / count;
        NumArray::new(vec![mean])
    }
    /// Computes the mean along the specified axis.
    ///
    /// # Parameters
    /// * `axis` - An optional reference to a vector of axis to compute the mean along.
    ///
    /// # Returns
    /// A new `NumArray` instance containing the mean values along the specified axis.
    ///
    /// # Panics
    /// Panics if any of the specified axis is out of bounds.
    ///
    /// # Example
    /// ```
    /// use numrus_rs::NumArrayF32;
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let array = NumArrayF32::new_with_shape(data, vec![2, 3]);
    /// let mean_array = array.mean_axis(Some(&[1]));
    /// println!("Mean array: {:?}", mean_array.get_data());
    /// ```
    pub fn mean_axis(&self, axis: Option<&[usize]>) -> NumArray<T, Ops> {
        match axis {
            Some(axis) => {
                for &axis in axis {
                    assert!(axis < self.shape.len(), "Axis {} out of bounds.", axis);
                }

                let mut reduced_shape = self.shape.clone();
                let mut total_elements_to_reduce = 1;

                for &axis in axis {
                    total_elements_to_reduce *= self.shape[axis];
                    reduced_shape[axis] = 1; // Marking this axis for reduction
                }

                let reduced_size: usize = reduced_shape.iter().product();
                let mut reduced_data = vec![T::from_u32(0); reduced_size];

                // Process each element in the data
                for (i, &val) in self.data.iter().enumerate() {
                    let reduced_idx = self.calculate_reduced_index(i, &reduced_shape);
                    reduced_data[reduced_idx] = reduced_data[reduced_idx] + val;
                }

                // Divide each element in reduced_data by the number of elements that contributed to it
                for val in reduced_data.iter_mut() {
                    *val = *val / T::from_usize(total_elements_to_reduce);
                }
                // let's squeeze the reduced shape
                reduced_shape = reduced_shape
                    .into_iter()
                    .filter(|&x| x != 1)
                    .collect::<Vec<_>>();
                NumArray::new_with_shape(reduced_data, reduced_shape)
            }
            None => self.mean(),
        }
    }
    /// Sorts the array in ascending order.
    /// The original array is not modified.
    /// # Returns
    /// A new `NumArray` instance containing the sorted values.
    /// # Example
    /// ```
    /// use numrus_rs::NumArrayF32;
    /// let data = vec![3.0, 1.0, 4.0, 2.0];
    /// let array = NumArrayF32::new(data);
    /// let sorted_array = array.sort();
    /// println!("Sorted array: {:?}", sorted_array.get_data());
    /// ```
    pub fn sort(&self) -> NumArray<T, Ops> {
        let mut sorted_data = self.data.clone();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        NumArray::new(sorted_data)
    }

    /// Computes the median of the array.
    /// The original array is not modified.
    ///
    /// # Returns
    /// A new `NumArray` instance containing the median value.
    ///
    /// # Examples
    /// ```
    /// use numrus_rs::NumArrayF32;
    /// let data = vec![3.0, 1.0, 4.0, 2.0];
    /// let array = NumArrayF32::new(data);
    /// let median_array = array.median();
    /// println!("Median array: {:?}", median_array.get_data());
    /// ```
    pub fn median(&self) -> NumArray<T, Ops> {
        let sorted_data = self.sort();
        let median = Self::calculate_median(sorted_data.get_data());

        NumArray::new(vec![median])
    }
    /// Computes the median along the specified axis.
    /// The original array is not modified.
    ///
    /// # Parameters
    /// * `axis` - An optional reference to a vector of axis to compute the median along.
    ///
    /// # Returns
    /// A new `NumArray` instance containing the median values along the specified axis.
    ///
    /// # Panics
    /// Panics if any of the specified axis is out of bounds.
    ///
    /// # Examples
    /// ```
    /// use numrus_rs::NumArrayF32;
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let array = NumArrayF32::new_with_shape(data, vec![2, 3]);
    /// let median_array = array.median_axis(Some(&[1]));
    /// println!("Median array: {:?}", median_array.get_data());
    /// ```
    pub fn median_axis(&self, axis: Option<&[usize]>) -> NumArray<T, Ops> {
        match axis {
            Some(axis) => {
                let mut reduced_shape = self.shape.clone();
                let mut total_elements_to_reduce = 1;
                for &axis in axis {
                    assert!(axis < self.shape.len(), "Axis {} out of bounds.", axis);
                    reduced_shape[axis] = 1;
                    total_elements_to_reduce *= self.shape[axis];
                }

                let reduced_size: usize = reduced_shape.iter().product();
                let mut reduced_data = vec![T::from_u32(0); reduced_size];

                let mut accumulator = vec![T::from_u32(0); total_elements_to_reduce * reduced_size];

                let mut accumulator_ptrs: Vec<&mut [T]> =
                    accumulator.chunks_mut(total_elements_to_reduce).collect();
                let mut counts = vec![0; accumulator_ptrs.len()];

                for (i, &val) in self.data.iter().enumerate() {
                    let reduced_idx = self.calculate_reduced_index(i, &reduced_shape);
                    accumulator_ptrs[reduced_idx][counts[reduced_idx]] = val;
                    counts[reduced_idx] += 1;
                }

                for (i, ptr) in accumulator_ptrs.iter_mut().enumerate() {
                    ptr.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    reduced_data[i] = Self::calculate_median(ptr);
                }

                reduced_shape = reduced_shape
                    .into_iter()
                    .filter(|&x| x != 1)
                    .collect::<Vec<_>>();

                if reduced_shape.is_empty() {
                    NumArray::new(reduced_data)
                } else {
                    NumArray::new_with_shape(reduced_data, reduced_shape)
                }
            }
            None => self.median(),
        }
    }

    fn calculate_median(values: &[T]) -> T {
        let len = values.len();
        if len.is_multiple_of(2) {
            (values[len / 2 - 1] + values[len / 2]) / T::from_u32(2)
        } else {
            values[len / 2]
        }
    }

    /// Computes the variance of the array (population variance).
    ///
    /// Variance = mean((x - mean(x))^2)
    ///
    /// # Returns
    /// A new `NumArray` instance containing the variance value.
    ///
    /// # Example
    /// ```
    /// use numrus_rs::NumArrayF32;
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    /// let array = NumArrayF32::new(data);
    /// let var = array.var().item();
    /// assert!((var - 2.0).abs() < 1e-5);
    /// ```
    // TODO(simd): REFACTOR — var() uses scalar sum-of-squared-deviations loop.
    // Fix: SIMD sub_scalar(data, mean) → SIMD mul_array(diff, diff) → SIMD sum.
    pub fn var(&self) -> NumArray<T, Ops> {
        let n = self.data.len();
        if n == 0 {
            return NumArray::new(vec![T::from_u32(0)]);
        }
        let mean_val = Ops::sum(&self.data) / T::from_u32(n as u32);
        let mut sum_sq = T::from_u32(0);
        for &x in &self.data {
            let diff = x - mean_val;
            sum_sq = sum_sq + diff * diff;
        }
        NumArray::new(vec![sum_sq / T::from_u32(n as u32)])
    }

    /// Computes the variance along the specified axis (population variance).
    ///
    /// # Parameters
    /// * `axis` - An optional reference to a slice of axes to compute variance along.
    ///
    /// # Returns
    /// A new `NumArray` instance containing the variance values along the specified axis.
    ///
    /// # Example
    /// ```
    /// use numrus_rs::NumArrayF32;
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let array = NumArrayF32::new_with_shape(data, vec![2, 3]);
    /// let var = array.var_axis(Some(&[1]));
    /// ```
    pub fn var_axis(&self, axis: Option<&[usize]>) -> NumArray<T, Ops> {
        match axis {
            Some(axes) => {
                let mean_arr = self.mean_axis(Some(axes));
                let mean_data = mean_arr.get_data();

                let mut reduced_shape = self.shape.clone();
                let mut total_elements_to_reduce = 1;
                for &ax in axes {
                    assert!(ax < self.shape.len(), "Axis {} out of bounds.", ax);
                    total_elements_to_reduce *= self.shape[ax];
                    reduced_shape[ax] = 1;
                }

                let reduced_size: usize = reduced_shape.iter().product();
                let mut sum_sq = vec![T::from_u32(0); reduced_size];

                for (i, &val) in self.data.iter().enumerate() {
                    let reduced_idx = self.calculate_reduced_index(i, &reduced_shape);
                    let diff = val - mean_data[reduced_idx];
                    sum_sq[reduced_idx] = sum_sq[reduced_idx] + diff * diff;
                }

                for val in sum_sq.iter_mut() {
                    *val = *val / T::from_usize(total_elements_to_reduce);
                }

                let squeezed_shape: Vec<usize> =
                    reduced_shape.into_iter().filter(|&x| x != 1).collect();
                if squeezed_shape.is_empty() {
                    NumArray::new(sum_sq)
                } else {
                    NumArray::new_with_shape(sum_sq, squeezed_shape)
                }
            }
            None => self.var(),
        }
    }

    /// Computes the percentile of the array using linear interpolation.
    ///
    /// # Parameters
    /// * `p` - The percentile to compute, must be in [0, 100].
    ///
    /// # Returns
    /// A new `NumArray` instance containing the percentile value.
    ///
    /// # Panics
    /// Panics if `p` is outside [0, 100] or if the array is empty.
    ///
    /// # Example
    /// ```
    /// use numrus_rs::NumArrayF32;
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    /// let array = NumArrayF32::new(data);
    /// let p50 = array.percentile(50.0).item();
    /// assert!((p50 - 3.0).abs() < 1e-5);
    /// ```
    pub fn percentile(&self, p: T) -> NumArray<T, Ops> {
        assert!(
            !self.data.is_empty(),
            "Cannot compute percentile of empty array."
        );
        assert!(
            p >= T::from_u32(0) && p <= T::from_u32(100),
            "Percentile must be between 0 and 100."
        );
        let sorted = self.sort();
        let data = sorted.get_data();
        let n = data.len();
        if n == 1 {
            return NumArray::new(vec![data[0]]);
        }
        // Linear interpolation: index = p/100 * (n-1)
        let idx_f = p / T::from_u32(100) * T::from_u32((n - 1) as u32);
        // We need floor and ceil. Use integer conversion approach.
        // For floats this works; for integers it's exact.
        let n_minus_1 = T::from_u32((n - 1) as u32);
        let hundred = T::from_u32(100);
        // Compute fractional position manually
        // lower = p * (n-1) / 100 truncated
        // We'll iterate to find the right spot
        let mut lower = 0usize;
        while lower < n - 1 {
            let threshold = T::from_usize(lower + 1) * hundred;
            if p * n_minus_1 < threshold {
                break;
            }
            lower += 1;
        }
        if lower > 0 {
            // Check if we overshot
            let threshold = T::from_usize(lower) * hundred;
            if p * n_minus_1 < threshold {
                lower -= 1;
            }
        }
        let upper = if lower + 1 < n { lower + 1 } else { lower };
        if lower == upper {
            return NumArray::new(vec![data[lower]]);
        }
        // fraction = idx_f - floor(idx_f) = (p*(n-1)/100) - lower
        let frac = idx_f - T::from_usize(lower);
        let result = data[lower] + frac * (data[upper] - data[lower]);
        NumArray::new(vec![result])
    }

    /// Computes the percentile along the specified axis.
    ///
    /// # Parameters
    /// * `p` - The percentile to compute, must be in [0, 100].
    /// * `axis` - An optional reference to a slice of axes.
    ///
    /// # Returns
    /// A new `NumArray` instance containing the percentile values along the specified axis.
    ///
    /// # Example
    /// ```
    /// use numrus_rs::NumArrayF32;
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let array = NumArrayF32::new_with_shape(data, vec![2, 3]);
    /// let p50 = array.percentile_axis(50.0, Some(&[1]));
    /// ```
    pub fn percentile_axis(&self, p: T, axis: Option<&[usize]>) -> NumArray<T, Ops> {
        match axis {
            Some(axes) => {
                let mut reduced_shape = self.shape.clone();
                let mut total_elements_to_reduce = 1;
                for &ax in axes {
                    assert!(ax < self.shape.len(), "Axis {} out of bounds.", ax);
                    reduced_shape[ax] = 1;
                    total_elements_to_reduce *= self.shape[ax];
                }

                let reduced_size: usize = reduced_shape.iter().product();
                let mut accumulator = vec![T::from_u32(0); total_elements_to_reduce * reduced_size];
                let mut acc_slices: Vec<&mut [T]> =
                    accumulator.chunks_mut(total_elements_to_reduce).collect();
                let mut counts = vec![0usize; reduced_size];

                for (i, &val) in self.data.iter().enumerate() {
                    let reduced_idx = self.calculate_reduced_index(i, &reduced_shape);
                    acc_slices[reduced_idx][counts[reduced_idx]] = val;
                    counts[reduced_idx] += 1;
                }

                let mut result_data = vec![T::from_u32(0); reduced_size];
                for (i, slice) in acc_slices.iter_mut().enumerate() {
                    slice.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let temp = NumArray::<T, Ops>::new(slice.to_vec());
                    result_data[i] = temp.percentile(p).get_data()[0];
                }

                let squeezed_shape: Vec<usize> =
                    reduced_shape.into_iter().filter(|&x| x != 1).collect();
                if squeezed_shape.is_empty() {
                    NumArray::new(result_data)
                } else {
                    NumArray::new_with_shape(result_data, squeezed_shape)
                }
            }
            None => self.percentile(p),
        }
    }
}

/// Standard deviation requires NumOps for sqrt
impl<T, Ops> NumArray<T, Ops>
where
    T: Clone
        + Mul<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Copy
        + Debug
        + Default
        + PartialOrd
        + FromU32
        + FromUsize
        + NumOps,
    Ops: SimdOps<T>,
{
    /// Computes the standard deviation of the array (population std).
    ///
    /// std = sqrt(var)
    ///
    /// # Returns
    /// A new `NumArray` instance containing the standard deviation.
    ///
    /// # Example
    /// ```
    /// use numrus_rs::NumArrayF32;
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    /// let array = NumArrayF32::new(data);
    /// let std_val = array.std().item();
    /// assert!((std_val - 1.4142135).abs() < 1e-5);
    /// ```
    pub fn std(&self) -> NumArray<T, Ops> {
        let var_val = self.var().get_data()[0];
        NumArray::new(vec![var_val.sqrt()])
    }

    /// Computes the standard deviation along the specified axis (population std).
    ///
    /// # Parameters
    /// * `axis` - An optional reference to a slice of axes.
    ///
    /// # Returns
    /// A new `NumArray` instance containing the std values along the specified axis.
    ///
    /// # Example
    /// ```
    /// use numrus_rs::NumArrayF32;
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let array = NumArrayF32::new_with_shape(data, vec![2, 3]);
    /// let std_arr = array.std_axis(Some(&[1]));
    /// ```
    pub fn std_axis(&self, axis: Option<&[usize]>) -> NumArray<T, Ops> {
        let var_arr = self.var_axis(axis);
        // TODO(simd): REFACTOR — scalar sqrt via iter().map(). Route through VML vssqrt/vdsqrt.
        let std_data: Vec<T> = var_arr.get_data().iter().map(|&v| v.sqrt()).collect();
        if var_arr.shape().len() > 1 || (var_arr.shape().len() == 1 && var_arr.shape()[0] > 1) {
            NumArray::new_with_shape(std_data, var_arr.shape().to_vec())
        } else {
            NumArray::new(std_data)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::num_array::{NumArrayF32, NumArrayF64};

    #[test]
    fn test_mean_f32() {
        let a = NumArrayF32::new(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(a.mean().item(), 2.5);
    }

    #[test]
    fn test_mean_f64() {
        let a = NumArrayF64::new(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(a.mean().item(), 2.5);
    }

    #[test]
    fn test_mean_axis_1d() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let array = NumArrayF32::new_with_shape(data, vec![4]);
        let mean_array = array.mean_axis(Some(&[0]));
        assert_eq!(mean_array.get_data(), &vec![2.5]);
    }

    #[test]
    fn test_mean_axis_2d() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let array = NumArrayF32::new_with_shape(data, vec![2, 3]);
        let mean_array = array.mean_axis(Some(&[1]));

        assert_eq!(mean_array.shape(), &[2]);
        assert_eq!(mean_array.get_data(), &vec![2.0, 5.0]); // Mean along the second axis (columns)
    }

    #[test]
    fn test_mean_axis_2d_column() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let array = NumArrayF32::new_with_shape(data, vec![2, 3]);
        // Compute mean across columns (axis 1)
        let mean_array = array.mean_axis(Some(&[1]));
        assert_eq!(mean_array.get_data(), &vec![2.0, 5.0]); // Mean along the second axis (columns)
    }

    #[test]
    fn test_mean_axis_3d() {
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let array = NumArrayF32::new_with_shape(data, vec![2, 2, 3]);
        // Compute mean across the last two axis (1 and 2)
        let mean_array = array.mean_axis(Some(&[1, 2]));
        assert_eq!(mean_array.get_data(), &vec![3.5, 9.5]);
    }

    #[test]
    fn test_mean_axis_invalid_axis() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let array = NumArrayF32::new_with_shape(data, vec![4]);
        // Attempt to compute mean across an invalid axis
        let result = std::panic::catch_unwind(|| array.mean_axis(Some(&[1])));
        assert!(result.is_err(), "Should panic due to invalid axis");
    }

    #[test]
    fn test_sort_f32() {
        let a = NumArrayF32::new(vec![5.0, 2.0, 3.0, 1.0, 4.0]);
        assert_eq!(a.sort().get_data(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_sort_f64() {
        let a = NumArrayF64::new(vec![5.0, 2.0, 3.0, 1.0, 4.0]);
        assert_eq!(a.sort().get_data(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_median_f32_one_elem() {
        let a = NumArrayF32::new(vec![1.0]);
        assert_eq!(a.median().item(), 1.0);
    }

    #[test]
    fn test_median_f32_even() {
        let a = NumArrayF32::new(vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0]);
        assert_eq!(a.median().item(), 3.5);
    }

    #[test]
    fn test_median_f32_uneven() {
        let a = NumArrayF32::new(vec![2.0, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0]);
        assert_eq!(a.median().item(), 4.0);
    }

    #[test]
    fn test_median_f64_uneven() {
        let a = NumArrayF64::new(vec![1.0, 2.0, 2.0, 4.0, 5.0, 6.0, 7.0]);
        assert_eq!(a.median().item(), 4.0);
    }

    #[test]
    fn test_median_axis_1d_even() {
        let data = vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0];
        let array = NumArrayF32::new_with_shape(data, vec![6]);
        let max_array = array.median_axis(Some(&[0]));
        assert_eq!(max_array.get_data(), &vec![3.5]);
    }

    #[test]
    fn test_median_axis_1d_uneven() {
        let data = vec![2.0, 2.0, 4.0, 3.0, 6.0, 5.0, 7.0];
        let array = NumArrayF32::new_with_shape(data, vec![7]);
        let max_array = array.median_axis(Some(&[0]));
        assert_eq!(max_array.get_data(), &vec![4.0]);
    }

    #[test]
    fn test_median_axis_2d_even() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let array = NumArrayF32::new_with_shape(data, vec![2, 3]);
        let max_array = array.median_axis(Some(&[0]));
        assert_eq!(max_array.get_data(), &vec![2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_median_axis_2d_uneven() {
        let data = vec![2.0, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let array = NumArrayF32::new_with_shape(data, vec![3, 3]);
        let max_array = array.median_axis(Some(&[1]));
        assert_eq!(max_array.get_data(), &vec![2.0, 5.0, 8.0]);
    }

    #[test]
    fn test_median_axis_3d_even() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let array = NumArrayF32::new_with_shape(data, vec![2, 2, 2]);
        let max_array = array.median_axis(Some(&[0, 1]));
        assert_eq!(max_array.get_data(), &vec![4.0, 5.0]);
    }

    #[test]
    fn test_median_axis_3d_uneven() {
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let array = NumArrayF32::new_with_shape(data, vec![3, 2, 2]);
        let max_array = array.median_axis(Some(&[0]));
        assert_eq!(max_array.get_data(), &vec![5.0, 6.0, 7.0, 8.0]);
    }

    // --- Variance tests ---

    #[test]
    fn test_var_f32() {
        let a = NumArrayF32::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let var = a.var().item();
        assert!((var - 2.0).abs() < 1e-5, "Expected 2.0, got {}", var);
    }

    #[test]
    fn test_var_f64() {
        let a = NumArrayF64::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let var = a.var().item();
        assert!((var - 2.0).abs() < 1e-10, "Expected 2.0, got {}", var);
    }

    #[test]
    fn test_var_constant() {
        let a = NumArrayF32::new(vec![5.0, 5.0, 5.0, 5.0]);
        assert!((a.var().item() - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_var_axis_2d() {
        // [[1, 2, 3], [4, 5, 6]]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let array = NumArrayF32::new_with_shape(data, vec![2, 3]);
        let var = array.var_axis(Some(&[1]));
        // Row 0: mean=2, var = ((1-2)^2+(2-2)^2+(3-2)^2)/3 = 2/3
        // Row 1: mean=5, var = ((4-5)^2+(5-5)^2+(6-5)^2)/3 = 2/3
        let expected = 2.0 / 3.0;
        assert!(
            (var.get_data()[0] - expected).abs() < 1e-5,
            "Expected {}, got {}",
            expected,
            var.get_data()[0]
        );
        assert!(
            (var.get_data()[1] - expected).abs() < 1e-5,
            "Expected {}, got {}",
            expected,
            var.get_data()[1]
        );
    }

    // --- Standard deviation tests ---

    #[test]
    fn test_std_f32() {
        let a = NumArrayF32::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let std_val = a.std().item();
        // var = 2.0, std = sqrt(2) ≈ 1.4142135
        let expected = 2.0_f32.sqrt();
        assert!(
            (std_val - expected).abs() < 1e-5,
            "Expected {}, got {}",
            expected,
            std_val
        );
    }

    #[test]
    fn test_std_f64() {
        let a = NumArrayF64::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let std_val = a.std().item();
        let expected = 2.0_f64.sqrt();
        assert!(
            (std_val - expected).abs() < 1e-10,
            "Expected {}, got {}",
            expected,
            std_val
        );
    }

    #[test]
    fn test_std_axis_2d() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let array = NumArrayF32::new_with_shape(data, vec![2, 3]);
        let std_arr = array.std_axis(Some(&[1]));
        let expected = (2.0_f32 / 3.0).sqrt();
        assert!((std_arr.get_data()[0] - expected).abs() < 1e-5);
        assert!((std_arr.get_data()[1] - expected).abs() < 1e-5);
    }

    // --- Percentile tests ---

    #[test]
    fn test_percentile_50() {
        let a = NumArrayF32::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let p50 = a.percentile(50.0).item();
        assert!((p50 - 3.0).abs() < 1e-5, "Expected 3.0, got {}", p50);
    }

    #[test]
    fn test_percentile_0_and_100() {
        let a = NumArrayF32::new(vec![10.0, 20.0, 30.0, 40.0, 50.0]);
        let p0 = a.percentile(0.0).item();
        let p100 = a.percentile(100.0).item();
        assert!((p0 - 10.0).abs() < 1e-5, "Expected 10.0, got {}", p0);
        assert!((p100 - 50.0).abs() < 1e-5, "Expected 50.0, got {}", p100);
    }

    #[test]
    fn test_percentile_25_75() {
        let a = NumArrayF32::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let p25 = a.percentile(25.0).item();
        let p75 = a.percentile(75.0).item();
        // p25 = 1 + 0.25*4*(5-1-position) ... linear interp on index 1.0 -> value 2.0
        // index = 25/100 * 4 = 1.0, so exact -> data[1] = 2.0
        assert!((p25 - 2.0).abs() < 1e-5, "Expected 2.0, got {}", p25);
        // index = 75/100 * 4 = 3.0, exact -> data[3] = 4.0
        assert!((p75 - 4.0).abs() < 1e-5, "Expected 4.0, got {}", p75);
    }

    #[test]
    fn test_percentile_f64() {
        let a = NumArrayF64::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let p50 = a.percentile(50.0).item();
        assert!((p50 - 3.0).abs() < 1e-10, "Expected 3.0, got {}", p50);
    }

    #[test]
    fn test_percentile_axis_2d() {
        let data = vec![3.0, 1.0, 2.0, 6.0, 4.0, 5.0];
        let array = NumArrayF32::new_with_shape(data, vec![2, 3]);
        let p50 = array.percentile_axis(50.0, Some(&[1]));
        // Row 0: sorted [1,2,3], median=2.0
        // Row 1: sorted [4,5,6], median=5.0
        assert!((p50.get_data()[0] - 2.0).abs() < 1e-5);
        assert!((p50.get_data()[1] - 5.0).abs() < 1e-5);
    }
}
