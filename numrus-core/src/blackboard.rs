//! Zero-copy blackboard: a shared mutable memory arena with SIMD-aligned allocations.
//!
//! The blackboard eliminates serialization between crates. All three crates (numrus-rs,
//! numrus_blas, numrus_mkl) can allocate buffers from the same arena and operate on them
//! directly — no copies, no marshalling.
//!
//! # Design
//!
//! - 64-byte aligned allocations (AVX-512 cache-line aligned)
//! - Named buffers for clarity (`"A"`, `"B"`, `"C"` for GEMM operands)
//! - Split-borrow API: multiple buffers can be mutably borrowed simultaneously
//!   as long as they don't alias (like struct field borrows)
//! - Sound because each named buffer is a separate heap allocation; `&mut self`
//!   guarantees exclusive access to the Blackboard, and name-distinctness assertions
//!   guarantee the returned slices point to disjoint memory.
//!
//! # Example
//!
//! ```
//! use numrus_core::Blackboard;
//!
//! let mut bb = Blackboard::new();
//!
//! // Allocate GEMM operands
//! bb.alloc_f32("A", 1024 * 1024);
//! bb.alloc_f32("B", 1024 * 1024);
//! bb.alloc_f32("C", 1024 * 1024);
//!
//! // Get non-overlapping mutable slices — no borrow conflicts
//! let (a_slice, b_slice, c_slice) = bb.borrow_3_mut_f32("A", "B", "C").unwrap();
//!
//! // Fill A and B, compute into C — all zero-copy
//! a_slice.fill(1.0);
//! b_slice.fill(2.0);
//! // ... numrus_blas::sgemm operates directly on these slices
//! ```

use std::alloc;
use std::collections::HashMap;
use std::marker::PhantomData;

/// Alignment for all blackboard allocations (AVX-512 = 64 bytes).
const ALIGNMENT: usize = 64;

/// Type tag for buffer element types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DType {
    #[default]
    F32,
    F64,
    I32,
    I64,
    U8,
}

impl DType {
    fn element_size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F64 => 8,
            DType::I32 => 4,
            DType::I64 => 8,
            DType::U8 => 1,
        }
    }
}

/// Metadata for a single buffer allocation.
///
/// The `ptr` field points to a separately heap-allocated, 64-byte-aligned region.
/// This pointer is *not* interior to the HashMap — it's an independent allocation.
struct BufferMeta {
    ptr: *mut u8,
    len_elements: usize,
    dtype: DType,
    layout: alloc::Layout,
}

impl Drop for BufferMeta {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.layout.size() > 0 {
            unsafe { alloc::dealloc(self.ptr, self.layout) };
        }
    }
}

/// Zero-copy shared memory arena with SIMD-aligned allocations.
///
/// The blackboard owns all buffer memory. Crates borrow slices directly
/// from the arena — no serialization, no copies.
///
/// # Safety model
///
/// Each named buffer is a separate heap allocation at a distinct address.
/// The split-borrow methods (`borrow_2_mut_*`, `borrow_3_mut_*`) take `&mut self`
/// for exclusive access, assert name-distinctness, then create `&mut [T]` slices
/// from the heap pointers. Since each buffer's data lives in its own allocation
/// (disjoint from both the HashMap and other buffers), no aliasing occurs.
///
/// # Thread safety
///
/// `Blackboard` is `!Send` and `!Sync` (due to raw pointers in `BufferMeta`).
/// For multi-threaded access, wrap in `Mutex<Blackboard>`.
pub struct Blackboard {
    buffers: HashMap<String, BufferMeta>,
    /// Explicit `!Send + !Sync` marker.  `BufferMeta` already contains `*mut u8`
    /// which poisons auto-traits, but this marker guarantees `!Send + !Sync`
    /// survives any future refactor of `BufferMeta` internals.
    _not_send_sync: PhantomData<*mut ()>,
}

impl Blackboard {
    pub fn new() -> Self {
        Self {
            buffers: HashMap::new(),
            _not_send_sync: PhantomData,
        }
    }

    /// Allocate a 64-byte-aligned f32 buffer with `len` elements.
    pub fn alloc_f32(&mut self, name: &str, len: usize) {
        self.alloc_typed(name, len, DType::F32);
    }

    /// Allocate a 64-byte-aligned f64 buffer with `len` elements.
    pub fn alloc_f64(&mut self, name: &str, len: usize) {
        self.alloc_typed(name, len, DType::F64);
    }

    /// Allocate a 64-byte-aligned i32 buffer with `len` elements.
    pub fn alloc_i32(&mut self, name: &str, len: usize) {
        self.alloc_typed(name, len, DType::I32);
    }

    /// Allocate a 64-byte-aligned i64 buffer with `len` elements.
    pub fn alloc_i64(&mut self, name: &str, len: usize) {
        self.alloc_typed(name, len, DType::I64);
    }

    /// Allocate a 64-byte-aligned u8 buffer with `len` elements.
    pub fn alloc_u8(&mut self, name: &str, len: usize) {
        self.alloc_typed(name, len, DType::U8);
    }

    fn alloc_typed(&mut self, name: &str, len: usize, dtype: DType) {
        // Deallocate existing buffer with the same name if present
        self.buffers.remove(name);

        let byte_len = len * dtype.element_size();
        let layout =
            alloc::Layout::from_size_align(byte_len.max(1), ALIGNMENT).expect("Invalid layout");

        let ptr = if byte_len == 0 {
            std::ptr::null_mut()
        } else {
            let p = unsafe { alloc::alloc_zeroed(layout) };
            if p.is_null() {
                alloc::handle_alloc_error(layout);
            }
            p
        };

        self.buffers.insert(
            name.to_string(),
            BufferMeta {
                ptr,
                len_elements: len,
                dtype,
                layout,
            },
        );
    }

    /// Helper: look up a buffer and verify its dtype.
    fn meta_checked(&self, name: &str, expected: DType) -> Option<&BufferMeta> {
        let meta = self.buffers.get(name)?;
        if meta.dtype != expected {
            return None;
        }
        Some(meta)
    }

    /// Get an immutable f32 slice for the named buffer.
    ///
    /// Returns `None` if the buffer doesn't exist or isn't f32.
    pub fn get_f32(&self, name: &str) -> Option<&[f32]> {
        let meta = self.meta_checked(name, DType::F32)?;
        if meta.len_elements == 0 {
            return Some(&[]);
        }
        Some(unsafe { std::slice::from_raw_parts(meta.ptr as *const f32, meta.len_elements) })
    }

    /// Get a mutable f32 slice for the named buffer.
    ///
    /// Returns `None` if the buffer doesn't exist or isn't f32.
    pub fn get_f32_mut(&mut self, name: &str) -> Option<&mut [f32]> {
        let meta = self.meta_checked(name, DType::F32)?;
        if meta.len_elements == 0 {
            return Some(&mut []);
        }
        Some(unsafe { std::slice::from_raw_parts_mut(meta.ptr as *mut f32, meta.len_elements) })
    }

    /// Get an immutable f64 slice for the named buffer.
    ///
    /// Returns `None` if the buffer doesn't exist or isn't f64.
    pub fn get_f64(&self, name: &str) -> Option<&[f64]> {
        let meta = self.meta_checked(name, DType::F64)?;
        if meta.len_elements == 0 {
            return Some(&[]);
        }
        Some(unsafe { std::slice::from_raw_parts(meta.ptr as *const f64, meta.len_elements) })
    }

    /// Get a mutable f64 slice for the named buffer.
    ///
    /// Returns `None` if the buffer doesn't exist or isn't f64.
    pub fn get_f64_mut(&mut self, name: &str) -> Option<&mut [f64]> {
        let meta = self.meta_checked(name, DType::F64)?;
        if meta.len_elements == 0 {
            return Some(&mut []);
        }
        Some(unsafe { std::slice::from_raw_parts_mut(meta.ptr as *mut f64, meta.len_elements) })
    }

    /// Get an immutable u8 slice for the named buffer.
    ///
    /// Returns `None` if the buffer doesn't exist or isn't u8.
    pub fn get_u8(&self, name: &str) -> Option<&[u8]> {
        let meta = self.meta_checked(name, DType::U8)?;
        if meta.len_elements == 0 {
            return Some(&[]);
        }
        Some(unsafe { std::slice::from_raw_parts(meta.ptr as *const u8, meta.len_elements) })
    }

    /// Get a mutable u8 slice for the named buffer.
    ///
    /// Returns `None` if the buffer doesn't exist or isn't u8.
    pub fn get_u8_mut(&mut self, name: &str) -> Option<&mut [u8]> {
        let meta = self.meta_checked(name, DType::U8)?;
        if meta.len_elements == 0 {
            return Some(&mut []);
        }
        Some(unsafe { std::slice::from_raw_parts_mut(meta.ptr, meta.len_elements) })
    }

    /// Split-borrow: get 2 non-overlapping mutable f32 slices simultaneously.
    ///
    /// Returns `None` if either buffer doesn't exist or isn't f32.
    ///
    /// # Panics
    ///
    /// Panics if names are the same (logic error — aliasing would be unsound).
    pub fn borrow_2_mut_f32<'a>(
        &'a mut self,
        a: &str,
        b: &str,
    ) -> Option<(&'a mut [f32], &'a mut [f32])> {
        assert_ne!(a, b, "Cannot borrow the same buffer twice mutably");
        let ma = self.meta_checked(a, DType::F32)?;
        let mb = self.meta_checked(b, DType::F32)?;
        // Safety: &mut self → exclusive access. Names are distinct → pointers are
        // to different heap allocations. No aliasing.
        unsafe {
            Some((
                std::slice::from_raw_parts_mut(ma.ptr as *mut f32, ma.len_elements),
                std::slice::from_raw_parts_mut(mb.ptr as *mut f32, mb.len_elements),
            ))
        }
    }

    /// Split-borrow: get 3 non-overlapping mutable f32 slices simultaneously.
    /// This is the key pattern for GEMM: A, B, C all mutable at once.
    ///
    /// Returns `None` if any buffer doesn't exist or isn't f32.
    ///
    /// # Panics
    ///
    /// Panics if any names are the same (logic error — aliasing would be unsound).
    pub fn borrow_3_mut_f32<'a>(
        &'a mut self,
        a: &str,
        b: &str,
        c: &str,
    ) -> Option<(&'a mut [f32], &'a mut [f32], &'a mut [f32])> {
        assert!(a != b && b != c && a != c, "Buffer names must be distinct");
        let ma = self.meta_checked(a, DType::F32)?;
        let mb = self.meta_checked(b, DType::F32)?;
        let mc = self.meta_checked(c, DType::F32)?;
        unsafe {
            Some((
                std::slice::from_raw_parts_mut(ma.ptr as *mut f32, ma.len_elements),
                std::slice::from_raw_parts_mut(mb.ptr as *mut f32, mb.len_elements),
                std::slice::from_raw_parts_mut(mc.ptr as *mut f32, mc.len_elements),
            ))
        }
    }

    /// Split-borrow: get 3 non-overlapping mutable f64 slices simultaneously.
    ///
    /// Returns `None` if any buffer doesn't exist or isn't f64.
    ///
    /// # Panics
    ///
    /// Panics if any names are the same (logic error — aliasing would be unsound).
    pub fn borrow_3_mut_f64<'a>(
        &'a mut self,
        a: &str,
        b: &str,
        c: &str,
    ) -> Option<(&'a mut [f64], &'a mut [f64], &'a mut [f64])> {
        assert!(a != b && b != c && a != c, "Buffer names must be distinct");
        let ma = self.meta_checked(a, DType::F64)?;
        let mb = self.meta_checked(b, DType::F64)?;
        let mc = self.meta_checked(c, DType::F64)?;
        unsafe {
            Some((
                std::slice::from_raw_parts_mut(ma.ptr as *mut f64, ma.len_elements),
                std::slice::from_raw_parts_mut(mb.ptr as *mut f64, mb.len_elements),
                std::slice::from_raw_parts_mut(mc.ptr as *mut f64, mc.len_elements),
            ))
        }
    }

    /// Get the raw pointer and length for a named buffer (for FFI or advanced usage).
    ///
    /// Returns `None` if the buffer doesn't exist.
    ///
    /// # Safety
    ///
    /// The returned `*mut u8` points to a live allocation owned by this `Blackboard`.
    /// The caller must not:
    /// - Dereference the pointer after the buffer is freed or reallocated.
    /// - Create `&mut` slices from the pointer while any `&`-borrow of the same
    ///   buffer is alive (use the typed `get_*_mut` methods instead).
    pub unsafe fn raw_ptr(&self, name: &str) -> Option<(*mut u8, usize, DType)> {
        let meta = self.buffers.get(name)?;
        Some((meta.ptr, meta.len_elements, meta.dtype))
    }

    /// Returns the number of elements in a named buffer.
    ///
    /// Returns `None` if the buffer doesn't exist.
    pub fn len(&self, name: &str) -> Option<usize> {
        let meta = self.buffers.get(name)?;
        Some(meta.len_elements)
    }

    /// Check if a named buffer exists.
    pub fn contains(&self, name: &str) -> bool {
        self.buffers.contains_key(name)
    }

    /// Remove and deallocate a named buffer.
    pub fn free(&mut self, name: &str) {
        self.buffers.remove(name);
    }

    /// List all buffer names.
    pub fn buffer_names(&self) -> Vec<&str> {
        self.buffers.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for Blackboard {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alloc_and_access_f32() {
        let mut bb = Blackboard::new();
        bb.alloc_f32("test", 1024);
        let slice = bb.get_f32_mut("test").unwrap();
        assert_eq!(slice.len(), 1024);
        // Should be zero-initialized
        assert!(slice.iter().all(|&x| x == 0.0));
        slice[0] = 42.0;
        assert_eq!(bb.get_f32("test").unwrap()[0], 42.0);
    }

    #[test]
    fn test_alloc_and_access_f64() {
        let mut bb = Blackboard::new();
        bb.alloc_f64("test", 512);
        let slice = bb.get_f64_mut("test").unwrap();
        assert_eq!(slice.len(), 512);
        slice[0] = 99.0;
        assert_eq!(bb.get_f64("test").unwrap()[0], 99.0);
    }

    #[test]
    fn test_alignment() {
        let mut bb = Blackboard::new();
        bb.alloc_f32("aligned", 256);
        let (ptr, _, _) = unsafe { bb.raw_ptr("aligned") }.unwrap();
        assert_eq!(ptr as usize % ALIGNMENT, 0, "Buffer not 64-byte aligned");
    }

    #[test]
    fn test_split_borrow_3() {
        let mut bb = Blackboard::new();
        bb.alloc_f32("A", 16);
        bb.alloc_f32("B", 16);
        bb.alloc_f32("C", 16);

        let (a, b, c) = bb.borrow_3_mut_f32("A", "B", "C").unwrap();
        a.fill(1.0);
        b.fill(2.0);
        c.fill(0.0);

        // Verify independent
        assert!(a.iter().all(|&x| x == 1.0));
        assert!(b.iter().all(|&x| x == 2.0));
        assert!(c.iter().all(|&x| x == 0.0));
    }

    #[test]
    #[should_panic(expected = "Buffer names must be distinct")]
    fn test_split_borrow_same_name_panics() {
        let mut bb = Blackboard::new();
        bb.alloc_f32("A", 16);
        bb.borrow_3_mut_f32("A", "A", "B");
    }

    #[test]
    fn test_realloc_overwrites() {
        let mut bb = Blackboard::new();
        bb.alloc_f32("buf", 8);
        bb.get_f32_mut("buf").unwrap()[0] = 42.0;
        // Re-allocate with different size
        bb.alloc_f32("buf", 16);
        assert_eq!(bb.len("buf").unwrap(), 16);
        // Should be zero-initialized again
        assert_eq!(bb.get_f32("buf").unwrap()[0], 0.0);
    }

    #[test]
    fn test_free() {
        let mut bb = Blackboard::new();
        bb.alloc_f32("temp", 64);
        assert!(bb.contains("temp"));
        bb.free("temp");
        assert!(!bb.contains("temp"));
    }

    #[test]
    fn test_split_borrow_2() {
        let mut bb = Blackboard::new();
        bb.alloc_f32("X", 8);
        bb.alloc_f32("Y", 8);

        let (x, y) = bb.borrow_2_mut_f32("X", "Y").unwrap();
        x.fill(3.0);
        y.fill(7.0);

        assert!(x.iter().all(|&v| v == 3.0));
        assert!(y.iter().all(|&v| v == 7.0));
    }

    #[test]
    fn test_split_borrow_f64() {
        let mut bb = Blackboard::new();
        bb.alloc_f64("A", 8);
        bb.alloc_f64("B", 8);
        bb.alloc_f64("C", 8);

        let (a, b, c) = bb.borrow_3_mut_f64("A", "B", "C").unwrap();
        a.fill(1.0);
        b.fill(2.0);
        c.fill(3.0);

        assert!(a.iter().all(|&v| v == 1.0));
        assert!(b.iter().all(|&v| v == 2.0));
        assert!(c.iter().all(|&v| v == 3.0));
    }

    #[test]
    fn test_missing_buffer_returns_none() {
        let bb = Blackboard::new();
        assert!(bb.get_f32("nonexistent").is_none());
        assert!(bb.len("nonexistent").is_none());
        assert!(unsafe { bb.raw_ptr("nonexistent") }.is_none());
    }

    #[test]
    fn test_wrong_dtype_returns_none() {
        let mut bb = Blackboard::new();
        bb.alloc_f32("buf", 8);
        assert!(bb.get_f64("buf").is_none());
        assert!(bb.get_u8("buf").is_none());
    }
}
