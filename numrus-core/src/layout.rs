//! CBLAS-style layout and transpose enumerations.
//!
//! Both row-major and column-major layouts are supported throughout the
//! numrus ecosystem. This matches the CBLAS API convention.

/// Memory layout for matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum Layout {
    /// Row-major (C-style): elements in a row are contiguous.
    #[default]
    RowMajor = 101,
    /// Column-major (Fortran-style): elements in a column are contiguous.
    ColMajor = 102,
}

/// Transpose operation for matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum Transpose {
    /// No transpose.
    #[default]
    NoTrans = 111,
    /// Transpose.
    Trans = 112,
    /// Conjugate transpose (for complex types).
    ConjTrans = 113,
}

impl Layout {
    /// Leading dimension stride for an M x N matrix.
    #[inline(always)]
    pub fn leading_dim(self, rows: usize, cols: usize) -> usize {
        match self {
            Layout::RowMajor => cols,
            Layout::ColMajor => rows,
        }
    }

    /// Linear index into a flat array for element (i, j) of an M x N matrix.
    #[inline(always)]
    pub fn index(self, i: usize, j: usize, ld: usize) -> usize {
        match self {
            Layout::RowMajor => i * ld + j,
            Layout::ColMajor => j * ld + i,
        }
    }
}

/// BLAS triangle specifier (upper/lower).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum Uplo {
    #[default]
    Upper = 121,
    Lower = 122,
}

/// BLAS side specifier (left/right multiplication).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum Side {
    #[default]
    Left = 141,
    Right = 142,
}

/// BLAS diagonal specifier (unit/non-unit).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum Diag {
    #[default]
    NonUnit = 131,
    Unit = 132,
}
