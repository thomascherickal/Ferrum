//! Single error type used by every module in ferrum_core.
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum InferError {
    ShapeMismatch {
        expected: usize,
        got: usize,
    },
    DimMismatch(String),
    NotAMatrix(Vec<usize>),
    Io(String),
    Format(String),
    /// Returned by the CSV loader when a value cannot be parsed.
    ParseError(String),
}

impl fmt::Display for InferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InferError::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {expected} elements, got {got}")
            }
            InferError::DimMismatch(m) => write!(f, "dimension mismatch: {m}"),
            InferError::NotAMatrix(s) => write!(f, "expected rank-2 tensor, got shape {s:?}"),
            InferError::Io(m) => write!(f, "i/o error: {m}"),
            InferError::Format(m) => write!(f, "model format error: {m}"),
            InferError::ParseError(m) => write!(f, "parse error: {m}"),
        }
    }
}

impl std::error::Error for InferError {}

impl From<std::io::Error> for InferError {
    fn from(e: std::io::Error) -> Self {
        InferError::Io(e.to_string())
    }
}

impl From<std::num::ParseFloatError> for InferError {
    fn from(e: std::num::ParseFloatError) -> Self {
        InferError::ParseError(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, InferError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_shape_mismatch() {
        let e = InferError::ShapeMismatch { expected: 4, got: 3 };
        let s = e.to_string();
        assert!(s.contains("expected 4") && s.contains("got 3"));
    }

    #[test]
    fn display_dim_mismatch() {
        let e = InferError::DimMismatch("test msg".into());
        assert!(e.to_string().contains("test msg"));
    }

    #[test]
    fn display_not_a_matrix() {
        let e = InferError::NotAMatrix(vec![3]);
        let s = e.to_string();
        assert!(s.contains("rank-2") && s.contains("[3]"));
    }

    #[test]
    fn display_io() {
        let e = InferError::Io("disk full".into());
        assert!(e.to_string().contains("disk full"));
    }

    #[test]
    fn display_format() {
        let e = InferError::Format("bad magic".into());
        assert!(e.to_string().contains("bad magic"));
    }

    #[test]
    fn display_parse_error() {
        let e = InferError::ParseError("not a float".into());
        assert!(e.to_string().contains("not a float"));
    }

    #[test]
    fn io_conversion() {
        use std::io;
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file missing");
        let e: InferError = io_err.into();
        assert!(matches!(e, InferError::Io(_)));
    }

    #[test]
    fn parse_float_conversion() {
        let r: std::result::Result<f32, _> = "not_a_float".parse();
        let e: InferError = r.unwrap_err().into();
        assert!(matches!(e, InferError::ParseError(_)));
    }

    #[test]
    fn infer_error_implements_std_error() {
        // Ensures InferError satisfies the std::error::Error bound.
        let e: Box<dyn std::error::Error> = Box::new(InferError::Format("x".into()));
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn clone_and_eq() {
        let a = InferError::DimMismatch("a".into());
        let b = a.clone();
        assert_eq!(a, b);
    }
}

