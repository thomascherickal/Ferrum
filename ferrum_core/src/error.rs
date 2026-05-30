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
        let e = InferError::ShapeMismatch {
            expected: 4,
            got: 3,
        };
        assert!(e.to_string().contains("expected 4"));
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
}
