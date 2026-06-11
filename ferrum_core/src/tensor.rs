//! The fundamental data structure: a flat `Vec<f32>` with a shape.
use crate::error::{InferError, Result};

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self> {
        let expected: usize = shape.iter().product();
        if expected != data.len() {
            return Err(InferError::ShapeMismatch {
                expected,
                got: data.len(),
            });
        }
        Ok(Self { shape, data })
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let n = shape.iter().product();
        Self {
            shape,
            data: vec![0.0; n],
        }
    }

    pub fn matrix(rows: usize, cols: usize, data: Vec<f32>) -> Result<Self> {
        Self::new(vec![rows, cols], data)
    }

    pub fn vector(data: Vec<f32>) -> Self {
        let n = data.len();
        Self {
            shape: vec![n],
            data,
        }
    }

    pub fn row(data: Vec<f32>) -> Result<Self> {
        let n = data.len();
        Self::matrix(1, n, data)
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn matrix_dims(&self) -> Result<(usize, usize)> {
        match self.shape.as_slice() {
            [r, c] => Ok((*r, *c)),
            _ => Err(InferError::NotAMatrix(self.shape.clone())),
        }
    }

    pub fn at(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.shape[1] + c]
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Result<Tensor> {
        Tensor::new(shape, self.data.clone())
    }

    pub fn map<F: Fn(f32) -> f32>(&self, f: F) -> Tensor {
        Tensor {
            shape: self.shape.clone(),
            data: self.data.iter().copied().map(f).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_validates_element_count() {
        assert!(Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).is_ok());
        let err = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0]).unwrap_err();
        assert!(matches!(
            err,
            InferError::ShapeMismatch {
                expected: 4,
                got: 3
            }
        ));
    }

    #[test]
    fn zeros_has_correct_shape_and_data() {
        let t = Tensor::zeros(vec![3, 2]);
        assert_eq!(t.shape, vec![3, 2]);
        assert!(t.data.iter().all(|&x| x == 0.0));
        assert_eq!(t.numel(), 6);
    }

    #[test]
    fn at_indexes_row_major() {
        // [[0,1,2],[3,4,5]]
        let m = Tensor::matrix(2, 3, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert_eq!(m.at(0, 2), 2.0);
        assert_eq!(m.at(1, 0), 3.0);
        assert_eq!(m.at(1, 2), 5.0);
    }

    #[test]
    fn matrix_dims_rejects_non_matrix() {
        let v = Tensor::vector(vec![1.0, 2.0, 3.0]);
        assert!(matches!(v.matrix_dims(), Err(InferError::NotAMatrix(_))));
    }

    #[test]
    fn map_transforms_and_preserves_shape() {
        let t = Tensor::vector(vec![1.0, -2.0, 3.0]);
        let out = t.map(|x| x * 2.0);
        assert_eq!(out.data, vec![2.0, -4.0, 6.0]);
        assert_eq!(out.shape, t.shape);
    }

    #[test]
    fn reshape_checks_element_count() {
        let t = Tensor::matrix(2, 3, vec![0.0; 6]).unwrap();
        assert!(t.reshape(vec![3, 2]).is_ok());
        assert!(t.reshape(vec![2, 2]).is_err());
    }

    #[test]
    fn row_constructor() {
        let r = Tensor::row(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(r.shape, vec![1, 3]);
    }
}
