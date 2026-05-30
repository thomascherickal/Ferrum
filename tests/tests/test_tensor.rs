use ferrum_core::Tensor;

#[test]
fn tensor_creation_and_basic_access() {
    let t = Tensor::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    assert_eq!(t.shape, vec![2, 3]);
    assert_eq!(t.numel(), 6);
    assert_eq!(t.at(0, 0), 1.0);
    assert_eq!(t.at(1, 2), 6.0);
}

#[test]
fn tensor_matrix_dims() {
    let t = Tensor::matrix(2, 3, vec![0.0; 6]).unwrap();
    let dims = t.matrix_dims().unwrap();
    assert_eq!(dims, (2, 3));
}

#[test]
fn tensor_vector_creation() {
    let t = Tensor::vector(vec![1.0, 2.0, 3.0]);
    assert_eq!(t.shape, vec![3]);
    assert_eq!(t.numel(), 3);
}

#[test]
fn tensor_row_creation() {
    let t = Tensor::row(vec![1.0, 2.0, 3.0]).unwrap();
    assert_eq!(t.shape, vec![1, 3]);
    assert_eq!(t.numel(), 3);
}
