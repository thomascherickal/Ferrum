use ferrum_core::{
    activation::Activation,
    layer::{ActivationLayer, Embedding, Layer, LayerNorm, Linear, TransformerBlock},
    Tensor,
};

#[test]
fn test_linear_forward() {
    let lin = Linear::new(
        2,
        3,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![0.1, 0.2, 0.3],
    )
    .unwrap();
    let x = Tensor::matrix(1, 2, vec![0.5, 1.5]).unwrap();
    let out = lin.forward(&x).unwrap();
    assert_eq!(out.shape, vec![1, 3]);
    // y = [0.5, 1.5] * [1 2 3; 4 5 6] + [0.1, 0.2, 0.3]
    // col 0: 0.5*1 + 1.5*4 + 0.1 = 0.5 + 6.0 + 0.1 = 6.6
    // col 1: 0.5*2 + 1.5*5 + 0.2 = 1.0 + 7.5 + 0.2 = 8.7
    // col 2: 0.5*3 + 1.5*6 + 0.3 = 1.5 + 9.0 + 0.3 = 10.8
    assert!((out.data[0] - 6.6).abs() < 1e-5);
    assert!((out.data[1] - 8.7).abs() < 1e-5);
    assert!((out.data[2] - 10.8).abs() < 1e-5);
}

#[test]
fn test_activation_forward() {
    let relu = ActivationLayer::new(Activation::ReLU);
    let x = Tensor::matrix(1, 4, vec![-1.0, 0.0, 1.0, 2.0]).unwrap();
    let out = relu.forward(&x).unwrap();
    assert_eq!(out.data, vec![0.0, 0.0, 1.0, 2.0]);
}

#[test]
fn test_layernorm_forward() {
    let ln = LayerNorm::new(4, vec![1.0; 4], vec![0.0; 4]).unwrap();
    let x = Tensor::matrix(1, 4, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let out = ln.forward(&x).unwrap();
    assert_eq!(out.shape, vec![1, 4]);
    // Means should be 0, stds should be 1
    let sum: f32 = out.data.iter().sum();
    let mean = sum / 4.0;
    assert!(mean.abs() < 1e-5);
}

#[test]
fn test_embedding_forward() {
    let emb = Embedding::new(5, 4, 3, vec![0.1; 15], vec![0.01; 12]).unwrap();
    let x = Tensor::matrix(1, 4, vec![0.0, 1.0, 2.0, 3.0]).unwrap();
    let out = emb.forward(&x).unwrap();
    assert_eq!(out.shape, vec![4, 3]);
}

#[test]
fn test_transformer_block_forward() {
    let context_len = 4;
    let num_heads = 2;
    let embedding_dim = 8;
    let hidden_dim = 16;
    let c = embedding_dim;
    let h = hidden_dim;

    let tb = TransformerBlock::new(
        context_len,
        num_heads,
        embedding_dim,
        vec![1.0; c],
        vec![0.0; c],
        vec![0.1; c * c],
        vec![0.0; c],
        vec![0.1; c * c],
        vec![0.0; c],
        vec![0.1; c * c],
        vec![0.0; c],
        vec![0.1; c * c],
        vec![0.0; c],
        vec![1.0; c],
        vec![0.0; c],
        vec![0.1; c * h],
        vec![0.0; h],
        vec![0.1; h * c],
        vec![0.0; c],
    )
    .unwrap();

    let x = Tensor::matrix(4, embedding_dim, vec![0.5; 4 * embedding_dim]).unwrap();
    let out = tb.forward(&x).unwrap();
    assert_eq!(out.shape, vec![4, embedding_dim]);
    assert!(out.data.iter().all(|&v| v.is_finite()));
}
