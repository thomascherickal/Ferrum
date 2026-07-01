use ferrum_core::{
    activation::Activation,
    layer::{ActivationLayer, Embedding, TransformerBlock},
    model::Sequential,
    Tensor,
};

#[test]
fn test_slm_pipeline_and_causal_attention() {
    let vocab_size = 5;
    let max_seq_len = 4;
    let embedding_dim = 8;
    let hidden_dim = 16;
    let c = embedding_dim;
    let h = hidden_dim;

    let emb = Embedding::new(
        vocab_size,
        max_seq_len,
        embedding_dim,
        vec![0.1; vocab_size * embedding_dim],
        vec![0.01; max_seq_len * embedding_dim],
    )
    .unwrap();

    let tb = TransformerBlock::new(
        max_seq_len,
        2,
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

    let model = Sequential::new()
        .with(Box::new(emb))
        .with(Box::new(tb))
        .with(Box::new(ActivationLayer::new(Activation::Softmax)));

    // Inputs: indices of tokens [0, 1, 2, 3] (length 4)
    let x = Tensor::matrix(1, 4, vec![0.0, 1.0, 2.0, 3.0]).unwrap();
    let out = model.forward(&x).unwrap();

    assert_eq!(out.shape, vec![4, 8]);
    // Softmax output sums to 1 per row, so total sum for 4 rows is 4.0
    let sum: f32 = out.data.iter().sum();
    assert!((sum - 4.0).abs() < 1e-4);

    // Let's retrieve attention weights from tb
    let tb_ref = model.layers()[1]
        .as_any()
        .downcast_ref::<TransformerBlock>()
        .unwrap();
    let att_borrow = tb_ref.last_attention.borrow();
    let att = &*att_borrow;
    // 2 heads, seq_len=4, so 2 * 4 * 4 = 32 weights
    assert_eq!(att.len(), 32);

    // Causal mask: upper triangle of attention matrix in each head must be 0
    // Each head matrix is 4x4.
    // head 0 row 0: [att[0], att[1], att[2], att[3]] => upper triangle elements at col > row must be 0
    for head in 0..2 {
        let offset = head * 16;
        // row 0: col 1, 2, 3 must be 0
        assert_eq!(att[offset + 1], 0.0);
        assert_eq!(att[offset + 2], 0.0);
        assert_eq!(att[offset + 3], 0.0);
        // row 1: col 2, 3 must be 0
        assert_eq!(att[offset + 4 + 2], 0.0);
        assert_eq!(att[offset + 4 + 3], 0.0);
        // row 2: col 3 must be 0
        assert_eq!(att[offset + 8 + 3], 0.0);
    }
}
