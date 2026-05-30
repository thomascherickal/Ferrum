use ferrum_core::{
    loss::softmax_cross_entropy,
    Net, Rng, Sgd, Tensor,
};

#[test]
fn net_parameter_count() {
    let mut rng = Rng::new(42);
    let net = Net::mlp(4, 16, 3, &mut rng);
    // weight1: 4*16 = 64, bias1: 16
    // weight2: 16*3 = 48, bias2: 3
    // Total = 64 + 16 + 48 + 3 = 131
    assert_eq!(net.num_params(), 131);
}

#[test]
fn net_training_iteration() {
    let mut rng = Rng::new(42);
    let mut net = Net::mlp(4, 8, 3, &mut rng);
    let x = Tensor::matrix(1, 4, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let y = vec![1usize];
    let opt = Sgd::with_momentum(0.1, 0.9);

    let logits = net.forward(&x).unwrap();
    let (loss1, dlogits) = softmax_cross_entropy(&logits, &y).unwrap();

    net.backward(&dlogits).unwrap();
    net.step(&opt).unwrap();

    let logits2 = net.forward(&x).unwrap();
    let (loss2, _) = softmax_cross_entropy(&logits2, &y).unwrap();

    assert!(loss2 < loss1, "loss didn't decrease: {loss1} -> {loss2}");
}
