use ferrum_core::{
    fit_normalizer_with_target, mse, CsvDataset, Net, Rng, Sgd, TaskType,
};

const REGRESSION_CSV: &str = "\
x1,x2,price
1.0,2.0,100000.0
2.0,3.0,200000.0
3.0,4.0,300000.0
4.0,5.0,400000.0
";

#[test]
fn test_regression_pipeline() {
    let ds = CsvDataset::from_str(REGRESSION_CSV).unwrap();
    assert_eq!(ds.task, TaskType::Regression);

    let (x_raw, _, y_reg) = ds.to_tensors().unwrap();
    let norm = fit_normalizer_with_target(&x_raw, &y_reg).unwrap();
    let x = norm.transform(&x_raw).unwrap();
    let y_norm: Vec<f32> = y_reg.iter().map(|&v| norm.normalise_target(v)).collect();

    let mut rng = Rng::new(42);
    let mut net = Net::mlp(2, 8, 1, &mut rng);
    let opt = Sgd::with_momentum(0.01, 0.9);

    // Initial loss
    let (l0, _) = mse(&net.forward(&x).unwrap(), &y_norm).unwrap();

    // Train minibatches
    for _ in 0..100 {
        let logits = net.forward(&x).unwrap();
        let (_, dl) = mse(&logits, &y_norm).unwrap();
        net.backward(&dl).unwrap();
        net.step(&opt).unwrap();
    }

    // Final loss should be lower
    let (l1, _) = mse(&net.forward(&x).unwrap(), &y_norm).unwrap();
    assert!(l1 < l0, "regression training didn't reduce loss: {l0} -> {l1}");
}
