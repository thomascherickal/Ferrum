use ferrum_core::{
    activation::Activation,
    from_bytes,
    layer::{ActivationLayer, Linear},
    model::Sequential,
    to_bytes, ModelMetadata, Normalizer, TaskType, Tensor,
};

fn make_test_bundle() -> (Sequential, Normalizer, ModelMetadata) {
    let l1 = Linear::new(2, 4, vec![0.1; 8], vec![0.0; 4]).unwrap();
    let model = Sequential::new()
        .with(Box::new(l1))
        .with(Box::new(ActivationLayer::new(Activation::ReLU)));
    let norm = Normalizer {
        means: vec![1.0, 2.0],
        stds: vec![0.5, 0.5],
    };
    let meta = ModelMetadata {
        dataset_name: "test".into(),
        task: TaskType::Classification,
        feature_names: vec!["a".into(), "b".into()],
        feature_ranges: vec![[0.0, 5.0], [0.0, 5.0]],
        class_names: vec!["no".into(), "yes".into()],
        target_name: "".into(),
        target_range: [0.0, 1.0],
        input_dim: 2,
        output_dim: 2,
        tokenizer_state: String::new(),
    };
    (model, norm, meta)
}

#[test]
fn roundtrip_restores_identical_outputs() {
    let (model, norm, meta) = make_test_bundle();
    let bytes = to_bytes(&model, &norm, &meta).unwrap();
    let (model2, norm2, meta2) = from_bytes(&bytes).unwrap();

    assert_eq!(meta2.dataset_name, "test");
    assert_eq!(meta2.task, TaskType::Classification);

    let x = Tensor::matrix(1, 2, vec![1.5, 2.5]).unwrap();
    let y1 = model.forward(&norm.transform(&x).unwrap()).unwrap();
    let y2 = model2.forward(&norm2.transform(&x).unwrap()).unwrap();

    for (a, b) in y1.data.iter().zip(&y2.data) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn wrong_magic_fails() {
    assert!(from_bytes(b"NOT_FINF").is_err());
}

#[test]
fn wrong_version_fails() {
    let (model, norm, meta) = make_test_bundle();
    let mut bytes = to_bytes(&model, &norm, &meta).unwrap();
    // Corrupt the version field at [4..8]
    bytes[4..8].copy_from_slice(&[99, 0, 0, 0]);
    assert!(from_bytes(&bytes).is_err());
}
