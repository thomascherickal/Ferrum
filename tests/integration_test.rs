//! End-to-end integration tests — updated for ferrum_core v0.2 API.
//!
//! Pipeline under test:
//!   CSV parse → normalise → train → serialise (FINF v3) → deserialise → infer

use ferrum_core::{
    accuracy, argmax_rows, from_bytes, mse, softmax_cross_entropy, to_bytes, train_epoch,
    train_val_split, CsvDataset, ModelMetadata, Net, Normalizer, Rng, Sgd, TaskType, Tensor,
};

const IRIS_CSV: &str = include_str!("../iris.data");
const REGRESSION_CSV: &str = "\
x1,x2,x3,price
1.0,2.0,3.0,100000.0
2.0,3.0,4.0,200000.0
3.0,4.0,5.0,300000.0
4.0,5.0,6.0,400000.0
5.0,6.0,7.0,500000.0
6.0,7.0,8.0,600000.0
7.0,8.0,9.0,700000.0
8.0,9.0,10.0,800000.0
9.0,10.0,11.0,900000.0
10.0,11.0,12.0,1000000.0
";

// ── Helpers ──────────────────────────────────────────────────────────────────

fn load_iris() -> CsvDataset {
    CsvDataset::from_str(IRIS_CSV).expect("iris.data must parse")
}

fn make_meta(ds: &CsvDataset) -> ModelMetadata {
    ModelMetadata {
        dataset_name: ds
            .class_names
            .first()
            .map(|_| "Iris".into())
            .unwrap_or_default(),
        task: ds.task,
        feature_names: ds.feature_names.clone(),
        feature_ranges: ds.feature_ranges.clone(),
        class_names: ds.class_names.clone(),
        target_name: String::new(),
        target_range: ds.target_range,
        input_dim: ds.num_features,
        output_dim: ds.num_classes,
    }
}

/// Train a small net for 300 epochs and return it along with the fitted normalizer.
fn trained_iris() -> (Net, Normalizer, CsvDataset) {
    let ds = load_iris();
    let mut rng = Rng::new(42);
    let (x_raw, y_cls, _) = ds.to_tensors().unwrap();
    let norm = Normalizer::fit(&x_raw).unwrap();
    let x = norm.transform(&x_raw).unwrap();
    let mut net = Net::mlp(ds.num_features, 32, ds.num_classes, &mut rng);
    let opt = Sgd::with_momentum(0.05, 0.9);
    for _ in 0..300 {
        train_epoch(&mut net, &x, &y_cls, 16, &opt, &mut rng).unwrap();
    }
    (net, norm, ds)
}

// ── Dataset tests ─────────────────────────────────────────────────────────────

#[test]
fn iris_row_count_is_150() {
    assert_eq!(load_iris().rows.len(), 150);
}

#[test]
fn iris_has_four_features() {
    assert_eq!(load_iris().num_features, 4);
}

#[test]
fn iris_has_three_classes() {
    let ds = load_iris();
    assert_eq!(ds.num_classes, 3);
    assert!(ds.class_names.iter().any(|n| n.contains("setosa")));
    assert!(ds.class_names.iter().any(|n| n.contains("versicolor")));
    assert!(ds.class_names.iter().any(|n| n.contains("virginica")));
}

#[test]
fn iris_task_is_classification() {
    assert_eq!(load_iris().task, TaskType::Classification);
}

#[test]
fn first_row_is_setosa() {
    let ds = load_iris();
    let row = &ds.rows[0];
    assert_eq!(row.features, vec![5.1, 3.5, 1.4, 0.2]);
    assert_eq!(ds.class_names[row.label], "Iris-setosa");
}

#[test]
fn feature_names_read_from_header() {
    let ds = load_iris();
    assert_eq!(ds.feature_names[0], "sepal_length");
    assert_eq!(ds.feature_names[3], "petal_width");
}

#[test]
fn to_tensors_returns_3_tuple() {
    let ds = load_iris();
    let (x, labels, targets) = ds.to_tensors().unwrap();
    assert_eq!(x.shape, vec![150, 4]);
    assert_eq!(labels.len(), 150);
    assert_eq!(targets.len(), 150);
}

#[test]
fn label_indices_in_range() {
    let ds = load_iris();
    for row in &ds.rows {
        assert!(row.label < ds.num_classes);
    }
}

#[test]
fn feature_ranges_cover_data() {
    let ds = load_iris();
    // sepal_length: known [4.3, 7.9]
    assert!(ds.feature_ranges[0][0] <= 4.3 + 0.1);
    assert!(ds.feature_ranges[0][1] >= 7.9 - 0.1);
}

#[test]
fn regression_dataset_detected() {
    let ds = CsvDataset::from_str(REGRESSION_CSV).unwrap();
    assert_eq!(ds.task, TaskType::Regression);
    assert_eq!(ds.num_features, 3);
    assert!((ds.rows[0].target - 100_000.0).abs() < 1.0);
}

// ── Normalizer tests ──────────────────────────────────────────────────────────

#[test]
fn normalizer_produces_zero_mean() {
    let ds = load_iris();
    let (x_raw, _, _) = ds.to_tensors().unwrap();
    let norm = Normalizer::fit(&x_raw).unwrap();
    let z = norm.transform(&x_raw).unwrap();
    let (n, cols) = z.matrix_dims().unwrap();
    for c in 0..cols {
        let mean: f32 = (0..n).map(|r| z.at(r, c)).sum::<f32>() / n as f32;
        assert!(mean.abs() < 1e-4, "col {c} mean = {mean:.6}");
    }
}

#[test]
fn normalizer_produces_unit_variance() {
    let ds = load_iris();
    let (x_raw, _, _) = ds.to_tensors().unwrap();
    let norm = Normalizer::fit(&x_raw).unwrap();
    let z = norm.transform(&x_raw).unwrap();
    let (n, cols) = z.matrix_dims().unwrap();
    for c in 0..cols {
        let mean: f32 = (0..n).map(|r| z.at(r, c)).sum::<f32>() / n as f32;
        let var: f32 = (0..n).map(|r| (z.at(r, c) - mean).powi(2)).sum::<f32>() / n as f32;
        assert!((var - 1.0).abs() < 0.05, "col {c} var = {var:.4}");
    }
}

#[test]
fn normalizer_encode_decode_roundtrip() {
    let ds = load_iris();
    let (x_raw, _, _) = ds.to_tensors().unwrap();
    let n1 = Normalizer::fit(&x_raw).unwrap();
    let n2 = Normalizer::decode(&n1.encode()).unwrap();
    for (a, b) in n1.means.iter().zip(&n2.means) {
        assert!((a - b).abs() < 1e-6);
    }
    for (a, b) in n1.stds.iter().zip(&n2.stds) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn normalizer_transform_row_works() {
    let ds = load_iris();
    let (x_raw, _, _) = ds.to_tensors().unwrap();
    let norm = Normalizer::fit(&x_raw).unwrap();
    let row = norm.transform_row(&[5.1, 3.5, 1.4, 0.2]).unwrap();
    assert_eq!(row.shape, vec![1, 4]);
}

// ── Train / val split tests ───────────────────────────────────────────────────

#[test]
fn split_covers_all_examples() {
    let ds = load_iris();
    let mut rng = Rng::new(1);
    let (tr, va) = train_val_split(&ds, 0.2, &mut rng);
    assert_eq!(tr.rows.len() + va.rows.len(), 150);
}

#[test]
fn split_preserves_metadata() {
    let ds = load_iris();
    let mut rng = Rng::new(2);
    let (tr, _) = train_val_split(&ds, 0.2, &mut rng);
    assert_eq!(tr.num_classes, ds.num_classes);
    assert_eq!(tr.num_features, ds.num_features);
    assert_eq!(tr.class_names, ds.class_names);
    assert_eq!(tr.feature_names, ds.feature_names);
}

// ── Training tests ────────────────────────────────────────────────────────────

#[test]
fn initial_loss_near_log_vocab() {
    let ds = load_iris();
    let (x_raw, y_cls, _) = ds.to_tensors().unwrap();
    let norm = Normalizer::fit(&x_raw).unwrap();
    let x = norm.transform(&x_raw).unwrap();
    let mut rng = Rng::new(7);
    let mut net = Net::mlp(4, 16, 3, &mut rng);
    let (loss, _) = softmax_cross_entropy(&net.forward(&x).unwrap(), &y_cls).unwrap();
    assert!(
        loss > 0.8 && loss < 2.5,
        "initial loss {loss:.4} out of range"
    );
}

#[test]
fn training_achieves_90pct_accuracy() {
    let ds = load_iris();
    let (x_raw, y_cls, _) = ds.to_tensors().unwrap();
    let norm = Normalizer::fit(&x_raw).unwrap();
    let x = norm.transform(&x_raw).unwrap();
    let mut rng = Rng::new(1337);
    let mut net = Net::mlp(4, 32, 3, &mut rng);
    let opt = Sgd::with_momentum(0.05, 0.9);
    for _ in 0..500 {
        train_epoch(&mut net, &x, &y_cls, 16, &opt, &mut rng).unwrap();
    }
    let acc = accuracy(&mut net, &x, &y_cls).unwrap();
    assert!(acc >= 0.90, "accuracy {:.1}% < 90%", acc * 100.0);
}

#[test]
fn loss_decreases_over_training() {
    let ds = load_iris();
    let (x_raw, y_cls, _) = ds.to_tensors().unwrap();
    let norm = Normalizer::fit(&x_raw).unwrap();
    let x = norm.transform(&x_raw).unwrap();
    let mut rng = Rng::new(55);
    let mut net = Net::mlp(4, 16, 3, &mut rng);
    let opt = Sgd::with_momentum(0.05, 0.9);
    let (l0, _) = softmax_cross_entropy(&net.forward(&x).unwrap(), &y_cls).unwrap();
    for _ in 0..200 {
        train_epoch(&mut net, &x, &y_cls, 16, &opt, &mut rng).unwrap();
    }
    let (l1, _) = softmax_cross_entropy(&net.forward(&x).unwrap(), &y_cls).unwrap();
    assert!(l1 < l0 * 0.5, "loss {l0:.4} → {l1:.4}: did not halve");
}

#[test]
fn mse_loss_decreases_for_regression() {
    let ds = CsvDataset::from_str(REGRESSION_CSV).unwrap();
    let (x_raw, _, y_reg) = ds.to_tensors().unwrap();
    let norm = ferrum_core::fit_normalizer_with_target(&x_raw, &y_reg).unwrap();
    let x = norm.transform(&x_raw).unwrap();
    let y_norm: Vec<f32> = y_reg.iter().map(|&v| norm.normalise_target(v)).collect();
    let mut rng = Rng::new(9);
    let mut net = Net::mlp(3, 8, 1, &mut rng);
    let opt = Sgd::with_momentum(0.01, 0.9);
    let (l0, _) = mse(&net.forward(&x).unwrap(), &y_norm).unwrap();
    for _ in 0..300 {
        let logits = net.forward(&x).unwrap();
        let (_, dl) = mse(&logits, &y_norm).unwrap();
        net.backward(&dl).unwrap();
        net.step(&opt).unwrap();
    }
    let (l1, _) = mse(&net.forward(&x).unwrap(), &y_norm).unwrap();
    assert!(
        l1 < l0 * 0.5,
        "regression loss {l0:.4} → {l1:.4}: did not halve"
    );
}

// ── Serialisation / loader tests (FINF v3) ────────────────────────────────────

#[test]
fn serialise_deserialise_identical_outputs() {
    let (net, norm, ds) = trained_iris();
    let model = net.to_inference().unwrap();
    let meta = make_meta(&ds);
    let bytes = to_bytes(&model, &norm, &meta).unwrap();
    let (model2, norm2, _meta2) = from_bytes(&bytes).unwrap();

    let probe_raw = Tensor::row(vec![5.1f32, 3.5, 1.4, 0.2]).unwrap();
    let p1 = model.forward(&norm.transform(&probe_raw).unwrap()).unwrap();
    let p2 = model2
        .forward(&norm2.transform(&probe_raw).unwrap())
        .unwrap();

    for (a, b) in p1.data.iter().zip(&p2.data) {
        assert!((a - b).abs() < 1e-6, "outputs differ: {a} vs {b}");
    }
}

#[test]
fn metadata_survives_serialisation() {
    let (net, norm, ds) = trained_iris();
    let model = net.to_inference().unwrap();
    let meta = make_meta(&ds);
    let bytes = to_bytes(&model, &norm, &meta).unwrap();
    let (_, _, meta2) = from_bytes(&bytes).unwrap();
    assert_eq!(meta2.task, TaskType::Classification);
    assert_eq!(meta2.feature_names, ds.feature_names);
    assert_eq!(meta2.class_names, ds.class_names);
    assert_eq!(meta2.input_dim, 4);
    assert_eq!(meta2.output_dim, 3);
}

#[test]
fn model_bytes_under_16kb() {
    let (net, norm, ds) = trained_iris();
    let model = net.to_inference().unwrap();
    let meta = make_meta(&ds);
    let bytes = to_bytes(&model, &norm, &meta).unwrap();
    assert!(bytes.len() < 16_384, "model {} bytes > 16 KB", bytes.len());
}

#[test]
fn corrupt_bytes_rejected() {
    assert!(from_bytes(b"NOT_A_FINF_FILE").is_err());
    assert!(from_bytes(&[]).is_err());
}

#[test]
fn wrong_finf_version_rejected() {
    let (net, norm, ds) = trained_iris();
    let model = net.to_inference().unwrap();
    let meta = make_meta(&ds);
    let mut bytes = to_bytes(&model, &norm, &meta).unwrap();
    // Overwrite version u32 at offset 4 with 99
    bytes[4] = 99;
    bytes[5] = 0;
    bytes[6] = 0;
    bytes[7] = 0;
    assert!(from_bytes(&bytes).is_err());
}

// ── Known-sample inference tests (uses live trained models) ───────────────────

fn try_load_dataset_model(
    path: &str,
) -> Option<(ferrum_core::Sequential, Normalizer, ModelMetadata)> {
    std::fs::read(path).ok().and_then(|b| from_bytes(&b).ok())
}

#[test]
fn iris_setosa_sample_classifies_correctly() {
    let Some((model, norm, meta)) = try_load_dataset_model("web/datasets/iris/model.bin") else {
        return;
    };
    assert_eq!(meta.task, TaskType::Classification);
    let raw = Tensor::row(vec![5.1f32, 3.5, 1.4, 0.2]).unwrap();
    let input = norm.transform(&raw).unwrap();
    let probs = model.forward(&input).unwrap();
    assert_eq!(
        argmax_rows(&probs).unwrap()[0],
        0,
        "expected Iris-setosa (class 0)"
    );
}

#[test]
fn iris_virginica_sample_classifies_correctly() {
    let Some((model, norm, _)) = try_load_dataset_model("web/datasets/iris/model.bin") else {
        return;
    };
    let raw = Tensor::row(vec![6.3f32, 3.3, 6.0, 2.5]).unwrap();
    let input = norm.transform(&raw).unwrap();
    let probs = model.forward(&input).unwrap();
    assert_eq!(
        argmax_rows(&probs).unwrap()[0],
        2,
        "expected Iris-virginica (class 2)"
    );
}

#[test]
fn all_dataset_models_load_and_produce_finite_outputs() {
    let paths = [
        ("web/datasets/iris/model.bin", 4usize),
        ("web/datasets/wine/model.bin", 11),
        ("web/datasets/diabetes/model.bin", 8),
        ("web/datasets/titanic/model.bin", 6),
        ("web/datasets/housing/model.bin", 8),
    ];
    for (path, n_features) in &paths {
        let Some((model, norm, meta)) = try_load_dataset_model(path) else {
            continue;
        };
        assert_eq!(meta.input_dim, *n_features, "input_dim mismatch for {path}");
        let raw = Tensor::row(vec![0.5f32; *n_features]).unwrap();
        let input = norm.transform(&raw).unwrap();
        let out = model.forward(&input).unwrap();
        assert!(
            out.data.iter().all(|v| v.is_finite()),
            "NaN/Inf output for {path}"
        );
        println!("  {path}: output {:?}", &out.data[..out.data.len().min(3)]);
    }
}

#[test]
fn classification_outputs_sum_to_one() {
    let clf_paths = [
        "web/datasets/iris/model.bin",
        "web/datasets/wine/model.bin",
        "web/datasets/diabetes/model.bin",
        "web/datasets/titanic/model.bin",
    ];
    for path in &clf_paths {
        let Some((model, norm, meta)) = try_load_dataset_model(path) else {
            continue;
        };
        assert_eq!(
            meta.task,
            TaskType::Classification,
            "{path} should be classification"
        );
        let raw = Tensor::row(vec![0.5f32; meta.input_dim]).unwrap();
        let input = norm.transform(&raw).unwrap();
        let out = model.forward(&input).unwrap();
        let sum: f32 = out.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "{path}: probs sum = {sum:.6}");
    }
}

#[test]
fn regression_output_is_scalar() {
    let Some((model, norm, meta)) = try_load_dataset_model("web/datasets/housing/model.bin") else {
        return;
    };
    assert_eq!(meta.task, TaskType::Regression);
    let raw = Tensor::row(vec![0.5f32; meta.input_dim]).unwrap();
    let input = norm.transform(&raw).unwrap();
    let out = model.forward(&input).unwrap();
    assert_eq!(out.shape, vec![1, 1]);
    let pred_raw = norm.denormalise_target(out.data[0]);
    // Should be in some plausible range for CA housing (say $10k – $10M)
    assert!(
        pred_raw > 10_000.0 && pred_raw < 10_000_000.0,
        "housing prediction {pred_raw:.0} out of plausible range"
    );
}

#[test]
fn batch_inference_matches_individual() {
    let Some((model, norm, _)) = try_load_dataset_model("web/datasets/iris/model.bin") else {
        return;
    };
    let samples = vec![
        vec![5.1f32, 3.5, 1.4, 0.2],
        vec![6.4, 3.2, 4.5, 1.5],
        vec![6.3, 3.3, 6.0, 2.5],
    ];
    let individual: Vec<usize> = samples
        .iter()
        .map(|f| {
            let p = model
                .forward(&norm.transform(&Tensor::row(f.clone()).unwrap()).unwrap())
                .unwrap();
            argmax_rows(&p).unwrap()[0]
        })
        .collect();
    let mut all_data = Vec::new();
    for s in &samples {
        all_data.extend_from_slice(s);
    }
    let batch_in = norm
        .transform(&Tensor::matrix(3, 4, all_data).unwrap())
        .unwrap();
    let batch_out = model.forward(&batch_in).unwrap();
    assert_eq!(individual, argmax_rows(&batch_out).unwrap());
}

// ── New dataset tests ─────────────────────────────────────────────────────────

#[test]
fn heart_dataset_parses() {
    let csv = std::fs::read_to_string("heart.csv").unwrap_or_default();
    if csv.is_empty() {
        return;
    }
    let ds = CsvDataset::from_str(&csv).unwrap();
    assert_eq!(ds.task, TaskType::Classification);
    assert_eq!(ds.num_features, 13);
    assert_eq!(ds.num_classes, 2);
    assert!(ds.rows.len() > 250);
}

#[test]
fn cancer_dataset_parses() {
    let csv = std::fs::read_to_string("cancer.csv").unwrap_or_default();
    if csv.is_empty() {
        return;
    }
    let ds = CsvDataset::from_str(&csv).unwrap();
    assert_eq!(ds.task, TaskType::Classification);
    assert_eq!(ds.num_features, 30);
    assert_eq!(ds.num_classes, 2);
}

#[test]
fn penguins_dataset_parses() {
    let csv = std::fs::read_to_string("penguins.csv").unwrap_or_default();
    if csv.is_empty() {
        return;
    }
    let ds = CsvDataset::from_str(&csv).unwrap();
    assert_eq!(ds.task, TaskType::Classification);
    assert_eq!(ds.num_features, 4);
    assert_eq!(ds.num_classes, 3);
    assert!(ds.class_names.iter().any(|n| n == "Adelie"));
    assert!(ds.class_names.iter().any(|n| n == "Gentoo"));
}

#[test]
fn mpg_dataset_parses_as_regression() {
    let csv = std::fs::read_to_string("mpg.csv").unwrap_or_default();
    if csv.is_empty() {
        return;
    }
    let ds = CsvDataset::from_str(&csv).unwrap();
    assert_eq!(ds.task, TaskType::Regression);
    assert_eq!(ds.num_features, 6);
    assert!(ds.rows.len() > 380);
}

#[test]
fn seeds_dataset_parses() {
    let csv = std::fs::read_to_string("seeds.csv").unwrap_or_default();
    if csv.is_empty() {
        return;
    }
    let ds = CsvDataset::from_str(&csv).unwrap();
    assert_eq!(ds.task, TaskType::Classification);
    assert_eq!(ds.num_features, 7);
    assert_eq!(ds.num_classes, 3);
    assert!(ds.class_names.iter().any(|n| n == "Kama"));
}

#[test]
fn all_ten_model_files_load_and_produce_finite_outputs() {
    let configs: &[(&str, usize)] = &[
        ("web/datasets/iris/model.bin", 4),
        ("web/datasets/wine/model.bin", 11),
        ("web/datasets/diabetes/model.bin", 8),
        ("web/datasets/titanic/model.bin", 6),
        ("web/datasets/housing/model.bin", 8),
        ("web/datasets/heart/model.bin", 13),
        ("web/datasets/cancer/model.bin", 30),
        ("web/datasets/penguins/model.bin", 4),
        ("web/datasets/mpg/model.bin", 6),
        ("web/datasets/seeds/model.bin", 7),
    ];
    let mut loaded = 0;
    for (path, n_features) in configs {
        let Some((model, norm, meta)) = try_load_dataset_model(path) else {
            continue;
        };
        assert_eq!(meta.input_dim, *n_features, "input_dim wrong for {path}");
        let raw = Tensor::row(vec![0.5f32; *n_features]).unwrap();
        let input = norm.transform(&raw).unwrap();
        let out = model.forward(&input).unwrap();
        assert!(
            out.data.iter().all(|v| v.is_finite()),
            "NaN/Inf output for {path}: {:?}",
            out.data
        );
        loaded += 1;
    }
    println!(
        "Loaded and tested {loaded}/{} dataset models",
        configs.len()
    );
}

#[test]
fn all_classification_models_output_valid_distributions() {
    let clf_paths: &[(&str, usize)] = &[
        ("web/datasets/iris/model.bin", 4),
        ("web/datasets/wine/model.bin", 11),
        ("web/datasets/diabetes/model.bin", 8),
        ("web/datasets/titanic/model.bin", 6),
        ("web/datasets/heart/model.bin", 13),
        ("web/datasets/cancer/model.bin", 30),
        ("web/datasets/penguins/model.bin", 4),
        ("web/datasets/seeds/model.bin", 7),
    ];
    for (path, n) in clf_paths {
        let Some((model, norm, meta)) = try_load_dataset_model(path) else {
            continue;
        };
        assert_eq!(meta.task, TaskType::Classification, "{path}");
        for probe in [vec![0.3f32; *n], vec![0.7f32; *n]] {
            let raw = Tensor::row(probe).unwrap();
            let input = norm.transform(&raw).unwrap();
            let out = model.forward(&input).unwrap();
            // All probabilities in [0,1]
            assert!(
                out.data.iter().all(|&p| p >= -1e-6 && p <= 1.0 + 1e-6),
                "{path}: probability out of [0,1]: {:?}",
                out.data
            );
            // Sum to 1
            let sum: f32 = out.data.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "{path}: probs sum {sum}");
            // At least one probability > threshold (not all-zero)
            assert!(
                out.data.iter().any(|&p| p > 0.01),
                "{path}: all probabilities near zero"
            );
        }
    }
}

#[test]
fn both_regression_models_produce_plausible_values() {
    let reg_configs: &[(&str, usize, f32, f32)] = &[
        ("web/datasets/housing/model.bin", 8, 10_000.0, 10_000_000.0),
        ("web/datasets/mpg/model.bin", 6, 5.0, 60.0),
    ];
    for (path, n, lo, hi) in reg_configs {
        let Some((model, norm, meta)) = try_load_dataset_model(path) else {
            continue;
        };
        assert_eq!(meta.task, TaskType::Regression, "{path}");
        let raw = Tensor::row(vec![0.5f32; *n]).unwrap();
        let input = norm.transform(&raw).unwrap();
        let out = model.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 1], "{path}: output not scalar");
        let pred = norm.denormalise_target(out.data[0]);
        assert!(pred.is_finite(), "{path}: NaN/Inf prediction");
        assert!(
            pred > *lo && pred < *hi,
            "{path}: prediction {pred:.2} outside [{lo},{hi}]"
        );
    }
}
