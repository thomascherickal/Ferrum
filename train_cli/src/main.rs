//! Generic tabular trainer. Accepts any CSV, auto-detects task, trains, exports FINF.
//!
//! Usage:
//!   train_cli <csv_path> <model_output.bin> [dataset_name] [hidden_size] [epochs]
//!
//! Examples:
//!   train_cli iris.data          web/datasets/iris/model.bin      "Iris"        32  500
//!   train_cli wine.csv           web/datasets/wine/model.bin      "Wine"        64  600
//!   train_cli diabetes.csv       web/datasets/diabetes/model.bin  "Diabetes"    48  600
//!   train_cli titanic.csv        web/datasets/titanic/model.bin   "Titanic"     32  500
//!   train_cli housing.csv        web/datasets/housing/model.bin   "Housing"     64  400

use ferrum_core::{
    accuracy, argmax_rows, fit_normalizer_with_target, mse, softmax_cross_entropy, to_bytes,
    train_epoch, train_val_split, CsvDataset, ModelMetadata, Net, Normalizer, Rng, Sgd, TaskType,
    Tensor,
};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: train_cli <csv> <model.bin> [name] [hidden] [epochs]");
        std::process::exit(1);
    }
    let csv_path = &args[1];
    let model_path = &args[2];
    let ds_name = args.get(3).map(|s| s.as_str()).unwrap_or("Dataset");
    let hidden: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(64);
    let epochs: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(500);

    // ── Load ─────────────────────────────────────────────────────────────────
    let raw =
        std::fs::read_to_string(csv_path).map_err(|e| format!("cannot read {csv_path}: {e}"))?;
    let ds = CsvDataset::from_str(&raw)?;
    if ds.task == TaskType::TransformerSLM {
        return Err("TransformerSLM is not supported by the tabular train_cli. Use train_transformer instead.".into());
    }
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("  Dataset : {ds_name}");
    println!("  File    : {csv_path}");
    println!(
        "  Rows    : {}  Features: {}  Task: {:?}",
        ds.len(),
        ds.num_features,
        ds.task
    );
    if ds.task == TaskType::Classification {
        println!("  Classes : {:?}", ds.class_names);
    }
    println!("  Features: {:?}", ds.feature_names);

    // ── Split ─────────────────────────────────────────────────────────────────
    let mut rng = Rng::new(1337);
    let (train_ds, val_ds) = train_val_split(&ds, 0.20, &mut rng);
    println!(
        "  Split   : {} train / {} val",
        train_ds.len(),
        val_ds.len()
    );

    // ── Normalise ─────────────────────────────────────────────────────────────
    let (x_train_raw, y_train_cls, y_train_reg) = train_ds.to_tensors()?;
    let (x_val_raw, y_val_cls, y_val_reg) = val_ds.to_tensors()?;

    let norm = match ds.task {
        TaskType::Regression => fit_normalizer_with_target(&x_train_raw, &y_train_reg)?,
        TaskType::Classification => Normalizer::fit(&x_train_raw)?,
        TaskType::TransformerSLM => unreachable!(),
    };
    let x_train = norm.transform(&x_train_raw)?;
    let x_val = norm.transform(&x_val_raw)?;

    // For regression: normalise targets too
    let y_train_norm: Vec<f32> = y_train_reg
        .iter()
        .map(|&v| norm.normalise_target(v))
        .collect();
    let y_val_norm: Vec<f32> = y_val_reg
        .iter()
        .map(|&v| norm.normalise_target(v))
        .collect();

    // ── Network ───────────────────────────────────────────────────────────────
    let output_dim = match ds.task {
        TaskType::Classification => ds.num_classes,
        TaskType::Regression => 1,
        TaskType::TransformerSLM => unreachable!(),
    };
    let mut net = Net::mlp(ds.num_features, hidden, output_dim, &mut rng);
    println!(
        "  Network : {} → {} → {}  ({} params)",
        ds.num_features,
        hidden,
        output_dim,
        net.num_params()
    );
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // ── Train ─────────────────────────────────────────────────────────────────
    let lr = if ds.task == TaskType::Regression {
        0.01
    } else {
        0.05
    };
    let opt = Sgd::with_momentum(lr, 0.9);
    let batch = 32.min(train_ds.len());
    let t0 = Instant::now();

    println!(
        "{:>6}  {:>10}  {:>10}  {:>10}",
        "epoch", "train_loss", "val_loss", "metric"
    );
    for ep in 1..=epochs {
        match ds.task {
            TaskType::Classification => {
                train_epoch(&mut net, &x_train, &y_train_cls, batch, &opt, &mut rng)?;
            }
            TaskType::Regression => {
                // Manual minibatch loop for regression with MSE
                let n = y_train_norm.len();
                let steps = n.div_ceil(batch);
                for _ in 0..steps {
                    let idx: Vec<usize> =
                        (0..batch).map(|_| (rng.next_u64() as usize) % n).collect();
                    let cols = ds.num_features;
                    let mut xb_data = Vec::with_capacity(batch * cols);
                    let mut yb = Vec::with_capacity(batch);
                    for &i in &idx {
                        xb_data.extend_from_slice(&x_train.data[i * cols..(i + 1) * cols]);
                        yb.push(y_train_norm[i]);
                    }
                    let xb = Tensor::matrix(batch, cols, xb_data)?;
                    let logits = net.forward(&xb)?;
                    let (_, dl) = mse(&logits, &yb)?;
                    net.backward(&dl)?;
                    net.step(&opt)?;
                }
            }
            TaskType::TransformerSLM => unreachable!(),
        }

        if ep % (epochs / 10).max(1) == 0 || ep == 1 {
            let (tl, metric_str) = match ds.task {
                TaskType::Classification => {
                    let loss = softmax_cross_entropy(&net.forward(&x_train)?, &y_train_cls)?.0;
                    let acc = accuracy(&mut net, &x_val, &y_val_cls)?;
                    (loss, format!("val_acc={:.1}%", acc * 100.0))
                }
                TaskType::Regression => {
                    let loss = mse(&net.forward(&x_train)?, &y_train_norm)?.0;
                    let vl = mse(&net.forward(&x_val)?, &y_val_norm)?.0;
                    let rmse_raw = (vl * norm.stds[norm.stds.len() - 1].powi(2)).sqrt();
                    (loss, format!("val_rmse={:.0}", rmse_raw))
                }
                TaskType::TransformerSLM => unreachable!(),
            };
            let vl = match ds.task {
                TaskType::Classification => {
                    softmax_cross_entropy(&net.forward(&x_val)?, &y_val_cls)?.0
                }
                TaskType::Regression => mse(&net.forward(&x_val)?, &y_val_norm)?.0,
                TaskType::TransformerSLM => unreachable!(),
            };
            println!("{ep:>6}  {tl:>10.4}  {vl:>10.4}  {metric_str}");
        }
    }
    println!("\nTraining finished in {:.1}s", t0.elapsed().as_secs_f32());

    // ── Evaluate ──────────────────────────────────────────────────────────────
    let (x_all_raw, y_all_cls, y_all_reg) = ds.to_tensors()?;
    let x_all = norm.transform(&x_all_raw)?;
    let y_all_norm: Vec<f32> = y_all_reg
        .iter()
        .map(|&v| norm.normalise_target(v))
        .collect();
    match ds.task {
        TaskType::Classification => {
            let acc = accuracy(&mut net, &x_all, &y_all_cls)?;
            println!("Full-dataset accuracy : {:.1}%", acc * 100.0);
        }
        TaskType::Regression => {
            let rmse_norm = mse(&net.forward(&x_all)?, &y_all_norm)?.0.sqrt();
            let rmse_raw = rmse_norm * norm.stds[norm.stds.len() - 1];
            println!("Full-dataset RMSE : {:.2} (raw scale)", rmse_raw);
        }
        TaskType::TransformerSLM => unreachable!(),
    }

    // ── Build metadata ────────────────────────────────────────────────────────
    let model = net.to_inference_task(ds.task)?;
    let meta = ModelMetadata {
        dataset_name: ds_name.to_string(),
        task: ds.task,
        feature_names: ds.feature_names.clone(),
        feature_ranges: ds.feature_ranges.clone(),
        class_names: ds.class_names.clone(),
        target_name: if ds.task == TaskType::Regression {
            "value".into()
        } else {
            "".into()
        },
        target_range: ds.target_range,
        input_dim: ds.num_features,
        output_dim,
    };

    // ── Export ────────────────────────────────────────────────────────────────
    let bytes = to_bytes(&model, &norm, &meta)?;
    if let Some(parent) = std::path::Path::new(model_path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(model_path, &bytes)?;
    println!("Saved {} bytes → {model_path}", bytes.len());

    // ── Spot-check ────────────────────────────────────────────────────────────
    let (model2, norm2, _) = ferrum_core::from_bytes(&bytes)?;
    let probe_raw = Tensor::row(ds.rows[0].features.clone())?;
    let probe = norm2.transform(&probe_raw)?;
    let out = model2.forward(&probe)?;
    match ds.task {
        TaskType::Classification => {
            let cls = argmax_rows(&out)?[0];
            println!(
                "Reload check: row[0] → class {} ({:?})",
                cls,
                ds.class_names.get(cls).map(|s| s.as_str()).unwrap_or("?")
            );
        }
        TaskType::Regression => {
            let pred_raw = norm2.denormalise_target(out.data[0]);
            println!(
                "Reload check: row[0] → {pred_raw:.2} (actual: {:.2})",
                ds.rows[0].target
            );
        }
        TaskType::TransformerSLM => unreachable!(),
    }
    Ok(())
}
