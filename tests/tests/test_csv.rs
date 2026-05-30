use ferrum_core::{CsvDataset, TaskType};

const CLASSIFICATION_CSV: &str = "\
sepal_length,sepal_width,label
5.1,3.5,Iris-setosa
4.9,3.0,Iris-setosa
6.0,3.0,Iris-virginica
";

const REGRESSION_CSV: &str = "\
x,y,price
1.0,2.0,100.0
2.0,3.0,200.0
";

#[test]
fn classification_dataset_parsing() {
    let ds = CsvDataset::from_str(CLASSIFICATION_CSV).unwrap();
    assert_eq!(ds.task, TaskType::Classification);
    assert_eq!(ds.num_features, 2);
    assert_eq!(ds.num_classes, 2);
    assert_eq!(ds.feature_names, vec!["sepal_length", "sepal_width"]);
    assert_eq!(ds.class_names, vec!["Iris-setosa", "Iris-virginica"]);
    assert_eq!(ds.rows.len(), 3);
}

#[test]
fn regression_dataset_parsing() {
    let ds = CsvDataset::from_str(REGRESSION_CSV).unwrap();
    assert_eq!(ds.task, TaskType::Regression);
    assert_eq!(ds.num_features, 2);
    assert_eq!(ds.rows.len(), 2);
}
