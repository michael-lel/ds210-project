mod data_setup;
mod neural_network;
use rand::seq::SliceRandom;
use rand::thread_rng;
use ndarray::Array2;
use ndarray::Axis;
use ndarray::Array1;

fn k_folds_cv(x: &Array2<f64>, y: &Array2<f64>, k: usize) -> f64 {
    let num_samples = x.shape()[0];
    let mut indicies: Vec<usize> = (0..num_samples).collect();
    let mut rng = thread_rng();
    indicies.shuffle(&mut rng);
    let fold_size = num_samples/k;

    let mut accuracy_scores = 0.0;
    for fold in 0..k {
        let start = fold * fold_size;
        let end = start + fold_size;
        let test_range = &indicies[start..end];
        let train_range: &Vec<usize> = &indicies.iter().cloned().filter(|a| !test_range.contains(a)).collect();
        let x_test = x.select(Axis(0), &test_range);
        let x_train = x.select(Axis(0), &train_range);
        let y_test = y.select(Axis(0), &test_range);
        let y_train = y.select(Axis(0), &train_range);
        let mut network = neural_network::NeuralNetwork::new(18,64,8);
        network.train(&x_train, &y_train, 1);
        let accuracy = network.accuracy(&x_test, &y_test);
        accuracy_scores += accuracy;
    }
    let average = accuracy_scores/k as f64;
    average
}

fn apply_model(train_x: &Array2<f64>, train_y: &Array2<f64>, test_x: &Array2<f64>) -> Array1<usize> {
    let mut network = neural_network::NeuralNetwork::new(18,50,8);
    network.train(&train_x, &train_y, 50);
    network.test(test_x)
}

fn main() {
    let cbb_2020_data = data_setup::read_csv("cbb_2020data.csv");
    let cbb_full_data = data_setup::read_csv("cbb_fulldata.csv");
    let x_fullset = data_setup::create_x(&cbb_full_data);
    let y_fullset = data_setup::create_y(&cbb_full_data);
    let x_2020 = data_setup::create_x(&cbb_2020_data);
    let cbb_nn = k_folds_cv(&x_fullset, &y_fullset, 200);
    println!("Model Accuracy: {}%", cbb_nn*100.0);
    let predict_2020 = apply_model(&x_fullset, &y_fullset, &x_2020);
    println!("Predicted results for 2020: {:?}", predict_2020);
}