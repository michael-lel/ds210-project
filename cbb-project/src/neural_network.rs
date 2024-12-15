use ndarray::Axis;
use ndarray::Array1;
use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

pub struct NeuralNetwork {
    w1: Array2<f64>,
    w2: Array2<f64>,
}

impl NeuralNetwork {
    pub fn new(input_size: usize, hidden_nodes: usize, output_size: usize) -> Self {
        let w1 = Array2::random((input_size, hidden_nodes), Uniform::new(0.0, 0.1));
        let w2 = Array2::random((hidden_nodes, output_size), Uniform::new(0.0, 0.1));
        Self {w1,w2}
    }

    pub fn sigmoid(arr: &Array2<f64>) -> Array2<f64>{
        arr.mapv(|x| 1.0 / (1.0 + (-x).exp()))
    }

    pub fn sigmoid_derivative(x: &Array2<f64>) -> Array2<f64> {
        x * &(1.0 - x)
    }

    pub fn forward_pass(&self, x: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        let z1 = x.dot(&self.w1);
        let a1 = Self::sigmoid(&z1);
        let z2 = a1.dot(&self.w2);
        let a2 = Self::sigmoid(&z2);
        (a1, a2)
    }

    pub fn backward_pass(&mut self, x: &Array2<f64>, a1: &Array2<f64>, a2: &Array2<f64>, y: &Array2<f64>) {
        let e = y - a2;
        let beta = 0.05;

        let delt2 = &e * &Self::sigmoid_derivative(a2);
        self.w2 = &self.w2 + &(a1.t().dot(&delt2) * beta);

        let delt1 = &delt2.dot(&self.w2.t()) * &Self::sigmoid_derivative(a1);
        self.w1 = &self.w1 + &(x.t().dot(&delt1) * beta);
    }

    pub fn train(&mut self, x_train: &Array2<f64>, y_train: &Array2<f64>, epochs: usize) {
        for _i in 0..epochs {
            let (a1, a2) = self.forward_pass(x_train);
            self.backward_pass(x_train, &a1, &a2, y_train);
        }
    }

    pub fn test(&self, x: &Array2<f64>) -> Array1<usize> {
        let (_a1,a2) = self.forward_pass(x);
        a2.map_axis(Axis(1), |row| {
            row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
        })
    }

    pub fn accuracy(&self, x_test: &Array2<f64>, y_test: &Array2<f64>) -> f64 {
        let predictions = self.test(x_test);
        let correct = predictions.iter().zip(y_test.iter()).filter(|(&pred, &true_label)| pred == true_label as usize).count();
        (correct as f64) / (y_test.len() as f64)
    }
}