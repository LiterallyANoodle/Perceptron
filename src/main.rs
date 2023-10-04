use perceptron::Perceptron;

use crate::matrix::Matrix;

mod matrix {

    #[derive(Debug)]
    pub struct Matrix<const WIDTH: usize, const HEIGHT: usize> {
        width: usize,
        height: usize,
        values: [[i32; WIDTH]; HEIGHT],
    }

    impl<const WIDTH: usize, const HEIGHT: usize> Matrix<WIDTH, HEIGHT> {

        pub fn new(values: [[i32; WIDTH]; HEIGHT]) -> Self {
            Matrix { width: WIDTH, height: HEIGHT, values: values }
        }

        pub fn get_width(&self) -> usize {
            self.width
        }

        pub fn get_height(&self) -> usize {
            self.height
        }

        pub fn get_values(&self) -> [[i32; WIDTH]; HEIGHT] {
            self.values
        }

        pub fn dot_product(&self, other: &Matrix<WIDTH, 1>) -> Result<i32, String> {

            if self.get_height() != 1 {
                return Err(format!("This dot product is meant to be used with two row vectors. This matrix has height {}.", self.get_height()));
            } else if other.get_width() != self.get_width() {
                return Err(format!("Matrices must be of the same length. Self is {} and other is {}.", self.get_width(), other.get_width()));
            } else {
                
                let self_row = self.get_values()[0];
                let other_row = other.get_values()[0];

                let mut sum = 0i32;
                for i in 0..self.get_width() {
                    sum += self_row[i] * other_row[i];
                }

                return Ok(sum);

            }

        }

    }

}

mod perceptron {
    use std::usize;
    use crate::matrix::Matrix;

    #[derive(Debug)]
    pub struct Perceptron<const INPUTS_LENGTH: usize> { 
        weights: Matrix<INPUTS_LENGTH, 1>,
        bias: i32,
    }

    impl<const INPUTS_LENGTH: usize> Perceptron<INPUTS_LENGTH> {

        pub fn new(weights: Matrix<INPUTS_LENGTH, 1>, bias: i32) -> Self {
            Perceptron { weights: weights, bias: bias }
        }

        pub fn get_bias(&self) -> i32 {
            self.bias
        }

        pub fn calculate_output(&self, inputs: &Matrix<INPUTS_LENGTH, 1>) -> Result<i32, String> {
            match self.weights.dot_product(&inputs) {
                Err(x) => Err(x),
                Ok(x) => Ok(x + self.get_bias())
            }
        }

    }

}

fn print_type_of<T>(_: &T) {
    println!("This is of type {}", std::any::type_name::<T>());
}

fn main() {
    let my_matrix1 = Matrix::new([[1, 2, 3], [4, 5, 6]]);
    let my_matrix2 = Matrix::new([[7, 8, 9]]);
    let my_perceptron = Perceptron::new(my_matrix2, 7);
    println!("This is my matrix1: {:?} ", my_matrix1);
    println!("This is my Perceptron: {:?} ", my_perceptron);
    print_type_of(&my_matrix1);
}
