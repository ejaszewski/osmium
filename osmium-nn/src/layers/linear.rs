use nalgebra::{SMatrix, SVector};

use rand::{
    distributions::{Distribution, Uniform},
    prelude::*,
};

use crate::{Module, TensorItem};

#[derive(Clone)]
pub struct Linear<T, const M: usize, const N: usize> {
    pub(crate) weights: SMatrix<T, M, N>,
    pub(crate) bias: SVector<T, N>,
}

impl<T, const M: usize, const N: usize> Module for Linear<T, M, N>
where
    T: TensorItem,
{
    type In = SVector<T, M>;

    type Out = SVector<T, N>;

    type Gradient = (SMatrix<T, M, N>, SVector<T, N>);

    fn forward(&self, x: Self::In) -> Self::Out {
        self.weights.transpose() * x + self.bias
    }

    fn backward(&self, x: Self::In, dldy: Self::Out) -> (Self::In, Self::Gradient) {
        let dldx = self.weights * dldy;
        let dldw = x * dldy.transpose();

        (dldx, (dldw, dldy))
    }

    fn update(&mut self, step: Self::Gradient) {
        self.weights += step.0;
        self.bias += step.1;
    }
}

impl<const M: usize, const N: usize> Linear<f32, M, N> {
    pub fn new() -> Self {
        let k = 1.0 / (N * M) as f32;
        let distr = Uniform::new(-k.sqrt(), k.sqrt());

        let rng = thread_rng();

        let mut weights = SMatrix::<f32, M, N>::zeros();
        weights
            .iter_mut()
            .zip(distr.sample_iter(rng))
            .for_each(|(w, r)| *w = r);

        let rng = thread_rng();
        let mut bias = SVector::<f32, N>::zeros();
        bias.iter_mut()
            .zip(distr.sample_iter(rng))
            .for_each(|(w, r)| *w = r);

        Self { weights, bias }
    }
}
