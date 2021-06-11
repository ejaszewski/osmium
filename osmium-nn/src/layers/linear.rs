use nalgebra::{SMatrix, SVector};

use rand::{
    distributions::{Distribution, Uniform},
    prelude::*,
};

use crate::TensorItem;

#[derive(Clone)]
pub struct Linear<T, const M: usize, const N: usize> {
    pub(crate) weights: SMatrix<T, M, N>,
    pub(crate) bias: SVector<T, N>,
}

impl<T, const M: usize, const N: usize> Linear<T, M, N>
where
    T: TensorItem,
{
    pub fn forward(&self, x: SVector<T, M>) -> SVector<T, N> {
        self.weights.transpose() * x + self.bias
    }

    pub fn grad(
        &self,
        x: SVector<T, M>,
        in_grad: SVector<T, N>,
    ) -> (SMatrix<T, M, N>, SVector<T, M>) {
        let grad = self.weights * in_grad;
        let wgrad = x * in_grad.transpose();

        (wgrad, grad)
    }

    pub fn update(&mut self, dw: SMatrix<T, M, N>, db: SVector<T, N>) {
        self.weights += dw;
        self.bias += db;
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
