use std::ops::{Add, Mul};

use rand::{
    distributions::{Distribution, Uniform},
    prelude::*,
};

#[derive(Clone)]
pub struct Linear<T, const N: usize, const M: usize> {
    pub(crate) weights: [[T; N]; M],
    pub(crate) bias: [T; M],
}

impl<T, const N: usize, const M: usize> Linear<T, N, M>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    pub fn forward(&self, x: [T; N]) -> [T; M] {
        let mut output = self.bias;

        output
            .iter_mut()
            .zip(self.weights.iter())
            .for_each(|(yj, wj)| {
                *yj = x
                    .iter()
                    .zip(wj.iter())
                    .map(|(&xi, &wij)| xi * wij)
                    .fold(*yj, |acc, x| acc + x);
            });

        output
    }
}

impl<const N: usize, const M: usize> Linear<f32, N, M> {
    pub fn new() -> Self {
        let k = 1.0 / (N * M) as f32;
        let distr = Uniform::new(-k.sqrt(), k.sqrt());

        let rng = thread_rng();
        let mut weights = [[0.0; N]; M];
        weights
            .iter_mut()
            .flat_map(|row| row.iter_mut())
            .zip(distr.sample_iter(rng))
            .for_each(|(w, r)| *w = r);

        let rng = thread_rng();
        let mut bias = [0.0; M];
        bias.iter_mut()
            .zip(distr.sample_iter(rng))
            .for_each(|(w, r)| *w = r);

        Self { weights, bias }
    }

    pub fn grad(&self, x: [f32; N], in_grad: [f32; M]) -> ([[f32; N]; M], [f32; N]) {
        let mut grad = [0.0; N];
        grad.iter_mut().enumerate().for_each(|(i, gi)| {
            *gi = in_grad
                .iter()
                .enumerate()
                .map(|(j, dij)| dij * self.weights[j][i])
                .sum()
        });

        let mut wgrad = [[0.0; N]; M];
        wgrad.iter_mut().enumerate().for_each(|(i, row)| {
            row.iter_mut()
                .enumerate()
                .for_each(|(j, wij)| *wij += in_grad[i] * x[j])
        });

        (wgrad, grad)
    }

    pub fn update(&mut self, dw: [[f32; N]; M], db: [f32; M]) {
        for j in 0..M {
            for i in 0..N {
                self.weights[j][i] += dw[j][i];
            }
            self.bias[j] += db[j];
        }
    }
}
