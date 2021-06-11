use std::marker::PhantomData;

use nalgebra::SVector;
use num_traits::Zero;

use crate::TensorItem;

pub struct ReLU<T, const N: usize> {
    phantom: PhantomData<T>,
}

impl<T, const N: usize> ReLU<T, N>
where
    T: TensorItem,
{
    pub fn forward(&self, x: SVector<T, N>) -> SVector<T, N> {
        x.sup(&SVector::<T, N>::zero())
    }
}

impl<const N: usize> ReLU<f32, N> {
    pub fn grad(&self, x: [f32; N], in_grad: [f32; N]) -> [f32; N] {
        // Initialize to last input
        let mut grad = x;

        // Compute gradient from last input
        grad.iter_mut()
            .zip(in_grad.iter())
            .for_each(|(o, g)| *o = if *o > 0.0 { *g } else { 0.0 });

        grad
    }
}
