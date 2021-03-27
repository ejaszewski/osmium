use std::{cmp::PartialOrd, marker::PhantomData};

use num_traits::{clamp_min, identities::Zero};

pub struct ReLU<T, const N: usize> {
    phantom: PhantomData<T>,
}

impl<T, const N: usize> ReLU<T, N>
where
    T: Copy + Zero + PartialOrd,
{
    pub fn forward(&self, x: [T; N]) -> [T; N] {
        let mut y = x;
        y.iter_mut()
            .for_each(|yi| *yi = clamp_min(*yi, Zero::zero()));
        y
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
