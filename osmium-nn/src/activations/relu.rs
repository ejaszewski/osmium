use std::marker::PhantomData;

use nalgebra::SVector;
use num_traits::{One, Zero};

use crate::{Module, TensorItem};

pub struct ReLU<T, const M: usize> {
    phantom: PhantomData<T>,
}

impl<T, const M: usize> Module for ReLU<T, M>
where
    T: TensorItem,
{
    type In = SVector<T, M>;

    type Out = SVector<T, M>;

    type Gradient = ();

    fn forward(&self, x: Self::In) -> Self::Out {
        x.sup(&SVector::<T, M>::zero())
    }

    fn backward(&self, x: Self::In, dldy: Self::Out) -> (Self::In, Self::Gradient) {
        let positive = x.map(|x| {
            if x > Zero::zero() {
                One::one()
            } else {
                Zero::zero()
            }
        });

        (dldy.component_mul(&positive), ())
    }

    fn update(&mut self, _step: Self::Gradient) {}
}
