pub mod activations;
pub mod layers;
pub mod losses;
pub mod optimizers;

use nalgebra::{ClosedAdd, ClosedMul, Scalar, SimdPartialOrd, SimdValue};
use num_traits::{One, Zero};

pub trait TensorItem:
    Copy + PartialOrd + ClosedAdd + ClosedMul + Scalar + SimdPartialOrd + SimdValue + One + Zero
{
}

impl TensorItem for i8 {}
impl TensorItem for i16 {}
impl TensorItem for i32 {}
impl TensorItem for i64 {}
impl TensorItem for f32 {}
impl TensorItem for f64 {}

pub trait Module {
    type In;
    type Out;
    type Gradient;

    fn forward(&self, x: Self::In) -> Self::Out;
    fn backward(&self, x: Self::In, dldy: Self::Out) -> (Self::In, Self::Gradient);
    fn update(&mut self, step: Self::Gradient);
}

