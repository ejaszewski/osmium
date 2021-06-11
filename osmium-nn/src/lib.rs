pub mod activations;
pub mod layers;

use nalgebra::{ClosedAdd, ClosedMul, Scalar, SimdPartialOrd, SimdValue};
use num_traits::{One, Zero};

pub trait TensorItem:
    Copy + ClosedAdd + ClosedMul + Scalar + SimdPartialOrd + SimdValue + One + Zero
{
}

impl TensorItem for i8 {}
impl TensorItem for i16 {}
impl TensorItem for i32 {}
impl TensorItem for i64 {}
impl TensorItem for f32 {}
impl TensorItem for f64 {}
