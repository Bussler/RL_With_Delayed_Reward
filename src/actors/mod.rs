pub mod player;
pub mod target;
use nalgebra::{Vector3};

pub trait Actor {
    fn id(&self) -> usize;
    fn position(&self) -> Vector3<f64>;
}