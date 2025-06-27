use nalgebra::Vector3;

pub trait Actor {
    fn id(&self) -> usize;
    fn position(&self) -> (f64, f64, f64);
}

pub fn vector3_to_tuple(vec: Vector3<f64>) -> (f64, f64, f64) {
    (vec.x, vec.y, vec.z)
}
