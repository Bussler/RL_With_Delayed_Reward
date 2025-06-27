use nalgebra::Vector3;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[derive(Debug, Clone)]
pub struct Observation {
    pub player_position: Vector3<f64>,
    pub target_ids: Vec<usize>,
    pub target_positions: Vec<Vector3<f64>>,
    pub target_velocities: Vec<Vector3<f64>>,
    pub target_distances: Vec<f64>,
}

impl Observation {
    // Convert to Python dictionary for PyO3 compatibility
    pub fn to_py_dict(&self, py: Python) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);

        // Add player position as tuple
        let player_pos = (
            self.player_position.x,
            self.player_position.y,
            self.player_position.z,
        );
        dict.set_item("player_position", player_pos)?;

        // Add target information
        dict.set_item("target_ids", &self.target_ids)?;

        // Convert Vector3 positions to tuples
        let target_positions: Vec<(f64, f64, f64)> = self
            .target_positions
            .iter()
            .map(|pos| (pos.x, pos.y, pos.z))
            .collect();
        dict.set_item("target_positions", target_positions)?;

        // Convert Vector3 velocities to tuples
        let target_velocities: Vec<(f64, f64, f64)> = self
            .target_velocities
            .iter()
            .map(|vel| (vel.x, vel.y, vel.z))
            .collect();
        dict.set_item("target_velocities", target_velocities)?;

        dict.set_item("target_distances", &self.target_distances)?;

        Ok(dict.unbind())
    }
}

pub struct Information {
    pub time: f64,
    pub hit_targets: usize,
    pub expired_missiles: usize,
}

impl Information {
    pub fn to_py_dict(&self, py: Python) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);

        dict.set_item("time", self.time)?;
        dict.set_item("hit_targets", &self.hit_targets)?;
        dict.set_item("expired_missiles", &self.expired_missiles)?;

        Ok(dict.unbind())
    }
}
