use nalgebra::Vector3;
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[derive(Debug, Clone)]
pub struct Observation {
    pub player_position: Vector3<f64>,
    pub target_ids: Vec<usize>,
    pub target_positions: Vec<Vector3<f64>>,
    pub target_velocities: Vec<Vector3<f64>>, // is set to zeros if target is dead
    pub target_distances: Vec<f64>,           // is set to high value if target is dead
    pub target_death_mask: Vec<i8>,
    pub time_left: f64,
}

impl Observation {
    // Convert to Python dictionary with numpy arrays for PyO3 compatibility
    pub fn to_numpy_dict(&self, py: Python) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);

        // Add player position as numpy array (shape: [3])
        let player_pos_array = PyArray1::from_slice(
            py,
            &[
                self.player_position.x,
                self.player_position.y,
                self.player_position.z,
            ],
        );
        dict.set_item("player_position", player_pos_array)?;

        // Add target IDs as numpy array
        let target_ids_i8: Vec<i8> = self.target_ids.iter().map(|&id| id as i8).collect();
        let target_ids_array = PyArray1::from_slice(py, &target_ids_i8);
        dict.set_item("target_ids", target_ids_array)?;

        // Convert Vector3 positions to 2D numpy array (shape: [n_targets, 3])
        let mut positions_flat = Vec::with_capacity(self.target_positions.len() * 3);
        for pos in &self.target_positions {
            positions_flat.extend_from_slice(&[pos.x, pos.y, pos.z]);
        }
        let target_positions_array = PyArray2::from_vec2(
            py,
            &self
                .target_positions
                .iter()
                .map(|pos| vec![pos.x, pos.y, pos.z])
                .collect::<Vec<_>>(),
        )?;
        dict.set_item("target_positions", target_positions_array)?;

        // Convert Vector3 velocities to 2D numpy array (shape: [n_targets, 3])
        let target_velocities_array = PyArray2::from_vec2(
            py,
            &self
                .target_velocities
                .iter()
                .map(|vel| vec![vel.x, vel.y, vel.z])
                .collect::<Vec<_>>(),
        )?;
        dict.set_item("target_velocities", target_velocities_array)?;

        // Add target distances as numpy array
        let target_distances_array = PyArray1::from_slice(py, &self.target_distances);
        dict.set_item("target_distances", target_distances_array)?;

        // Add target death mask as numpy i8 array
        let target_death_mask_array = PyArray1::from_slice(py, &self.target_death_mask);
        dict.set_item("target_death_mask", target_death_mask_array)?;

        // Add time left as scalar
        let time_left_array = PyArray1::from_slice(py, &vec![self.time_left]);
        dict.set_item("time_left", time_left_array)?;

        Ok(dict.unbind())
    }
}

pub struct Information {
    pub time: f64,
    pub hit_targets: usize,
    pub expired_targets: usize,
}

impl Information {
    pub fn to_py_dict(&self, py: Python) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);

        dict.set_item("time", self.time)?;
        dict.set_item("hit_targets", self.hit_targets as i64)?;
        dict.set_item("expired_targets", self.expired_targets as i64)?;

        Ok(dict.unbind())
    }
}
