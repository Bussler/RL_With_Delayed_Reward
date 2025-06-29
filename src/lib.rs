mod actors;
mod observation_info_utils;

use nalgebra::{Vector3, QR};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3::IntoPyObjectExt;

use actors::{Actor, Player, Target};
use observation_info_utils::{Information, Observation};

use crate::actors::target;

// Helper function to calculate distance between two Vector3 points
fn distance(a: Vector3<f64>, b: Vector3<f64>) -> f64 {
    (a - b).norm()
}

// Environment for RL training
pub struct DroneEnvironment {
    player: Player,
    targets: Vec<Target>,
    time: f64,
    dt: f64,
    max_time: f64,
    collision_radius: f64,
    original_targets: Vec<Target>,
    pub expired_targets_this_step: usize,
    pub hit_targets_this_step: usize,
}

impl DroneEnvironment {
    pub fn new(
        player_position: (f64, f64, f64),
        player_speed: f64,
        dt: f64,
        max_time: f64,
        collision_radius: Option<f64>,
    ) -> Self {
        DroneEnvironment {
            player: Player::new(1, player_position, player_speed),
            targets: Vec::new(),
            time: 0.0,
            dt,
            max_time,
            collision_radius: collision_radius.unwrap_or(1.0),
            original_targets: Vec::new(),
            expired_targets_this_step: 0,
            hit_targets_this_step: 0,
        }
    }

    pub fn add_target(
        &mut self,
        target_id: usize,
        position: (f64, f64, f64),
        velocity: (f64, f64, f64),
        trajectory_fn: Option<String>,
        max_flight_time: Option<f64>,
    ) {
        self.original_targets.push(Target::new(
            target_id,
            position,
            velocity,
            trajectory_fn,
            max_flight_time,
        ));
    }

    pub fn reset(&mut self, player_position: Option<(f64, f64, f64)>) -> Observation {
        self.time = 0.0;

        if let Some(pos) = player_position {
            self.player.position = Vector3::new(pos.0, pos.1, pos.2);
        }

        // Reset all targets to their initial state
        for target in &mut self.original_targets {
            target.time = 0.0;
        }

        self.targets = self.original_targets.clone();

        self.expired_targets_this_step = 0;
        self.hit_targets_this_step = 0;

        // Return observation
        self.get_observation()
    }

    pub fn step(&mut self, action: (f64, f64, f64)) -> Observation {
        // Move player according to action
        self.player.move_direction(action, self.dt);

        // Update targets
        self.expired_targets_this_step = 0;
        self.hit_targets_this_step = 0;
        for target in &mut self.targets {

            if target.is_dead(){
                continue;
            }

            let target_active = target.update(self.dt);
            if !target_active {
                self.expired_targets_this_step += 1;
                continue;
            }
            if distance(self.player.position, target.position) < self.collision_radius {
                self.hit_targets_this_step += 1;
                target.shoot_down();
            }
        }

        // Update time
        self.time += self.dt;

        // Return observation
        return self.get_observation();
    }

    pub fn get_observation(&self) -> Observation {
        let mut target_ids = Vec::new();
        let mut target_positions = Vec::new();
        let mut target_velocities = Vec::new();
        let mut target_distances = Vec::new();
        let mut target_death_mask = Vec::new();

        for target in &self.targets {
            target_ids.push(target.id);
            target_positions.push(target.position);
            target_velocities.push(target.velocity);

            let dist = distance(self.player.position, target.position);
            target_distances.push(dist);
            target_death_mask.push(target.is_dead());
        }

        Observation {
            player_position: self.player.position,
            target_ids,
            target_positions,
            target_velocities,
            target_distances,
            time_left: self.max_time - self.time,
            target_death_mask,
        }
    }

    pub fn get_information(&self) -> Information {
        Information {
            time: self.time,
            hit_targets: self.hit_targets_this_step,
            expired_targets: self.expired_targets_this_step,
        }
    }

    pub fn get_player_position(&self) -> (f64, f64, f64) {
        self.player.position()
    }

    pub fn set_player_speed(&mut self, speed: f64) {
        self.player.speed = speed;
    }

    pub fn get_target_positions(&self) -> Vec<(usize, bool, (f64, f64, f64))> {
        self.targets.iter().map(|t| (t.id, t.is_dead(), t.position())).collect()
    }

    pub fn get_time(&self) -> f64 {
        return self.time;
    }

    pub fn get_done(&self) -> bool {
        return self.time >= self.max_time || self.targets.iter().all(|t| t.is_dead());
    }
}

#[pyclass]
struct DoneEnvironmentWrapper {
    drone_environment: DroneEnvironment,
}

#[pymethods]
impl DoneEnvironmentWrapper {
    #[new]
    fn new(
        player_position: (f64, f64, f64),
        player_speed: f64,
        dt: f64,
        max_time: f64,
        collision_radius: Option<f64>,
    ) -> Self {
        DoneEnvironmentWrapper {
            drone_environment: DroneEnvironment::new(
                player_position,
                player_speed,
                dt,
                max_time,
                collision_radius,
            ),
        }
    }

    fn add_target(
        &mut self,
        target_id: usize,
        position: (f64, f64, f64),
        velocity: (f64, f64, f64),
        trajectory_fn: Option<String>,
        max_flight_time: Option<f64>,
    ) {
        self.drone_environment.add_target(
            target_id,
            position,
            velocity,
            trajectory_fn,
            max_flight_time,
        );
    }

    fn reset(&mut self, player_position: Option<(f64, f64, f64)>) -> PyResult<Py<PyDict>> {
        Python::with_gil(|py| {
            let obs = self.drone_environment.reset(player_position).to_py_dict(py);
            return obs;
        })
    }

    fn step(&mut self, action: (f64, f64, f64)) -> PyResult<Py<PyTuple>> {
        let sim_observation = self.drone_environment.step(action);
        let sim_reward = self.drone_environment.hit_targets_this_step
            - self.drone_environment.expired_targets_this_step;
        let sim_info = self.drone_environment.get_information();

        // Convert the result to a PyTuple and calculate the reward, done, info
        Python::with_gil(|py| {
            let observation = sim_observation.to_py_dict(py)?.into_py_any(py)?;
            let reward = sim_reward.into_py_any(py)?;
            let done = self.drone_environment.get_done().into_py_any(py)?;
            let info = sim_info.to_py_dict(py)?.into_py_any(py)?;

            let result = PyTuple::new(py, &[observation, reward, done, info])?;
            Ok(result.unbind())
        })
    }

    fn get_observation(&self) -> PyResult<Py<PyDict>> {
        Python::with_gil(|py| {
            let obs = self.drone_environment.get_observation().to_py_dict(py);
            return obs;
        })
    }

    fn get_information(&self) -> PyResult<Py<PyDict>> {
        Python::with_gil(|py| {
            let info = self.drone_environment.get_information().to_py_dict(py);
            return info;
        })
    }

    fn get_player_position(&self) -> (f64, f64, f64) {
        self.drone_environment.get_player_position()
    }

    fn set_player_speed(&mut self, speed: f64) {
        self.drone_environment.set_player_speed(speed);
    }

    fn get_target_positions(&self) -> PyResult<Py<PyDict>> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);

            for target in &self.drone_environment.targets {
                dict.set_item(target.id, (target.is_dead(), target.position()))?;
            }

            Ok(dict.unbind())
        })
    }

    fn get_time(&self) -> f64 {
        self.drone_environment.get_time()
    }

    fn get_done(&self) -> bool {
        self.drone_environment.get_done()
    }
}

// Python module definition
#[pymodule]
#[pyo3(name = "_lib")]
fn drone_environment(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DoneEnvironmentWrapper>()?;
    Ok(())
}
