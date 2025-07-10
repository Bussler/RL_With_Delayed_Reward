mod actors;
mod config;
mod observation_info_utils;

use nalgebra::Vector3;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3::IntoPyObjectExt;

use actors::{Actor, Player, Target};
pub use config::DroneEnvironmentConfig;
use observation_info_utils::{Information, Observation};

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
    arena_size: f64,
    collision_radius: f64,
    original_targets: Vec<Target>,
    expired_targets_this_step: usize,
    hit_targets_this_step: usize,
    hit_target_time_bonuses: Vec<f64>,
}

impl DroneEnvironment {
    pub fn new(
        player_position: (f64, f64, f64),
        player_speed: f64,
        dt: f64,
        max_time: f64,
        arena_size: f64,
        collision_radius: Option<f64>,
    ) -> Self {
        DroneEnvironment {
            player: Player::new(1, player_position, player_speed),
            targets: Vec::new(),
            time: 0.0,
            dt,
            max_time,
            arena_size,
            collision_radius: collision_radius.unwrap_or(1.0),
            original_targets: Vec::new(),
            expired_targets_this_step: 0,
            hit_targets_this_step: 0,
            hit_target_time_bonuses: Vec::new(),
        }
    }

    /// Create DroneEnvironment from YAML configuration file
    pub fn from_yaml_config<P: AsRef<std::path::Path>>(
        config_path: P,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let config = DroneEnvironmentConfig::from_yaml_file(config_path)?;
        Ok(Self::from_config(config))
    }

    /// Create DroneEnvironment from configuration struct
    pub fn from_config(config: DroneEnvironmentConfig) -> Self {
        let mut env = DroneEnvironment::new(
            config.player.position,
            config.player.speed,
            config.environment.dt,
            config.environment.max_time,
            config.environment.arena_size,
            config.environment.collision_radius,
        );

        // Add all targets from config
        for target_config in config.targets {
            env.add_target(
                target_config.id,
                target_config.position,
                target_config.velocity,
                target_config.trajectory_fn,
                target_config.max_flight_time,
            );
        }

        env
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
            Some(self.arena_size/2.0),
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
        self.hit_target_time_bonuses.clear();

        // Return observation
        self.get_observation()
    }

    pub fn step(&mut self, action: (f64, f64, f64)) -> Observation {
        // Move player according to action
        self.player.move_direction(action, self.dt);

        // Update targets
        self.expired_targets_this_step = 0;
        self.hit_targets_this_step = 0;
        self.hit_target_time_bonuses.clear();
        
        for target in &mut self.targets {
            if target.is_dead() {
                continue;
            }

            let target_active = target.update(self.dt);
            if !target_active {
                self.expired_targets_this_step += 1;
                continue;
            }

            if distance(self.player.position, target.position) < self.collision_radius {
                self.hit_targets_this_step += 1;
                // Store time bonus for hit targets for reward
                if let Some(remaining_time) = target.remaining_time() {
                    self.hit_target_time_bonuses.push(remaining_time);
                }
                target.shoot_down();
            }
        }

        // Update time
        self.time += self.dt;

        // Return observation
        return self.get_observation();
    }

    pub fn get_observation(&self) -> Observation {
        let mut target_positions = Vec::new();
        let mut target_velocities = Vec::new();
        let mut target_distances = Vec::new();
        let mut target_time_remaining = Vec::new();
        let mut target_death_mask = Vec::new();

        for target in &self.targets {
            let target_is_dead = target.is_dead();

            target_positions.push(target.position);
            target_velocities.push(if target_is_dead {
                Vector3::zeros()
            } else {
                target.velocity
            });

            let dist = distance(self.player.position, target.position);
            target_distances.push(dist); // TODO LOOK HERE
            target_time_remaining.push(match target.remaining_time() { // TODO LOOK HERE
                    Some(t) => t,
                    None => f64::MAX,
                });
            target_death_mask.push(if target_is_dead { 0 } else { 1 });
        }

        Observation {
            player_position: self.player.position,
            target_positions,
            target_velocities,
            target_distances,
            target_time_remaining,
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

    fn calculate_reward(&self) -> f64 {
        let mut reward = 0.0;
        
        // Base reward for hitting targets
        let hit_reward = self.hit_targets_this_step as f64 * 100.0;
        reward += hit_reward;
        
        // Time bonus for hitting targets early (normalized remaining time)
        let time_bonus: f64 = self.hit_target_time_bonuses.iter()
            .map(|&remaining_time| {
                // Bonus scales with remaining time (0-50 points)
                (remaining_time / self.max_time) * 50.0
            })
            .sum();
        reward += time_bonus;
        
        // Heavy penalty for expired targets
        let expiry_penalty = self.expired_targets_this_step as f64 * -150.0;
        reward += expiry_penalty;
        
        // Urgency bonus: reward for being close to targets about to expire
        // let urgency_bonus = self.calculate_urgency_bonus();
        // reward += urgency_bonus;
        
        // Completion bonus if all targets are eliminated by drone
        if self.targets.iter().all(|t| t.is_shot_down()) {
            let time_efficiency = (self.max_time - self.time) / self.max_time;
            reward += 200.0 * time_efficiency; // Up to 200 bonus points
        }
        
        // Small time penalty to encourage speed
        reward -= 0.1;
        
        reward
    }

    fn calculate_urgency_bonus(&self) -> f64 {
        let mut urgency_bonus = 0.0;
        
        for target in &self.targets {
            if target.is_dead() {
                continue;
            }
            
            if let Some(remaining_time) = target.remaining_time() {
                let distance_to_target = distance(self.player.position, target.position);
                let urgency_factor = 1.0 - (remaining_time / self.max_time);
                
                // Bonus for being close to urgent targets
                if urgency_factor > 0.7 { // Target has < 30% time remaining
                    let proximity_bonus = (10.0 / (1.0 + distance_to_target)) * urgency_factor;
                    urgency_bonus += proximity_bonus;
                }
            }
        }
        
        urgency_bonus
    }

    pub fn get_player_position(&self) -> (f64, f64, f64) {
        self.player.position()
    }

    pub fn set_player_speed(&mut self, speed: f64) {
        self.player.speed = speed;
    }

    pub fn get_target_positions(&self) -> Vec<(usize, bool, (f64, f64, f64))> {
        self.targets
            .iter()
            .map(|t| (t.id, t.is_dead(), t.position()))
            .collect()
    }

    pub fn get_time(&self) -> f64 {
        return self.time;
    }

    pub fn get_max_time(&self) -> f64 {
        return self.max_time;
    }

    pub fn get_done(&self) -> bool {
        return self.time >= self.max_time || self.targets.iter().all(|t| t.is_dead());
    }

    pub fn get_arena_size(&self) -> f64 {
        return self.arena_size;
    }
}

#[pyclass]
struct DroneEnvironmentWrapper {
    drone_environment: DroneEnvironment,
}

#[pymethods]
impl DroneEnvironmentWrapper {
    #[new]
    fn new(
        player_position: (f64, f64, f64),
        player_speed: f64,
        dt: f64,
        max_time: f64,
        arena_size: f64,
        collision_radius: Option<f64>,
    ) -> Self {
        DroneEnvironmentWrapper {
            drone_environment: DroneEnvironment::new(
                player_position,
                player_speed,
                dt,
                max_time,
                arena_size,
                collision_radius,
            ),
        }
    }

    #[classmethod]
    fn from_yaml_config(
        _cls: &Bound<'_, pyo3::types::PyType>,
        config_path: String,
    ) -> PyResult<Self> {
        match DroneEnvironment::from_yaml_config(config_path) {
            Ok(drone_environment) => Ok(DroneEnvironmentWrapper { drone_environment }),
            Err(e) => Err(pyo3::exceptions::PyIOError::new_err(format!(
                "Failed to load config: {}",
                e
            ))),
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
            let obs = self
                .drone_environment
                .reset(player_position)
                .to_numpy_dict(py);
            return obs;
        })
    }

    /// Take a step in the environment. Observations are already returned in numpy format
    fn step(&mut self, action: (f64, f64, f64)) -> PyResult<Py<PyTuple>> {
        let sim_observation = self.drone_environment.step(action);
        let sim_reward = self.drone_environment.calculate_reward();
        let sim_info = self.drone_environment.get_information();

        // Convert the result to a PyTuple and calculate the reward, done, info
        Python::with_gil(|py| {
            let observation = sim_observation.to_numpy_dict(py)?.into_py_any(py)?;
            let reward = sim_reward.into_py_any(py)?;
            let done = self.drone_environment.get_done().into_py_any(py)?;
            let info = sim_info.to_py_dict(py)?.into_py_any(py)?;

            let result = PyTuple::new(py, &[observation, reward, done, info])?;
            Ok(result.unbind())
        })
    }

    /// Observations are already returned in numpy format
    fn get_observation(&self) -> PyResult<Py<PyDict>> {
        Python::with_gil(|py| {
            let obs = self.drone_environment.get_observation().to_numpy_dict(py);
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

    fn get_max_time(&self) -> f64 {
        self.drone_environment.get_max_time()
    }

    fn get_done(&self) -> bool {
        self.drone_environment.get_done()
    }

    fn get_arena_size(&self) -> f64 {
        self.drone_environment.get_arena_size()
    }
}

// Python module definition
#[pymodule]
#[pyo3(name = "_lib")]
fn drone_environment(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DroneEnvironmentWrapper>()?;
    Ok(())
}
