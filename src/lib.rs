mod actors;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use nalgebra::Vector3;
use pyo3::IntoPyObjectExt;
use meval::eval_str;


fn vector3_to_tuple(vec: Vector3<f64>) -> (f64, f64, f64) {
    (vec.x, vec.y, vec.z)
}

// Helper function to calculate distance between two Vector3 points
// TODO check that this works!
fn distance(a: Vector3<f64>, b: Vector3<f64>) -> f64 {
    (a - b).norm()
}

// Target representation
#[pyclass]
#[derive(Clone, Debug)]
struct Target {
    #[pyo3(get)]
    id: usize,
    position: Vector3<f64>,
    velocity: Vector3<f64>,
    trajectory_fn: Option<String>,
    time: f64,
    #[pyo3(get)]
    max_flight_time: Option<f64>,
    #[pyo3(get)]
    expired: bool,
}

#[pymethods]
impl Target {
    #[new]
    fn new(id: usize, position: (f64, f64, f64), velocity: (f64, f64, f64), trajectory_fn: Option<String>, max_flight_time: Option<f64>) -> Self {
        Target {
            id,
            position: Vector3::new(position.0, position.1, position.2),
            velocity: Vector3::new(velocity.0, velocity.1, velocity.2),
            trajectory_fn,
            time: 0.0,
            max_flight_time,
            expired: false,
        }
    }

    #[getter]
    fn position(&self) -> (f64, f64, f64) {
        vector3_to_tuple(self.position)
    }

    #[getter]
    fn velocity(&self) -> (f64, f64, f64) {
        vector3_to_tuple(self.velocity)
    }

    fn update(&mut self, dt: f64) -> bool {
        self.time += dt;

        // Check if target has reached its maximum flight time
        if let Some(max_time) = self.max_flight_time {
            if self.time >= max_time {
                self.expired = true;
                return false;
            }
        }
        
        // Parse and evaluate trajectory function
        // The function should return a 3D position based on time
        if let Some(ref trajectory_fn) = self.trajectory_fn {
            if !trajectory_fn.is_empty() {
                let t = self.time;
                
                // Replace t with actual time in the function string
                let components: Vec<&str> = trajectory_fn.split(',').map(|s| s.trim()).collect();

                let x_fn = components.get(0).unwrap_or(&"0.0").replace("t", &t.to_string()).replace("x", &self.position.x.to_string());
                let y_fn = components.get(1).unwrap_or(&"0.0").replace("t", &t.to_string()).replace("y", &self.position.y.to_string());
                let z_fn = components.get(2).unwrap_or(&"0.0").replace("t", &t.to_string()).replace("z", &self.position.z.to_string());
                
                // TODO check that this works as intended and we do not need a deep copy here!
                let prev_position = self.position;

                if let Ok(x_result) = eval_str(&x_fn) {
                    self.position.x = x_result;
                }
                
                if let Ok(y_result) = eval_str(&y_fn) {
                    self.position.y = y_result;
                }
                
                if let Ok(z_result) = eval_str(&z_fn) {
                    self.position.z = z_result;
                }
                
                // Update velocity based on position change
                // This is a simple approximation
                self.velocity = (self.position - prev_position) / dt;

                return true;
            }
        } 
        
        self.position += self.velocity * dt;
        true
    }
}

// Player representation
#[pyclass]
#[derive(Clone, Debug)]
struct Player {
    position: Vector3<f64>,
    #[pyo3(get, set)]
    speed: f64,
}

#[pymethods]
impl Player {
    #[new]
    fn new(position: (f64, f64, f64), speed: f64) -> Self {
        Player {
            position: Vector3::new(position.0, position.1, position.2),
            speed,
        }
    }

    #[getter]
    fn position(&self) -> (f64, f64, f64) {
        vector3_to_tuple(self.position)
    }

    fn move_direction(&mut self, direction: (f64, f64, f64), dt: f64) {
        // Normalize direction vector
        let dir_vec = Vector3::new(direction.0, direction.1, direction.2);
        let norm = dir_vec.norm();
        
        if norm > 0.0 {
            let normalized = dir_vec / norm;
            let movement = normalized * self.speed * dt;
            
            self.position += movement;
        }
    }
}

// Environment for RL training
#[pyclass]
pub struct DroneEnvironment {
    player: Player,
    targets: Vec<Target>,
    time: f64,
    dt: f64,
    max_time: f64,
    collision_radius: f64,
    original_targets: Vec<Target>,
    expired_missiles_this_step: usize,
    #[pyo3(get)]
    done: bool,
}

#[pymethods]
impl DroneEnvironment {
    #[new]
    pub fn new(
        player_position: (f64, f64, f64), 
        player_speed: f64,
        dt: f64,
        max_time: f64,
        collision_radius: Option<f64>
    ) -> Self {
        DroneEnvironment {
            player: Player::new(player_position, player_speed),
            targets: Vec::new(),
            time: 0.0,
            dt,
            max_time,
            collision_radius: collision_radius.unwrap_or(1.0),
            original_targets: Vec::new(),
            expired_missiles_this_step: 0,
            done: false,
        }
    }

    pub fn add_target(&mut self, target_id: usize, position: (f64, f64, f64), velocity: (f64, f64, f64), trajectory_fn: Option<String>, max_flight_time: Option<f64>) {
        self.original_targets.push(Target::new(target_id, position, velocity, trajectory_fn, max_flight_time));
    }

    pub fn reset(&mut self, player_position: Option<(f64, f64, f64)>) -> PyResult<Py<PyDict>> {
        self.time = 0.0;
        self.done = false;
        
        if let Some(pos) = player_position {
            self.player.position = Vector3::new(pos.0, pos.1, pos.2);
        }
        
        // Reset all targets to their initial state
        for target in &mut self.original_targets {
            target.time = 0.0;
        }

        self.targets = self.original_targets.clone();

        self.expired_missiles_this_step = 0;
        
        // Return observation
        Python::with_gil(|py| {
            self.get_observation(py)
        })
    }

    pub fn step(&mut self, action: (f64, f64, f64)) -> PyResult<Py<PyTuple>> {
        // Move player according to action
        self.player.move_direction(action, self.dt);
        
        // Update targets
        let mut expired_target_ids = Vec::new();
        for target in &mut self.targets {
            let target_active = target.update(self.dt);
            if !target_active{
                expired_target_ids.push(target.id);
            }
        }
        self.expired_missiles_this_step = expired_target_ids.len();
        self.targets.retain(|target| !expired_target_ids.contains(&target.id));
        
        // Check for collisions and calculate reward
        let mut reward = self.expired_missiles_this_step as f64;
        // Use retain to keep only targets that are not collided with
        self.targets.retain(|target| {
            let dist = distance(self.player.position, target.position);
            
            if dist < self.collision_radius {
                reward += 1.0;
                false // Remove this target
            } else {
                true // Keep this target
            }
        });
        
        // Update time
        self.time += self.dt;

        if self.time >= self.max_time || self.targets.is_empty() {
            self.done = true;
        }
        
        // Return observation, reward, done, info
        // TODO handle reward, done, info with wrapper!!!
        Python::with_gil(|py| {
            let observation = self.get_observation(py)?.into_py_any(py)?;
            let reward = reward.into_py_any(py)?;
            let done = self.done.into_py_any(py)?;
            let info = PyDict::new(py).into_py_any(py)?;

            let result = PyTuple::new(py, &[observation, reward, done, info])?;
            Ok(result.unbind())
        })

    }

    fn get_observation(&self, py: Python) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        
        // Add player position
        dict.set_item("player_position", self.player.position()).unwrap();
        
        // Add target information
        let mut target_ids = Vec::new();
        let mut target_positions = Vec::new();
        let mut target_velocities = Vec::new();
        let mut target_distances = Vec::new();
        
        for target in &self.targets {
            target_ids.push(target.id);
            target_positions.push(target.position());
            target_velocities.push(target.velocity());

            let dist = distance(self.player.position, target.position);
            target_distances.push(dist);
        }
        
        dict.set_item("target_ids", target_ids).unwrap();
        dict.set_item("target_positions", target_positions).unwrap();
        dict.set_item("target_velocities", target_velocities).unwrap();
        dict.set_item("target_distances", target_distances).unwrap();
        
        Ok(dict.unbind())
    }
    
    pub fn get_player_position(&self) -> (f64, f64, f64) {
        self.player.position()
    }
    
    pub fn set_player_speed(&mut self, speed: f64) {
        self.player.speed = speed;
    }
    
    pub fn get_target_positions(&self) -> Vec<(usize, (f64, f64, f64))> {
        self.targets.iter().map(|t| (t.id, t.position())).collect()
    }

    pub fn get_target_by_id(&self, id: usize) -> Option<Target> {
        self.targets.iter().find(|t| t.id == id).cloned()
    }

    pub fn get_time(&self) -> f64 {
        return self.time;
    }
}

// Python module definition
#[pymodule]
#[pyo3(name="_lib")]
fn drone_environment(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Target>()?;
    m.add_class::<Player>()?;
    m.add_class::<DroneEnvironment>()?;
    Ok(())
}
