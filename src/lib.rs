use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use nalgebra::{Vector3};
use pyo3::IntoPyObjectExt;
use meval::eval_str;

// 3D Vector representation
#[pyclass]
#[derive(Clone, Debug)]
struct Vec3 {
    #[pyo3(get, set)]
    x: f64,
    #[pyo3(get, set)]
    y: f64,
    #[pyo3(get, set)]
    z: f64,
}

#[pymethods]
impl Vec3 {
    #[new]
    fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3 { x, y, z }
    }

    // Instead of returning Vector3, convert to a tuple which PyO3 can handle
    fn to_tuple(&self) -> (f64, f64, f64) {
        (self.x, self.y, self.z)
    }

    #[staticmethod]
    fn from_tuple(tuple: (f64, f64, f64)) -> Self {
        Vec3 { x: tuple.0, y: tuple.1, z: tuple.2 }
    }
}

// Helper function to convert Vec3 to Vector3 (not exposed to Python)
impl Vec3 {
    fn to_vector3(&self) -> Vector3<f64> {
        Vector3::new(self.x, self.y, self.z)
    }

    fn from_vector3(v: Vector3<f64>) -> Self {
        Vec3 { x: v[0], y: v[1], z: v[2] }
    }

    fn distance(&self, other: &Vec3) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2) + (self.z - other.z).powi(2)).sqrt()
    }
}

// Target representation
#[pyclass]
#[derive(Clone, Debug)]
struct Target {
    #[pyo3(get)]
    id: usize,
    #[pyo3(get)]
    position: Vec3,
    #[pyo3(get)]
    velocity: Vec3,
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
    fn new(id: usize, position: Vec3, velocity: Vec3, trajectory_fn: Option<String>, max_flight_time: Option<f64>) -> Self {

        Target {
            id,
            position,
            velocity,
            trajectory_fn,
            time: 0.0,
            max_flight_time,
            expired: false,
        }
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
                
                let prevPosition = self.position.clone();

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
                self.velocity.x = (self.position.x - prevPosition.x) / dt;
                self.velocity.y = (self.position.y - prevPosition.y) / dt;
                self.velocity.z = (self.position.z - prevPosition.z) / dt;

                return true;
            }
        } 
        
        self.position.x += self.velocity.x * dt;
        self.position.y += self.velocity.y * dt;
        self.position.z += self.velocity.z * dt;

        true
        
    }
}

// Player representation
#[pyclass]
#[derive(Clone, Debug)]
struct Player {
    #[pyo3(get)]
    position: Vec3,
    #[pyo3(get, set)]
    speed: f64,
}

#[pymethods]
impl Player {
    #[new]
    fn new(position: Vec3, speed: f64) -> Self {
        Player {
            position,
            speed,
        }
    }

    fn move_direction(&mut self, direction: Vec3, dt: f64) {
        // Normalize direction vector
        let dir_vec = direction.to_vector3();
        let norm = (dir_vec[0].powi(2) + dir_vec[1].powi(2) + dir_vec[2].powi(2)).sqrt();
        
        if norm > 0.0 {
            let normalized = dir_vec / norm;
            let movement = normalized * self.speed * dt;
            
            self.position.x += movement[0];
            self.position.y += movement[1];
            self.position.z += movement[2];
        }
    }
}

// Environment for RL training
#[pyclass]
struct DroneEnvironment {
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
    fn new(
        player_position: Vec3, 
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

    fn add_target(&mut self, target_id: usize, position: Vec3, velocity: Vec3, trajectory_fn: Option<String>, max_flight_time: Option<f64>) {
        self.original_targets.push(Target::new(target_id, position, velocity, trajectory_fn, max_flight_time));
    }

    fn reset(&mut self, player_position: Option<Vec3>) -> PyResult<Py<PyDict>> {
        self.time = 0.0;
        self.done = false;
        
        if let Some(pos) = player_position {
            self.player.position = pos;
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

    fn step(&mut self, action: Vec3) -> PyResult<Py<PyTuple>> {
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
            let dist = self.player.position.distance(&target.position);
            
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
        dict.set_item("player_position", (
            self.player.position.x,
            self.player.position.y,
            self.player.position.z
        )).unwrap();
        
        // Add target information
        let mut target_ids = Vec::new();
        let mut target_positions = Vec::new();
        let mut target_velocities = Vec::new();
        let mut target_distances = Vec::new();
        
        for target in &self.targets {
            target_ids.push(target.id);

            target_positions.push((
                target.position.x,
                target.position.y,
                target.position.z
            ));
            
            target_velocities.push((
                target.velocity.x,
                target.velocity.y,
                target.velocity.z
            ));

            let dist = self.player.position.distance(&target.position);
            
            target_distances.push(dist);
        }
        
        dict.set_item("target_ids", target_ids).unwrap();
        dict.set_item("target_positions", target_positions).unwrap();
        dict.set_item("target_velocities", target_velocities).unwrap();
        dict.set_item("target_distances", target_distances).unwrap();
        
        Ok(dict.unbind())
    }
    
    fn get_player_position(&self) -> Vec3 {
        self.player.position.clone()
    }
    
    fn set_player_speed(&mut self, speed: f64) {
        self.player.speed = speed;
    }
    
    fn get_target_positions(&self) -> Vec<(usize, Vec3)> {
        self.targets.iter().map(|t| (t.id, t.position.clone())).collect()
    }

    fn get_target_by_id(&self, id: usize) -> Option<Target> {
        self.targets.iter().find(|t| t.id == id).cloned()
    }

    fn get_time(&self) -> f64 {
        return self.time;
    }
}

// Python module definition
#[pymodule]
#[pyo3(name="_lib")]
fn drone_environment(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Vec3>()?;
    m.add_class::<Target>()?;
    m.add_class::<Player>()?;
    m.add_class::<DroneEnvironment>()?;
    Ok(())
}
