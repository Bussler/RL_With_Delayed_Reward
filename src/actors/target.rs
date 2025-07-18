use crate::actors::actor::{vector3_to_tuple, Actor};
use meval::eval_str;
use nalgebra::Vector3;

// Target representation
#[derive(Clone, Debug)]
pub struct Target {
    pub id: usize,
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
    trajectory_fn: Option<String>,
    pub time: f64,
    max_flight_time: Option<f64>,
    arena_radius: Option<f64>,
    expired: bool,
    shot_down: bool,
}

impl Actor for Target {
    fn id(&self) -> usize {
        self.id
    }

    fn position(&self) -> (f64, f64, f64) {
        vector3_to_tuple(self.position)
    }
}

impl Target {
    pub fn new(
        id: usize,
        position: (f64, f64, f64),
        velocity: (f64, f64, f64),
        trajectory_fn: Option<String>,
        max_flight_time: Option<f64>,
        arena_radius: Option<f64>,
    ) -> Self {
        Target {
            id,
            position: Vector3::new(position.0, position.1, position.2),
            velocity: Vector3::new(velocity.0, velocity.1, velocity.2),
            trajectory_fn,
            time: 0.0,
            max_flight_time,
            arena_radius,
            expired: false,
            shot_down: false,
        }
    }

    pub fn velocity(&self) -> (f64, f64, f64) {
        vector3_to_tuple(self.velocity)
    }

    pub fn update(&mut self, dt: f64) -> bool {
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

                let x_fn = components
                    .get(0)
                    .unwrap_or(&"0.0")
                    .replace("t", &t.to_string())
                    .replace("x", &self.position.x.to_string());
                let y_fn = components
                    .get(1)
                    .unwrap_or(&"0.0")
                    .replace("t", &t.to_string())
                    .replace("y", &self.position.y.to_string());
                let z_fn = components
                    .get(2)
                    .unwrap_or(&"0.0")
                    .replace("t", &t.to_string())
                    .replace("z", &self.position.z.to_string());

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

                // Check if target has gone out of bounds
                if let Some(arena_radius) = self.arena_radius {
                    if self.position.x.abs() > arena_radius
                        || self.position.y.abs() > arena_radius
                        || self.position.z.abs() > arena_radius
                    {
                        self.expired = true;
                        return false;
                    }
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

    pub fn is_dead(&self) -> bool {
        self.expired || self.shot_down
    }

    pub fn remaining_time(&self) -> Option<f64> {
        if let Some(max_time) = self.max_flight_time {
            Some(max_time - self.time)
        } else {
            None
        }
    }

    pub fn shoot_down(&mut self) {
        self.shot_down = true;
    }

    pub fn is_shot_down(&self) -> bool {
        self.shot_down
    }
}
