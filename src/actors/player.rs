use crate::actors::actor::{Actor, vector3_to_tuple};
use nalgebra::Vector3;

// Player representation
#[derive(Clone, Debug)]
pub struct Player {
    pub id: usize,
    pub position: Vector3<f64>,
    pub speed: f64,
}

impl Actor for Player {
    fn id(&self) -> usize {
        self.id
    }

    fn position(&self) -> (f64, f64, f64) {
        vector3_to_tuple(self.position)
    }
}

impl Player {
    pub fn new(id: usize, position: (f64, f64, f64), speed: f64) -> Self {
        Player {
            id,
            position: Vector3::new(position.0, position.1, position.2),
            speed,
        }
    }

    pub fn move_direction(&mut self, direction: (f64, f64, f64), dt: f64) {
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