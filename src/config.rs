use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PlayerConfig {
    pub position: (f64, f64, f64),
    pub speed: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TargetConfig {
    pub id: usize,
    pub position: (f64, f64, f64),
    pub velocity: (f64, f64, f64),
    pub trajectory_fn: Option<String>,
    pub max_flight_time: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EnvironmentConfig {
    pub dt: f64,
    pub max_time: f64,
    pub collision_radius: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DroneEnvironmentConfig {
    pub player: PlayerConfig,
    pub environment: EnvironmentConfig,
    pub targets: Vec<TargetConfig>,
}

impl DroneEnvironmentConfig {
    /// Load configuration from a YAML file
    pub fn from_yaml_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = fs::read_to_string(path)?;
        let config: DroneEnvironmentConfig = serde_yaml::from_str(&contents)?;
        Ok(config)
    }

    /// Save configuration to a YAML file
    pub fn to_yaml_file<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let yaml_string = serde_yaml::to_string(self)?;
        fs::write(path, yaml_string)?;
        Ok(())
    }
}
