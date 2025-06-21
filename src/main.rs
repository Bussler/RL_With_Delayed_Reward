use drone_environment::DroneEnvironment;
use drone_environment::Vec3;

fn main() {
    println!("Initializing Drone Environment...");
	let mut env = DroneEnvironment::new(
        Vec3::new(0.0, 0.0, 0.0),
        10.0,
        0.1,
        10.0,
        Some(1.0),
    );

    env.add_target(0, Vec3::new(10.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0), None, None);
    
}