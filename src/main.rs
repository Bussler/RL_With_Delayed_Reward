use drone_environment::DroneEnvironment;
use drone_environment::DroneEnvironmentConfig;

fn main() {
    println!("Initializing Drone Environment...");
    // let mut env = DroneEnvironment::new((0.0, 0.0, 0.0), 10.0, 0.1, 10.0, Some(1.0));

    let config =
        DroneEnvironmentConfig::from_yaml_file("configs/drone_env/default_config.yaml").unwrap();
    let mut env = DroneEnvironment::from_config(config);

    env.add_target(0, (10.0, 0.0, 0.0), (0.0, 0.0, 0.0), None, None);

    let obs1 = env.reset(None);
    println!("Observation 1: {:?}", obs1);

    let obs2 = env.step((1.0, 0.0, 0.0));
    println!("Observation 2: {:?}", obs2);
}
