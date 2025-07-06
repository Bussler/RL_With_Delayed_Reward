from drone_environment.gym import DroneGymEnv
from stable_baselines3 import PPO

model_path = ""  # "drone_ppo_model.zip"

if model_path:
    model = PPO.load(model_path)
    print(f"Loaded model from {model_path}")
else:
    print("No model path provided. Training...")

    # Create and wrap the environment
    env = DroneGymEnv(
        renderer="matplotlib" if model_path else None,
        render_mode="rgb_array" if model_path else None,
    )

    # Create the PPO agent
    model = PPO(
        "MultiInputPolicy",  # "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="./stable_baselines/drone_ppo_tensorboard/",
        device="cpu",
    )

    # Train the agent
    model.learn(total_timesteps=250000)

    # Save the trained model
    model.save("drone_ppo_model")

# Create a new environment for evaluation with rendering
eval_env = DroneGymEnv(renderer="matplotlib", render_mode="rgb_array")

# Evaluate the trained agent
# mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
# print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

# Test the trained agent
obs, info = eval_env.reset()
done = False
total_reward = 0.0
step = 0

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    if step % 5 == 0:
        eval_env.render()
    done = terminated or truncated
    total_reward += reward
    step += 1

print(f"Total reward: {total_reward}")
eval_env.close()
