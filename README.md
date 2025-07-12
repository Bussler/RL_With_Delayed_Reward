# RL_With_Delayed_Reward

Training an agent with reinforcement learning to function in a rust simulation that only gives delayed rewards.


## Gettings started

Initialize your environment with [uv](https://docs.astral.sh/uv/):

```
uv venv && uv sync
```


### In case of python linker errors while building rust files:

Try to set the `PYO3_PYTHON` environment variable to the python executable:
```
source .venv/bin/activate
export PYO3_PYTHON=$(which python3)
cargo build
```


### Linting

Linting and testing can easily be triggered via the makefile:
- make linting
- make tests

You can also set the following `setting.json` file in the local `.vscode/` directory for automatic code formatting on save:
```
{
    "[python]": {
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
        "source.fixAll": "explicit",
        "source.organizeImports": "explicit"
        },
        "editor.defaultFormatter": "charliermarsh.ruff"
    },
    "ruff.importStrategy": "fromEnvironment",
    "flake8.importStrategy": "fromEnvironment",
    "flake8.path": [
        "pflake8"
    ],
    "editor.formatOnSave": true,
    "workbench.colorCustomizations": {
        "[Your Color Theme]": {
            "editorRuler.foreground": "#ff0000"
        }
    },
    "editor.rulers": [
        100
    ]
}
```

## Example Usage

- Generate a scenario with the script in `./scripts/generate_scenario.py` or use the default one in `./configs/drone_env/default_config.yaml`
- Run and render the simulation with the script in `./scripts/example_usage.py`

## Train an agent on the environment

- Train with stable baselines: `uv run python scripts/train_stable_baseline_ppo.py`
- Train with skrl: `uv run python scripts/train_skrl_ppo.py`

Both scripts will save the trained model in the `./skrl` or `./stable_baselines` directory respectively.  
An example episode will be rendered to the `./animations` directory.  
You can scip the training parts of the scripts and go straight to inference by providing a path to a trained model.

## Contributing
- Install automatic git commit message with `git config --local commit.template .gitmessage`