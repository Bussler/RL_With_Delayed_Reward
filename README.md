# RL_With_Delayed_Reward

Training an agent with reinforcement learning to function in a rust simulation that only gives delayed rewards.


## Gettings started

Initialize your environment with [uv](https://docs.astral.sh/uv/):

```
uv venv && uv sync
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


## Train an agent on the environment

TODO

## Contributing
- Install automatic git commit message with `git config --local commit.template .gitmessage`