.PHONY: linting test

## Linting Targets ##

linting: cargo-fmt ruff isort mypy

cargo-fmt:
	@echo Running Cargo Fmt...
	uv run cargo fmt

ruff:
	@echo Running Ruff...
	uv run ruff format python/drone_environment/ scripts/
	uv run ruff check --fix python/drone_environment/ scripts/

isort:
	@echo Sorting imports with Isort...
	uv run isort python/drone_environment/ scripts/

mypy:
	@echo Type Checking with MyPy...
	uv run mypy python/drone_environment/ scripts/


## Testing Targets ##

tests:
	@echo Running Tests... TODO implement these!
	# uv run pytest tests/