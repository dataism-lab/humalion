#!make

lint:  ## [Local development] Run code quality checks (formatting, imports, lint, types, etc)
	ruff check ${PYTHON_SOURCES} && ruff format --check ${PYTHON_SOURCES}

format:  ## [Local development] Auto-format python code
	ruff format ${PYTHON_SOURCES} && ruff check --fix ${PYTHON_SOURCES}

.PHONY: help

help: # Run `make help` to get help on the make commands
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'