.PHONY: install fix format format-check lint types typecheck unit test quality check build-image

install:  ## Install the package with dev tooling from uv.lock (CPU; add extras on a GPU box).
	uv sync --frozen --group dev

format:  ## Auto-format imports and code.
	uv run --frozen ruff check --fix --select I .
	uv run --frozen ruff format .

fix: format  ## Alias for auto-formatting and import fixes.

format-check:  ## Verify formatting without mutating the tree.
	uv run --frozen ruff format --check .

lint:  ## Run Ruff lint checks.
	uv run --frozen ruff check .

types:  ## Run Pyright type checks.
	uv run --frozen --extra modal pyright

typecheck: types  ## Backwards-compatible alias.

unit:  ## Run the (GPU-free) test suite.
	uv run --frozen pytest -q

test: unit

quality: lint format-check types

check: quality test  ## CI-safe: verify lint, formatting, types, and tests.

build-image:  ## Build the GPU Docker image.
	docker build -t bumblebee .
