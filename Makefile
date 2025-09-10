install:
	poetry install

test:
	poetry run pytest -vv tests \
	--cov=. \
	--cov-report=term \
	--cov-report=xml:coverage.xml

test-docs:
	poetry run pytest --doctest-modules src/llm_perf_tools/

lint:
	poetry run ruff check src/ examples/

format:
	poetry run ruff format src/ examples/

run-single:
	poetry run python examples/ollama-tracker-single-example.py

run-batch:
	poetry run python examples/ollama-tracker-batch-example.py

run-single-dev:
	PYTHONPATH=src python examples/ollama-tracker-single-example.py

run-batch-dev:
	PYTHONPATH=src python examples/ollama-tracker-batch-example.py

docs:
	poetry install --with docs
	poetry run sphinx-build docs docs/_build
