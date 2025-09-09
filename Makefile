install:
	poetry install

test:
	poetry run pytest

lint:
	poetry run ruff check src/ examples/

format:
	poetry run ruff format src/ examples/
