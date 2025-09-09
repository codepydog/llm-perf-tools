install:
	poetry install

test:
	poetry run pytest

lint:
	poetry run ruff check src/
