# Makefile
format:
	black .
	isort .

lint:
	env PYTHONPATH=. pytest --flake8 --pylint --mypy

setup:
	pip install -r requirements.txt
	pre-commit install

