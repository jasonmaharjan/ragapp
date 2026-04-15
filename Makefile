SHELL := /bin/bash

# assume venv is already activated
install:
	pip install -r requirements.txt 

run:
	cd server && uvicorn main:app --reload

test:
	cd server && python test.py

eval:
	python scripts/eval.py
