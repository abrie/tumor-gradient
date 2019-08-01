install:
	@( \
		python3 -m venv $(PWD); \
		source ./bin/activate; \
		pip install -r requirements.txt; \
		)

test:
	@( \
		python3 -m venv $(PWD); \
		source ./bin/activate; \
		python3 test.py; \
		)

run:
	@( \
		python3 -m venv $(PWD); \
		source ./bin/activate; \
		python3 main.py; \
		)
