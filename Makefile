install:
	( \
		python3 -m venv $(PWD); \
		source ./bin/activate; \
		pip install -r requirements.txt; \
		)
