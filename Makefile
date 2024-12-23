python_version=3.12
groups=dev

setup:
	pyenv local $(python_version)
	pip install poetry
	poetry config virtualenvs.create true
	poetry config virtualenvs.in-project true
	poetry env use $(python_version)

install:
	poetry install --with $(groups) --all-extras

mlflow-server:
	mlflow server \
		--host 127.0.0.1 \
		--port 5000
		