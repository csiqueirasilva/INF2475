# INF2475

Repository with exercises from a data science class.

# python

- using python 3.12.3 on ubuntu 24.04

# poetry installation

```
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
source ~/.bashrc
```

check installation with:

```
poetry --version
```

after that do

```
poetry install
```

if using vscode and you run into import issues, set the interpreter path to the python binary in the venv created by poetry

# tests

run tests with

```
poetry run pytest
```