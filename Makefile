all: flake8 mypy bandit diatra safety vulture pyright dlint pytype
.PHONY: flake8 mypy all bandit diatra safety vulture pyright dlint pytype install

flake8:
	flake8 --ignore=E501,E303,E402,E252

mypy:
	mypy --ignore-missing-imports .

bandit:
	bandit -r .

diatra:
	python3 -m pydiatra .

safety:
	safety check

vulture:
	vulture .

pyright:
	pyright

dlint:
	flake8 --select=DUO .

pytype:
	pytype .

install:
	python3 -m pip install .

