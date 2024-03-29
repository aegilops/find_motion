all: flake8 mypy bandit diatra safety vulture pyright dlint pytype
.PHONY: flake8 mypy all bandit diatra safety vulture pyright dlint pytype install dev-requirements

flake8:
	flake8 --ignore=E501,E303,E402,E252

mypy:
	mypy --ignore-missing-imports .

bandit:
	bandit --quiet --silent -r .

diatra:
	python3 -m pydiatra .

safety:
	safety check --bare -r requirements.txt

vulture:
	-vulture .

dlint:
	flake8 --select=DUO .

dev-requirements:
	python3 -m pip install pip --upgrade --user
	python3 -m pip install -r make_requirements.txt --upgrade --user

install:
	python3 -m pip install pip --upgrade --user
	python3 -m pip install . --user

