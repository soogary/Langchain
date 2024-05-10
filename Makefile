install:
	pip install -r requirements.txt


test:
#	python -m pytest -vv test_hello.py
#	python -m pytest -vv --cov=nlp_textblob test_nlp_textblob.py


format:
	black *.py

lint:
#	pylint --disable=R,C --disable=invalid-name hello.py
#	pylint --disable=invalid-name fire-cli.py

all:  
	install lint test