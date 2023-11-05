test:
	python -m pytest --pyargs --doctest-modules turf

test-coverage:
	python -m pytest --pyargs --doctest-modules --cov=turf --cov-report term turf

test-coverage-html:
	python -m pytest --pyargs --doctest-modules --cov=turf --cov-report html turf
