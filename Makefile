test:
	python -m pytest --pyargs --doctest-modules turf
coverage-test:
	coverage run -m pytest --pyargs --doctest-modules turf	