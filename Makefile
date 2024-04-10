test:
	python -m pytest --pyargs --doctest-modules tests
coverage-test:
	coverage run -m pytest --pyargs --doctest-modules src/turf	