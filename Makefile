# Formatting
format-black:
	@echo Formatting with black ...
	@black blenderline

format-isort:
	@echo.
	@echo Formatting with isort ...
	@isort blenderline

format-project: format-black format-isort