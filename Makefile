# Formatting
format-black:
	@echo Formatting with black ...
	@black src

format-isort:
	@echo.
	@echo Formatting with isort ...
	@isort src

format-project: format-black format-isort