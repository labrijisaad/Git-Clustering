# Author information
AUTHOR := Labriji Saad

# Default target when no arguments are provided to make
.DEFAULT_GOAL := help

# Token and repository configuration
CONFIGURE_REPO = poetry config repositories.test-pypi https://test.pypi.org/legacy/
CONFIGURE_TOKEN = poetry config pypi-token.test-pypi xxx

.PHONY: configure install build publish test help

configure:
	@$(CONFIGURE_REPO)
	@$(CONFIGURE_TOKEN)

install:
	@poetry install

build:
	@poetry build

publish:
	@poetry publish -r test-pypi

test:
	@pytest tests/test_git_clustering.py

# Display help with available make targets
help:
	@echo  Available targets:
	@echo    configure   - Configure test-pypi repository and token
	@echo    install     - Install dependencies
	@echo    build       - Build the project
	@echo    publish     - Publish to test-pypi
	@echo    test        - Run tests
	@echo    help        - Display this help message
	@echo  Author: $(AUTHOR)