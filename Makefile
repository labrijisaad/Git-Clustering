# Token and repository configuration
CONFIGURE_REPO = poetry config repositories.test-pypi https://test.pypi.org/legacy/
CONFIGURE_TOKEN = poetry config pypi-token.test-pypi xxx

configure:
	@$(CONFIGURE_REPO)
	@$(CONFIGURE_TOKEN)

install:
	@poetry install

build:
	@poetry build

publish:
	@poetry publish -r test-pypi