# Licensed under the MIT License
# https://github.com/craigahobbs/ollama-chat/blob/main/LICENSE


# Set gh-pages source
GHPAGES_SRC := build/doc/


# Include python-build
include Makefile.base


# Development dependencies
TESTS_REQUIRE := bare-script


# Disable pylint docstring warnings
PYLINT_ARGS := $(PYLINT_ARGS) static/models --disable=missing-class-docstring --disable=missing-function-docstring --disable=missing-module-docstring


# Don't delete models.json in gh-pages branch
GHPAGES_RSYNC_ARGS := --exclude='models/models.json'


help:
	@echo "            [run|test-app|venv-rebuild]"


doc:
	rm -rf $(GHPAGES_SRC)
	mkdir -p $(GHPAGES_SRC)
	cp -R \
		README.md \
		static/* \
		src/ollama_chat/static/ollamaChat.smd \
		$(GHPAGES_SRC)


.PHONY: test-app
commit: test-app
test-app: $(DEFAULT_VENV_BUILD)
	$(DEFAULT_VENV_BIN)/bare -x -m src/ollama_chat/static/*.bare src/ollama_chat/static/test/*.bare
	$(DEFAULT_VENV_BIN)/bare -d -m src/ollama_chat/static/test/runTests.bare$(if $(TEST), -v vUnittestTest "'$(TEST)'")


.PHONY: venv-rebuild
venv-rebuild:
	rm -f $(DEFAULT_VENV_BUILD)
	$(MAKE) $(DEFAULT_VENV_BUILD)


.PHONY: run
run: $(DEFAULT_VENV_BUILD)
	$(DEFAULT_VENV_BIN)/ollama-chat$(if $(ARGS), $(ARGS))
