SPHINXBUILD ?= sphinx-build
SOURCEDIR = source
BUILDDIR = build

.PHONY: clean html

clean:
	$(SPHINXBUILD) -M clean $(SOURCEDIR) $(BUILDDIR)

html:
	$(SPHINXBUILD) -M html $(SOURCEDIR) $(BUILDDIR) -W --keep-going
