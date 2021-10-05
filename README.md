# icup22

Run GUI with `python -m tools.gui`. Run generator with `python -m tools.generator`.

This script runs all the tests of the pipeline. Add it to `.git/hooks/pre-commit` and make it executable. Then all the checks will be performed before commiting.
```
#!/bin/bash

# Exit if one of the following commands fails
set -e

# Execute Linter
pylint --disable=r $(git ls-files '*.py')

# Execute tests
pytest
```
