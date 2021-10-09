# icup22

Run GUI with `python -m tools.gui`. Run generator with `python -m tools.generator`.

This script runs all the tests of the pipeline. Add it to `.git/hooks/pre-commit` and make it executable. Then all the
checks will be performed before commiting.

```bash
#!/bin/bash

# Exit if one of the following commands fails
set -e

# Execute Linter
pylint --disable=r $(git ls-files '*.py')

# Execute tests
pytest
```

## Solution algorithms

### SimpleSolver

We only use a train with maximum capacity to transport all passengers.

If we want to go to a station with full capacity, we move a train from that station somewhere else.

Basic algorithm:
0. put wildcard trains somewhere (if possible, don't max out any station capacity)
1. select train we want to transport passengers with
2. select one group closest to train (tie-breaker: largest group)
3. use short path to transport this group to destination (when we want to drive to a full station, move fastest train
   from there to the station we started at in the same turn our passenger train arrives - always possible, even with
   line capacity 1)
4. if all passenger groups are at their destination, we're done
5. go to step 2
