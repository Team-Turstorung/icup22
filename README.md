# icup22

Run `main.py --help` to see everything that is available.

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

### SimpleSolution

We only use a train with maximum capacity to transport all passengers.

If we want to go to a station with full capacity, we move a train from that station somewhere else.

Basic algorithm:
0. put wildcard trains somewhere (if possible, don't max out any station capacity)
1. select train we want to transport passengers with
2. select one group closest to train
3. use short path to transport this group to destination
4. if all passenger groups are at their destination, we're done
5. go to step 2

## Ideas for solution

Give passengers priorities calculated like this: `|ShortestPath(Position, Destination)| / TimeRemaining`
This could make it more easy to prioritize which Passengers to transport first. (Should this priority be updated every round? If so what happens if TimeRemainig is zero or negative?)

Reduce Graph to nodes and edges on shortest paths --> could be problematic with traffic jam at central points (But should be easy to get subgraph with networkx)

Alternative betweenness centrality. `|v is in shortest path between start and destination of passenger| / number of passengers` Would indicate the importance of a station and possible problems with capacity. We could move unused Trains to stations with low betweenness.

Try to embedd shortest paths into each other. If shortest path for one passenger is subset of shortest path of other passenger try to transport them together.

Compute intersect between all shortest paths to get possible interchanges.
