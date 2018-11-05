MAiD
====

iD Modifications
----------------

Run ML, Jump, and Prune buttons are added to the iD editor. For iD README, see README.iD.md.

These buttons depend on a server, in server/ directory:

* server.go: start a server to respond to "Run ML" and "Prune" requests. Our implementation replies with the same graph every time, but this could be extended to run automatic map inference in the region corresponding to the user's current viewport.
* prune.go: apply the shortest-path-based pruning algorithm.
* convert.go: convert from pixel coordinates (e.g. indonesia-raw.graph) to longitude/latitude coordinates (e.g. indonesia-full.graph)
* indonesia-raw.graph: unprocessed output from automatic map inference algorithm. Coordinates correspond to pixels in the imagery. indonesia-raw.iface contains vertex IDs of interface junctions where some incident edges are in the existing map while some are inferred segments. indonesia-raw.probs contains a probability/confidence for each edge that comes from the inference algorithm.
* indonesia-full.graph: indonesia-raw.graph converted to longitude/latitude coordinates.
* indonesia-pruned.graph: pruned version, also longitude/latitude.
* washington.graph: graph from high-coverage rural Washington region (longitude/latitude).

To generate indonesia-pruned.graph:

	$ go run prune.go indonesia-raw.graph indonesia-raw.iface indonesia-raw.probs indonesia-raw-pruned.graph
	$ go run convert.go indonesia-raw-pruned.graph indonesia-pruned.graph

To run server for Indonesia region:

	$ go run server.go indonesia-full.graph indonesia-pruned.graph

To run server for Washington region:

	$ go run server.go washington.graph

In user studies we removed left panel, made all roads (both existing map and materialized inferred segments) show up as "highway=residential", removed right panel, removed point/area/undo/redo (but kept Line). Also no Run ML or Prune buttons, when session starts the inferred segments are already there.

Map Inference for Interactive Mapping
-------------------------------------

To run our automatic map inference algorithm:

1. Get RoadTracer from https://github.com/mitroadmaps/roadtracer/.
2. Copy ml/tileloader.py, ml/maid_infer.py, ml/maid_model.py, and ml/maid_train.py from this repository to roadtracer/roadtracer/.
3. Copy ml/6_angle_tiles.go to roadtracer/dataset/.
4. Follow instructions in roadtracer/dataset/.

Now, you can run:

	cd roadtracer/dataset
	mkdir /data/angles/
	go run 6_angle_tiles.go /data/graphs/ /data/angles/
	cd ../../roadtracer/roadtracer/
	mkdir /data/maid-model/ /data/maid-model/model_latest /data/maid-model/model_best
	python maid_train.py --t /data/imagery/ --g /data/graphs/ --a /data/angles/ --j /data/json/ /data/maid-model/
	python maid_infer.py --t /data/imagery --g /data/graphs/ --r chicago --s 0.4 --j /data/json/ /data/maid-model/model_best/model out.graph
