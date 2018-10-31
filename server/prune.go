package main

// cell-based version
// * split region into CELL_SIZE x CELL_SIZE grid cells
// * pick node closest to center in each grid cell
// * iteratively find shortest paths between cell centers that are CELL_DISTANCE cells apart (Manhattan distance)
// * cut PADDING off path and retain the rest
// * also retain edges based on shortest paths from vertices that interface between existing and inferred maps
// * some post-processing steps to remove some edges

import (
	"github.com/mitroadmaps/gomapinfer/common"

	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"sync"
)

const CELL_SIZE = 512
const PADDING = 1024
const CELL_DISTANCE = 10

func abs(x int) int {
	if x < 0 {
		return -x
	} else {
		return x
	}
}

func manh(c1 [2]int, c2 [2]int) int {
	return abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])
}

func main() {
	var inName, probsName, ifaceName, outName string
	if len(os.Args) == 3 {
		inName = os.Args[1]
		outName = os.Args[2]
	} else if len(os.Args) == 4 {
		inName = os.Args[1]
		probsName = os.Args[2]
		outName = os.Args[3]
	} else if len(os.Args) == 5 {
		inName = os.Args[1]
		probsName = os.Args[2]
		ifaceName = os.Args[3]
		outName = os.Args[4]
	} else {
		fmt.Println("usage: centrality3 in-name [probs-name [iface-name]] out-name")
		return
	}
	graph, err := common.ReadGraph(inName)
	if err != nil {
		panic(err)
	}
	var edgeProbs []int
	var ifaceNodes map[int]bool
	if probsName != "" {
		edgeProbs = func() []int {
			bytes, err := ioutil.ReadFile(probsName)
			if err != nil {
				panic(err)
			}
			var edgeProbs []int
			if err := json.Unmarshal(bytes, &edgeProbs); err != nil {
				panic(err)
			}
			return edgeProbs
		}()
	} else {
		for _ = range graph.Edges {
			edgeProbs = append(edgeProbs, 50)
		}
	}
	if ifaceName != "" {
		ifaceNodes = func() map[int]bool {
			bytes, err := ioutil.ReadFile(ifaceName)
			if err != nil {
				panic(err)
			}
			var ifaceList []int
			if err := json.Unmarshal(bytes, &ifaceList); err != nil {
				panic(err)
			}
			ifaceNodes := make(map[int]bool)
			for _, nodeID := range ifaceList {
				ifaceNodes[nodeID] = true
			}
			return ifaceNodes
		}()
	}

	rsGraph, edgeToRS, rsNodeMap := graph.GetRoadSegmentGraph()
	// compute edge lengths in rsGraph for shortest path
	// the length of each edge in the rs is scaled based on edgeProbs
	rsLengths := make(map[int]float64)
	for _, rsEdge := range rsGraph.Edges {
		rs := edgeToRS[rsEdge.ID]
		var length float64 = 0
		for _, edge := range rs.Edges {
			length += edge.Segment().Length() * (2 - float64(edgeProbs[edge.ID]) /  100)
		}
		rsLengths[rsEdge.ID] = length
	}
	// convert ifaceNodes to nodes in rsGraph
	rsIfaceNodes := make(map[int]bool)
	for nodeID := range ifaceNodes {
		if rsNodeMap[nodeID] == nil {
			continue
		}
		rsIfaceNodes[rsNodeMap[nodeID].ID] = true
	}

	// find cell centers for rsGraph
	fmt.Println("finding cell centers")
	cellNodes := make(map[[2]int][]*common.Node)
	for _, node := range rsGraph.Nodes {
		cell := [2]int{
			int(math.Floor(node.Point.X / CELL_SIZE)),
			int(math.Floor(node.Point.Y / CELL_SIZE)),
		}
		cellNodes[cell] = append(cellNodes[cell], node)
	}
	cellCenters := make(map[[2]int]*common.Node)
	for cell, nodes := range cellNodes {
		if len(nodes) < 8 {
			continue
		}
		p := common.Point{
			(float64(cell[0]) + 0.5) * CELL_SIZE,
			(float64(cell[1]) + 0.5) * CELL_SIZE,
		}
		var bestNode *common.Node
		var bestDistance float64
		for _, node := range nodes {
			d := node.Point.Distance(p)
			if bestNode == nil || d < bestDistance {
				bestNode = node
				bestDistance = d
			}
		}
		cellCenters[cell] = bestNode
	}

	// get shortest paths
	fmt.Println("computing shortest paths between centers")
	nthreads := 6
	goodRSEdges := addBetweenClusters(rsGraph, rsLengths, cellCenters, nthreads)

	// postprocessing: add shortest path from iface nodes to goodEdges
	fmt.Println("adding shortest paths from iface nodes")
	addFromIface(rsGraph, rsLengths, goodRSEdges, rsIfaceNodes, nthreads)

	// convert goodRSEdges to edges in the original graph
	goodEdges := make(map[int]bool)
	for edgeID := range goodRSEdges {
		rs := edgeToRS[edgeID]
		for _, edge := range rs.Edges {
			goodEdges[edge.ID] = true
		}
	}

	// filter bad edges
	badEdges := make(map[int]bool)
	for _, edge := range graph.Edges {
		if !goodEdges[edge.ID] {
			badEdges[edge.ID] = true
		}
	}
	origEdges := len(graph.Edges)
	graph, nodeMap, _ := graph.FilterEdgesWithMaps(badEdges)
	fmt.Printf("filter from %d to %d edges\n", origEdges, len(graph.Edges))
	graph.MakeBidirectional()

	// update ifaceNodes for new nodes after filtering
	newIfaceNodes := make(map[int]bool)
	for nodeID := range ifaceNodes {
		if nodeMap[nodeID] == nil {
			continue
		}
		newIfaceNodes[nodeMap[nodeID].ID] = true
	}

	// more postprocessing: remove dead-end non-iface road segments
	fmt.Println("removing dead-ends")
	roadSegments := graph.GetRoadSegments()
	badEdges = make(map[int]bool)
	for _, rs := range roadSegments {
		var deadEndNode *common.Node
		if len(rs.Src().Out) == 1 {
			deadEndNode = rs.Src()
		} else if len(rs.Dst().Out) == 1 {
			deadEndNode = rs.Dst()
		} else {
			continue
		}
		if newIfaceNodes[deadEndNode.ID] || rs.Length() > CELL_SIZE {
			continue
		}
		for _, edge := range rs.Edges {
			badEdges[edge.ID] = true
		}
	}
	graph = graph.FilterEdges(badEdges)

	if err := graph.Write(outName); err != nil {
		panic(err)
	}
}

func addBetweenClusters(graph *common.Graph, edgeLengths map[int]float64, cellCenters map[[2]int]*common.Node, nthreads int) map[int]bool {
	type job struct {
		cell [2]int
		center *common.Node
	}
	jobch := make(chan job)
	donech := make(chan map[int]bool)
	for i := 0; i < nthreads; i++ {
		go func() {
			goodEdges := make(map[int]bool)
			for job := range jobch {
				result := graph.ShortestPath(job.center, common.ShortestPathParams{
					MaxDistance: CELL_SIZE * 2 * CELL_DISTANCE * 1.25,
					EdgeLengths: edgeLengths,
				})
				for cell, center := range cellCenters {
					if manh(job.cell, cell) != CELL_DISTANCE {
						continue
					}
					if result.Remaining[center.ID] {
						continue
					}
					if _, ok := result.Distances[center.ID]; !ok {
						continue
					}
					path := result.GetFullPathTo(center)
					var totalDistance float64 = 0
					for i := 0; i < len(path) - 1; i++ {
						totalDistance += path[i].Point.Distance(path[i + 1].Point)
					}
					var curDistance float64 = 0
					for i := 0; i < len(path) - 1; i++ {
						if curDistance >= PADDING {
							rsEdge := graph.FindEdge(path[i], path[i + 1])
							goodEdges[rsEdge.ID] = true
						}
						curDistance += path[i].Point.Distance(path[i + 1].Point)
						if curDistance >= totalDistance - PADDING {
							break
						}
					}
				}
			}
			donech <- goodEdges
		}()
	}
	count := 0
	for cell, center := range cellCenters {
		jobch <- job{cell, center}
		fmt.Printf("... %d/%d\n", count, len(cellCenters))
		count++
	}
	close(jobch)
	goodEdges := make(map[int]bool)
	for i := 0; i < nthreads; i++ {
		m := <- donech
		for k := range m {
			goodEdges[k] = true
		}
	}
	return goodEdges
}

func addFromIface(graph *common.Graph, edgeLengths map[int]float64, goodEdges map[int]bool, ifaceNodes map[int]bool, nthreads int) {
	jobch := make(chan *common.Node)
	donech := make(chan bool)
	var mu sync.Mutex
	goodNodes := make(map[int]bool)
	for edgeID := range goodEdges {
		edge := graph.Edges[edgeID]
		goodNodes[edge.Src.ID] = true
		goodNodes[edge.Dst.ID] = true
	}
	for i := 0; i < nthreads; i++ {
		go func() {
			for node := range jobch {
				result := graph.ShortestPath(node, common.ShortestPathParams{
					MaxDistance: CELL_SIZE * 2,
					EdgeLengths: edgeLengths,
				})
				mu.Lock()
				var bestDst *common.Node
				var bestDistance float64
				for otherID, distance := range result.Distances {
					if result.Remaining[otherID] || !goodNodes[otherID] {
						continue
					}
					if bestDst == nil || distance < bestDistance {
						bestDst = graph.Nodes[otherID]
						bestDistance = distance
					}
				}
				if bestDst == nil {
					mu.Unlock()
					continue
				}
				path := result.GetFullPathTo(bestDst)
				for i := 0; i < len(path) - 1; i++ {
					edge := graph.FindEdge(path[i], path[i + 1])
					goodEdges[edge.ID] = true
					goodNodes[edge.Src.ID] = true
					goodNodes[edge.Dst.ID] = true
				}
				mu.Unlock()
			}
			donech <- true
		}()
	}
	count := 0
	for nodeID := range ifaceNodes {
		jobch <- graph.Nodes[nodeID]
		if count % 10 == 0 {
			fmt.Printf("... %d/%d\n", count, len(ifaceNodes))
		}
		count++
	}
	close(jobch)
	for i := 0; i < nthreads; i++ {
		<- donech
	}
}
