package main

import (
	"github.com/mitroadmaps/gomapinfer/common"
	"github.com/mitroadmaps/gomapinfer/googlemaps"

	"os"
)

const ZOOM = 18
//const ORIGIN_X = 40960
//const ORIGIN_Y = 92031
const ORIGIN_X = 211874
const ORIGIN_Y = 136405

func main() {
	var originTile = [2]int{ORIGIN_X, ORIGIN_Y}
	g, err := common.ReadGraph(os.Args[1])
	if err != nil {
		panic(err)
	}
	for _, node := range g.Nodes {
		node.Point = googlemaps.MapboxToLonLat(node.Point, ZOOM, originTile)
	}
	if err := g.Write(os.Args[2]); err != nil {
		panic(err)
	}
}

