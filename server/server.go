package main

import (
	"github.com/mitroadmaps/gomapinfer/common"

	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"sort"
	"strconv"
)

const DIAG_THRESHOLD = 800
const RADIUS = 512

type OsmObject struct {
	ID string `json:"id"`
	Location []float64 `json:"loc,omitempty"`
	Nodes []string `json:"nodes,omitempty"`
	Visible bool `json:"visible"`
	Tags map[string]string `json:"tags,omitempty"`
}

type Response struct {
	MLFull []OsmObject `json:"ml_full"`
	MLPruned []OsmObject `json:"ml_pruned"`
	Jumps [][][]float64 `json:"jumps"`
}

type Component struct {
	Rect common.Rectangle
	Edges []*common.Edge
}

func NewComponent(edges []*common.Edge) Component {
	r := edges[0].Segment().Bounds()
	for _, edge := range edges[1:] {
		r = r.ExtendRect(edge.Segment().Bounds())
	}
	return Component{r, edges}
}

func (c Component) Graph() *common.Graph {
	g := &common.Graph{}
	nodeMap := make(map[int]*common.Node)
	for _, edge := range c.Edges {
		for _, node := range []*common.Node{edge.Src, edge.Dst} {
			if nodeMap[node.ID] == nil {
				nodeMap[node.ID] = g.AddNode(node.Point)
			}
		}
		g.AddBidirectionalEdge(nodeMap[edge.Src.ID], nodeMap[edge.Dst.ID])
	}
	return g
}

type ComponentScore struct {
	Component Component
	Score float64
}

type ComponentScores []ComponentScore

func (p ComponentScores) Len() int           { return len(p) }
func (p ComponentScores) Less(i, j int) bool { return p[i].Score > p[j].Score }
func (p ComponentScores) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

func floodfill(node *common.Node, seenEdges map[int]bool, componentEdges *[]*common.Edge) {
	for _, edge := range node.Out {
		if seenEdges[edge.ID] {
			continue
		}
		seenEdges[edge.ID] = true
		*componentEdges = append(*componentEdges, edge)
		floodfill(edge.Dst, seenEdges, componentEdges)
	}
}

func main() {
	if len(os.Args) < 2 {
		fmt.Printf("usage: %s <graph fname> [pruned fname]\n", os.Args[0])
		return
	}

	fmt.Println("loading graph")
	graph, err := common.ReadGraph(os.Args[1])
	if err != nil {
		panic(err)
	}
	origin := graph.Bounds().Min
	fmt.Println("simplifying graph")
	graph.MakeBidirectional()
	simpleGraph := simplify(graph)
	response := Response{
		MLFull: objects(simpleGraph),
	}
	fmt.Println("loading pruned graph")
	if len(os.Args) >= 3 {
		graph, err := common.ReadGraph(os.Args[2])
		if err != nil {
			panic(err)
		}
		graph.MakeBidirectional()
		response.MLPruned = objects(simplify(graph))
	}
	fmt.Println("getting connected components")
	var components []Component
	seenEdges := make(map[int]bool)
	for _, edge := range graph.Edges {
		if seenEdges[edge.ID] {
			continue
		}
		componentEdges := new([]*common.Edge)
		floodfill(edge.Src, seenEdges, componentEdges)
		components = append(components, NewComponent(*componentEdges))
	}
	fmt.Println("computing scores")
	var scores ComponentScores
	for _, component := range components {
		p1 := component.Rect.Min.LonLatToMeters(origin)
		p2 := component.Rect.Max.LonLatToMeters(origin)
		score := p1.Distance(p2)
		if score < DIAG_THRESHOLD {
			continue
		}
		scores = append(scores, ComponentScore{component, score})
	}
	sort.Sort(scores)
	seenCells := make(map[[2]int]bool)
	for _, score := range scores {
		component := score.Component
		r := component.Rect
		rmin := r.Min.LonLatToMeters(origin)
		rmax := r.Max.LonLatToMeters(origin)
		sx, sy, ex, ey := int(rmin.X)/RADIUS - 1, int(rmin.Y)/RADIUS - 1, int(rmax.X)/RADIUS + 1, int(rmax.Y)/RADIUS + 1
		seen := false
		for i := sx; i <= ex; i++ {
			for j := sy; j <= ey; j++ {
				if seenCells[[2]int{i, j}] {
					seen = true
				}
			}
		}
		if seen && false {
			continue
		}
		for i := sx; i <= ex; i++ {
			for j := sy; j <= ey; j++ {
				seenCells[[2]int{i, j}] = true
			}
		}
		frect := [][]float64{[]float64{r.Min.X, r.Min.Y}, []float64{r.Max.X, r.Max.Y}}
		response.Jumps = append(response.Jumps, frect)
	}
	fmt.Printf("got %d components, %d after thresholding, %d disjoint\n", len(components), len(scores), len(response.Jumps))

	http.HandleFunc("/ml", func(w http.ResponseWriter, r *http.Request) {
		params := r.URL.Query()
		x1, _ := strconv.ParseFloat(params.Get("x1"), 64)
		y1, _ := strconv.ParseFloat(params.Get("y1"), 64)
		x2, _ := strconv.ParseFloat(params.Get("x2"), 64)
		y2, _ := strconv.ParseFloat(params.Get("y2"), 64)
		log.Printf("/ml %v %v %v %v", x1, y1, x2, y2)
		bytes, err := json.Marshal(response)
		if err != nil {
			panic(err)
		}
		w.Write(bytes)
	})
	fmt.Println("ready")
	log.Fatal(http.ListenAndServe(":8081", nil))
}

func simplify(graph *common.Graph) *common.Graph {
	origin := graph.Bounds().Min
	roadSegments := graph.GetRoadSegments()
	ngraph := &common.Graph{}
	nodeMap := make(map[int]*common.Node)
	seenNodePairs := make(map[[2]int]bool)
	for _, rs := range roadSegments {
		src := rs.Src()
		dst := rs.Dst()
		if seenNodePairs[[2]int{src.ID, dst.ID}] || seenNodePairs[[2]int{dst.ID, src.ID}] {
			continue
		}
		for _, node := range []*common.Node{src, dst} {
			if nodeMap[node.ID] == nil {
				nodeMap[node.ID] = ngraph.AddNode(node.Point)
			}
		}
		points := []common.Point{src.Point.LonLatToMeters(origin)}
		for _, edge := range rs.Edges {
			points = append(points, edge.Dst.Point.LonLatToMeters(origin))
		}
		points = common.RDP(points, 10)
		nodes := []*common.Node{nodeMap[src.ID]}
		for _, point := range points[1:len(points)-1] {
			nodes = append(nodes, ngraph.AddNode(point.MetersToLonLat(origin)))
		}
		nodes = append(nodes, nodeMap[dst.ID])
		for i := 0; i < len(nodes) - 1; i++ {
			ngraph.AddBidirectionalEdge(nodes[i], nodes[i + 1])
		}
		seenNodePairs[[2]int{src.ID, dst.ID}] = true
	}
	return ngraph
}

func objects(graph *common.Graph) []OsmObject {
	osmObjects := make([]OsmObject, 0) // suitable for json response
	for _, node := range graph.Nodes {
		osmObjects = append(osmObjects, OsmObject{
			ID: fmt.Sprintf("n%d", -node.ID - 1),
			Location: []float64{node.Point.X, node.Point.Y},
			Visible: true,
		})
	}
	seenNodePairs := make(map[[2]int]bool)
	idCounter := -99999
	for _, rs := range graph.GetRoadSegments() {
		srcID := rs.Src().ID
		dstID := rs.Dst().ID
		if seenNodePairs[[2]int{srcID, dstID}] || seenNodePairs[[2]int{dstID, srcID}] {
			continue
		}
		seenNodePairs[[2]int{srcID, dstID}] = true
		wayNodes := []string{fmt.Sprintf("n%d", -rs.Edges[0].Src.ID - 1)}
		for _, edge := range rs.Edges {
			wayNodes = append(wayNodes, fmt.Sprintf("n%d", -edge.Dst.ID - 1))
		}
		osmObjects = append(osmObjects, OsmObject{
			ID: fmt.Sprintf("w%d", idCounter),
			Nodes: wayNodes,
			Visible: true,
			Tags: map[string]string{"highway": "residential"},
		})
		idCounter--
	}
	return osmObjects
}
