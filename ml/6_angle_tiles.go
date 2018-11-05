package main

import (
	"./lib"
	"github.com/mitroadmaps/gomapinfer/common"

	"bytes"
	"fmt"
	"io/ioutil"
	"math"
	"os"
)

const TILE_SIZE = 4096
const ROAD_WIDTH = 30
const D = 20

type GraphContainer struct {
	Graph *common.Graph
	Index common.GridIndex
	RoadSegments []common.RoadSegment
	EdgeToSegment map[int]common.RoadSegment
}

func main() {
	graphDir := os.Args[1]
	outDir := os.Args[2]

	regions := lib.GetRegions()
	gcs := make(map[string]GraphContainer)
	for _, region := range regions {
		fmt.Printf("reading graph for region %s\n", region.Name)
		graph, err := common.ReadGraph(fmt.Sprintf("%s/%s.graph", graphDir, region.Name))
		if err != nil {
			panic(err)
		}
		roadSegments := graph.GetRoadSegments()
		edgeToSegment := make(map[int]common.RoadSegment)
		for _, rs := range roadSegments {
			for _, edge := range rs.Edges {
				edgeToSegment[edge.ID] = rs
			}
		}
		gcs[region.Name] = GraphContainer{
			graph,
			graph.GridIndex(256),
			roadSegments,
			edgeToSegment,
		}
	}

	fmt.Println("initializing tasks")
	type Task struct {
		Label string
		Region string
		Rect common.Rectangle
	}

	var tasks []Task
	for _, region := range regions {
		for x := -region.RadiusX; x < region.RadiusX; x++ {
			for y := -region.RadiusY; y < region.RadiusY; y++ {
				rect := common.Rectangle{
					common.Point{float64(x), float64(y)}.Scale(TILE_SIZE),
					common.Point{float64(x) + 1, float64(y) + 1}.Scale(TILE_SIZE),
				}
				tasks = append(tasks, Task{
					Label: fmt.Sprintf("%s_%d_%d", region.Name, x, y),
					Region: region.Name,
					Rect: rect,
				})
			}
		}
	}

	processTask := func(task Task, threadID int) {
		gc := gcs[task.Region]
		values := make([][][64]uint8, TILE_SIZE/4)
		for i := range values {
			values[i] = make([][64]uint8, TILE_SIZE/4)
		}
		for i := 0; i < TILE_SIZE/4; i++ {
			for j := 0; j < TILE_SIZE/4; j++ {
				p := task.Rect.Min.Add(common.Point{float64(i*4), float64(j*4)})

				// match to nearest edgepos
				var closestEdge *common.Edge
				var closestDistance float64
				for _, edge := range gcs[task.Region].Index.Search(p.Bounds().AddTol(ROAD_WIDTH)) {
					distance := edge.Segment().Distance(p)
					if distance < ROAD_WIDTH && (closestEdge == nil || distance < closestDistance) {
						closestEdge = edge
						closestDistance = distance
					}
				}
				if closestEdge == nil {
					continue
				}
				pos := closestEdge.ClosestPos(p)

				// get potential RS
				curEdge := pos.Edge
				curRS := gc.EdgeToSegment[curEdge.ID]
				oppositeRS := gc.EdgeToSegment[curEdge.GetOpposite().ID]
				potentialRS := []common.RoadSegment{curRS, oppositeRS}

				// get angles
				var buckets []int
				for _, rs := range potentialRS {
					pos := rs.ClosestPos(p)
					targetPositions := gc.Graph.Follow(common.FollowParams{
						SourcePos: pos,
						Distance: D,
					})

					for _, targetPos := range targetPositions {
						targetPoint := targetPos.Point()
						targetVector := targetPoint.Sub(p)
						edgeVector := targetPos.Edge.Segment().Vector()
						avgVector := targetVector.Scale(1 / targetVector.Magnitude()).Add(edgeVector.Scale(1 / edgeVector.Magnitude()))
						angle := common.Point{1, 0}.SignedAngle(avgVector)
						bucket := int((angle + math.Pi) * 64 / math.Pi / 2)
						if bucket < 0 || bucket > 63 {
							fmt.Printf("bad bucket: %v\n", bucket)
							fmt.Printf("target vector: %v\n", targetVector)
							fmt.Printf("edge vector: %v\n", edgeVector)
							fmt.Printf("avg vector: %v\n", avgVector)
							fmt.Printf("angle: %v\n", angle)
							fmt.Printf("rs length: %v\n", rs.Length())
						}
						buckets = append(buckets, bucket)
					}
				}

				// set targets
				for _, bucket := range buckets {
					for offset := 0; offset < 31; offset++ {
						weight := uint8(math.Pow(0.75, float64(offset)) * 255)
						b1 := (bucket + offset) % 64
						b2 := (bucket - offset + 64) % 64
						if weight > values[i][j][b1] {
							values[i][j][b1] = weight
						}
						if weight > values[i][j][b2] {
							values[i][j][b2] = weight
						}
					}
				}
			}
		}

		var buf bytes.Buffer
		for i := 0; i < TILE_SIZE/4; i++ {
			for j := 0; j < TILE_SIZE/4; j++ {
				buf.Write([]byte(values[i][j][:]))
			}
		}
		if err := ioutil.WriteFile(fmt.Sprintf("%s/%s.bin", outDir, task.Label), buf.Bytes(), 0644); err != nil {
			panic(err)
		}
	}

	fmt.Println("launching workers")
	n := 36
	taskCh := make(chan Task)
	doneCh := make(chan bool)
	for threadID := 0; threadID < n; threadID++ {
		go func(threadID int) {
			for task := range taskCh {
				processTask(task, threadID)
			}
			doneCh <- true
		}(threadID)
	}
	fmt.Println("running tasks")
	for i, task := range tasks {
		if i % 10 == 0 {
			fmt.Printf("... task progress: %d/%d\n", i, len(tasks))
		}
		taskCh <- task
	}
	close(taskCh)
	for threadID := 0; threadID < n; threadID++ {
		<- doneCh
	}
}
