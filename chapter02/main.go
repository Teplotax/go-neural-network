package main

import (
	"fmt"
	"time"

	t "gorgonia.org/tensor"
)

func main() {
	start := time.Now()
	defer microsecondsSince(start)

	inputs := t.New(t.WithBacking([]float32{1, 2, 3, 2.5}))

	rawWeights := []float32{
		0.2, 0.8, -0.5, 1.0,
		0.5, -0.91, 0.26, -0.5,
		-0.26, -0.27, 0.17, 0.87,
	}

	weights := t.New(t.WithShape(3, 4), t.WithBacking(rawWeights))

	bias := t.New(t.WithBacking([]float32{2, 3, 0.5}))

	dotProduct, _ := t.Dot(weights, inputs)
	neuronOutput, _ := t.Add(dotProduct, bias)

	fmt.Println(neuronOutput)

}

func microsecondsSince(start time.Time) {
	duration := time.Since(start)
	fmt.Println("Execution time: ", duration.Microseconds(), "μs")
}
