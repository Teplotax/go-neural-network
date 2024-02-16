package main

import (
	"fmt"
	"time"
)

func main() {
	start := time.Now()
	defer microsecondsSince(start)

	inputs := []float32{2, 2, 3, 2.5}
	weights := [][]float32{
		{0.2, 0.8, -0.5, 1.0},
		{0.5, -0.91, 0.26, -0.5},
		{-0.26, -0.27, 0.17, 0.87},
	}
	bias := []float32{2, 3, 0.5}

	solution01(inputs, weights, bias)
	solution02(inputs, weights, bias)

}

func solution01(inputs []float32, weights [][]float32, bias []float32) {

	layerOutputs := []float32{}

	for i := 0; i < len(bias); i++ {
		output := bias[i]

		for j := 0; j < len(inputs); j++ {
			output += inputs[j] * weights[i][j]
		}

		layerOutputs = append(layerOutputs, output)
	}

	fmt.Println(layerOutputs)
}

func solution02(inputs []float32, weights [][]float32, bias []float32) {

	layerOutputs := make([]float32, len(bias))

	for i, b := range bias {

		neuronWeights := weights[i]

		var neuronOutput float32 = 0

		for j := range inputs {
			neuronOutput += inputs[j] * neuronWeights[j]
		}

		neuronOutput += b
		layerOutputs[i] = neuronOutput
	}

	fmt.Println(layerOutputs)
}

func microsecondsSince(start time.Time) {
	duration := time.Since(start)
	fmt.Println("Execution time: ", duration.Microseconds(), "μs")
}
