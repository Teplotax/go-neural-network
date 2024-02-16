package main

import (
	"fmt"
	"time"

	t "gorgonia.org/tensor"
)

func main() {
	start := time.Now()
	defer microsecondsSince(start)

	// var inputs t.Tensor = t.New(t.WithShape(1, 4), t.WithBacking([]float32{1, 2, 3, 2.5}))
	// bias := t.New(t.WithBacking([]float32{2, 3, 0.5}))

	// rawWeights := []float32{
	// 	0.2, 0.8, -0.5, 1.0,
	// 	0.5, -0.91, 0.26, -0.5,
	// 	-0.26, -0.27, 0.17, 0.87,
	// }

	// weights := t.New(t.WithShape(bias.Size(), inputs.Size()), t.WithBacking(rawWeights))

	// dotProduct, _ := t.Dot(weights, inputs)
	// neuronOutput, _ := t.Add(dotProduct, bias)

	// fmt.Println(neuronOutput)
	rawInputs := []float32{
		1.0, 2.0, 3.0, 2.5,
		2.0, 5.0, -1, 2.0,
		-1.5, 2.7, 3.3, -0.8,
	}
	rawWeights := []float32{
		0.2, 0.8, -0.5, 1.0,
		0.5, -0.91, 0.26, -0.5,
		-0.26, -0.27, 0.17, 0.87,
	}
	rawBiases := []float32{2.0, 3.0, 0.5}

	inputs := t.New(t.WithShape(3, 4), t.WithBacking(rawInputs))
	weights := t.New(t.WithShape(3, 4), t.WithBacking(rawWeights))
	biases := biasTensor(rawBiases, inputs.Shape()[0])
	weights.T()

	dP, _ := t.Dot(inputs, weights)

	outputs, err := t.Add(dP, biases)
	if err != nil {
		fmt.Println(err)
	}

	fmt.Println(outputs)
}

func microsecondsSince(start time.Time) {
	duration := time.Since(start)
	fmt.Println("Execution time: ", duration.Microseconds(), "μs")
}

func biasTensor(rawBiases []float32, rows int) t.Tensor {
	newRawBiases := []float32{}

	for i := 0; i < rows; i++ {
		newRawBiases = append(newRawBiases, rawBiases...)
	}

	return t.New(t.WithShape(rows, len(rawBiases)), t.WithBacking(newRawBiases))
}
