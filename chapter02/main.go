package main

import (
	"fmt"
	"time"

	t "gorgonia.org/tensor"
)

func main() {
	start := time.Now()
	defer MicrosecondsSince(start)

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
	weights.T()

	dP, _ := t.Dot(inputs, weights)

	outputs, err := LayerOutput(dP, rawBiases)
	if err != nil {
		fmt.Println(err)
	}

	fmt.Println(outputs)
}
