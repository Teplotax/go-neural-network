package main

import (
	"fmt"
	"time"

	timing "github.com/Teplotax/go-neural-network/Timing"
	ml "github.com/Teplotax/go-neural-network/ml"
)

func main() {
	start := time.Now()
	defer timing.MicrosecondsSince(start)

	// rawInputs := []float32{
	// 	1.0, 2.0, 3.0, 2.5,
	// 	2.0, 5.0, -1, 2.0,
	// 	-1.5, 2.7, 3.3, -0.8,
	// }

	// // // Inputs
	// inputs := t.New(t.WithShape(3, 4), t.WithBacking(rawInputs))

	// // Layer 1
	// rawWeights1 := []float32{
	// 	0.2, 0.8, -0.5, 1.0,
	// 	0.5, -0.91, 0.26, -0.5,
	// 	-0.26, -0.27, 0.17, 0.87,
	// }
	// biases1 := []float32{2.0, 3.0, 0.5}

	// weights1 := t.New(t.WithShape(3, 4), t.WithBacking(rawWeights1))
	// weights1.T()

	// // Forward pass 1
	// dP1, _ := t.Dot(inputs, weights1)
	// output1, _ := ml.LayerOutput(dP1, biases1)

	// fmt.Println(output1)

	// // Layer 2
	// rawWeights2 := []float32{
	// 	0.1, -0.14, 0.5,
	// 	-0.5, 0.12, -0.33,
	// 	-0.44, 0.73, -0.13,
	// }
	// biases2 := []float32{-1, 2, -0.5}

	// weights2 := t.New(t.WithShape(3, 3), t.WithBacking(rawWeights2))
	// weights2.T()

	// // Forward pass 1
	// dP2, _ := t.Dot(output1, weights2)
	// output2, _ := ml.LayerOutput(dP2, biases2)

	// fmt.Println(output2)

	// p.PlotData(X, Y, CMap, "plot.png")

	// make the layer

	dense1 := ml.NewLayerDense(2, 3)
	dense2 := ml.NewLayerDense(3, 2)

	// forward pass
	dense1.Forward(X)
	dense2.Forward(dense1.Outputs)

	fmt.Println(dense2.Outputs)

}
