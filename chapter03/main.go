package main

import (
	"fmt"
	"math"
	"time"

	timing "github.com/Teplotax/go-neural-network/Timing"
)

func main() {
	start := time.Now()
	defer timing.MicrosecondsSince(start)

	// make the layer

	// dense1 := ml.NewLayerDense(2, 3)
	// activation1 := ml.NewActivationReLU()

	// forward pass
	// dense1.Forward(X)
	// activation1.Forward(dense1.Outputs)

	// fmt.Println(activation1.Outputs)

	// p.PlotData(X, Y, CMap, "plot.png")

	//---------------------------------------

	outputs := [][]float64{
		{4.8, 1.21, 2.385},
		{8.9, -1.81, 0.2},
		{1.41, 1.051, 0.026},
	}

	expVals := make([][]float64, len(outputs))
	normVals := make([][]float64, len(expVals))

	for i, output := range outputs {

		var normBase float64 = 0
		var max float64 = output[0]
		expVals[i] = make([]float64, len(output))
		normVals[i] = make([]float64, len(output))

		for _, val := range output {
			max = math.Max(max, val)
		}

		for j, val := range output {
			expVals[i][j] = math.Pow(math.E, val-max)
			max = math.Max(max, expVals[i][j])
			normBase += expVals[i][j]
		}

		for l, val := range expVals[i] {
			normVals[i][l] = val / normBase
		}
	}

	fmt.Println(normVals)

}
