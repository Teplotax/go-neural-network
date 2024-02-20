package main

import (
	"fmt"
	"time"

	timing "github.com/Teplotax/go-neural-network/Timing"
	"github.com/Teplotax/go-neural-network/ml"
)

func main() {
	start := time.Now()
	defer timing.MicrosecondsSince(start)

	// make the layer

	dense1 := ml.NewLayerDense(2, 3)
	activation1 := ml.NewActivationReLU()

	// // forward pass
	dense1.Forward(X)
	activation1.Forward(dense1.Outputs)

	fmt.Println(dense1.Outputs)
	fmt.Println(activation1.Outputs)

	// p.PlotData(X, Y, CMap, "plot.png")

}
