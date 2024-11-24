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

	// NEURAL NETWORK
	// make first layer
	dense1 := ml.NewLayerDense(2, 3)
	activation1 := ml.NewActivationReLU()

	// make second layer
	dense2 := ml.NewLayerDense(3, 3)
	activation2 := ml.NewActivationSoftMax()

	// forward pass
	dense1.Forward(X)
	activation1.Forward(dense1.Outputs)

	dense2.Forward(activation1.Outputs)
	activation2.Forward(dense2.Outputs, 1)

	loss, avgLoss := ml.CategoricalCrossEntropyLoss(activation2.Outputs, rawY)

	fmt.Println(activation2.Outputs)
	fmt.Println(loss)
	fmt.Println(avgLoss)

	// Accuracy Calculation

	outputs := activation2.Outputs

	accuracy := ml.Accuracy(outputs, rawY)

	fmt.Println(accuracy)
}
