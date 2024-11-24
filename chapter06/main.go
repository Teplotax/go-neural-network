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

	outputs := activation2.Outputs

	loss, avgLoss := ml.CategoricalCrossEntropyLoss(outputs, rawY)

	fmt.Printf("Outputs: %v\n", outputs)
	fmt.Printf("Loss: %v\n", loss)
	fmt.Printf("Average Loss: %v\n", avgLoss)

	// Accuracy Calculation
	accuracy := ml.Accuracy(outputs, rawY)

	fmt.Printf("Accuracy: %v\n", accuracy)
}
