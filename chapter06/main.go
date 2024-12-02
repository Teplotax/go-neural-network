package main

import (
	"fmt"
	timing "github.com/Teplotax/go-neural-network/Timing"
	ml "github.com/Teplotax/go-neural-network/ml"
	"gorgonia.org/tensor"
	"math"
	"math/rand"
	"time"
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
	layers := []ml.LayerDense{dense1, dense2}
	activationFunctions := []ml.ActivationFunction{&activation1, &activation2}

	outputs := forward(layers, activationFunctions)

	//dense1.Forward(X)
	//activation1.Forward(dense1.Outputs)
	//
	//dense2.Forward(activation1.Outputs)
	//activation2.Forward(dense2.Outputs)
	//
	//outputs := activation2.Outputs

	loss, avgLoss := ml.CategoricalCrossEntropyLoss(outputs, rawY)

	fmt.Printf("Outputs: %v\n", outputs)
	fmt.Printf("Loss: %v\n", loss)
	fmt.Printf("Average Loss: %v\n", avgLoss)

	// Accuracy Calculation
	accuracy := ml.Accuracy(outputs, rawY)

	fmt.Printf("Accuracy: %v\n", accuracy)

	// Generate a random float64 number between

	//Fix it, each value must sum a different Box-Muller rand number

	// Efficiency Consideration:
	//While the Box-Muller method is simple and effective,
	//it involves a logarithm and trigonometric operations,
	//which can be computationally expensive. In some cases,
	//more efficient algorithms, like the Ziggurat algorithm,
	//are preferred for generating normal random variables.

	//newLayers := make([]ml.LayerDense, len(layers))
	//copy(newLayers[:], layers[:])

	// run again
	for i := 0; i < 1000000; i++ {
		newLayers := ml.AdjustLayers(layers, 0.05)

		newOutputs := forward(newLayers, activationFunctions)
		_, newAvgLoss := ml.CategoricalCrossEntropyLoss(newOutputs, rawY)

		if newAvgLoss < avgLoss {
			newAccuracy := ml.Accuracy(newOutputs, rawY)
			fmt.Printf("Iteration: %v, Loss: %v, Accuracy: %v\n", i, newAvgLoss, newAccuracy)
			avgLoss = newAvgLoss
			layers = newLayers
		}
	}
}

func forward(layers []ml.LayerDense, activationFunctions []ml.ActivationFunction) tensor.Tensor {
	inputs := X
	var outputs tensor.Tensor

	for i := 0; i < len(layers); i++ {
		layers[i].Forward(inputs)
		activationFunctions[i].Forward(layers[i].Outputs)
		inputs = activationFunctions[i].GetOutputs()
	}
	outputs = inputs

	return outputs
}

func randn() float64 {
	// Generate two random values between 0 and 1
	u1 := rand.Float64()
	u2 := rand.Float64()

	// Box-Muller transform
	z0 := math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)

	return z0
}
