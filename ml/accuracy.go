package ml

import (
	"fmt"
	t "gorgonia.org/tensor"
)

func Accuracy(outputs t.Tensor, targetOutput []float64) float64 {

	predictions, _ := t.Argmax(outputs, 1)

	var hit float64 = 0

	for i := 0; i < predictions.Shape()[0]; i++ {
		// Get the value at index i
		val, _ := predictions.At(i)
		if val == int(targetOutput[i]) {
			hit++
		}

		// Print the value as float64

	}

	fmt.Printf("Hits: %v\n", hit)
	fmt.Printf("Total: %v\n", len(targetOutput))

	return hit / float64(len(targetOutput))
}
