package MachineLearning

import (
	"fmt"
	"time"

	t "gorgonia.org/tensor"
)

func LayerOutput(dotProduct t.Tensor, rawBiases []float32) (t.Tensor, error) {

	newShape := dotProduct.Shape()
	newRawBiases := []float32{}

	for i := 0; i < newShape[0]; i++ {
		newRawBiases = append(newRawBiases, rawBiases...)
	}

	biases := t.New(t.WithShape(newShape...), t.WithBacking(newRawBiases))

	return t.Add(dotProduct, biases)
}

func MicrosecondsSince(start time.Time) {
	duration := time.Since(start)
	fmt.Println("Execution time: ", duration.Microseconds(), "μs")
}
