package ml

import (
	"log"

	t "gorgonia.org/tensor"
)

type LayerDense struct {
	Weights t.Tensor
	Biases  []float64
	Outputs t.Tensor
}

func NewLayerDense(nInputs, nNeurons int) LayerDense {

	weights := t.New(
		t.WithShape(nInputs, nNeurons),
		t.WithBacking(t.Random(t.Float64, nInputs*nNeurons)),
	)
	weights, err := weights.MulScalar(0.01, false)
	handleErr(err)

	biases := make([]float64, nNeurons)

	return LayerDense{Weights: weights, Biases: biases}
}

func (layer *LayerDense) Forward(inputs t.Tensor) {
	dP, err := t.Dot(inputs, layer.Weights)
	handleErr(err)

	layer.Outputs, err = LayerOutput(dP, layer.Biases)
	handleErr(err)
}

func handleErr(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
