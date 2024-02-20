package ml

import (
	t "gorgonia.org/tensor"
)

type ActivationReLU struct {
	Outputs t.Tensor
}

func NewActivationReLU() ActivationReLU {
	return ActivationReLU{Outputs: t.New(t.Of(t.Float64))}
}

func (r *ActivationReLU) Forward(inputs t.Tensor) {
	zeros := t.New(t.WithShape(inputs.Shape()...), t.Of(t.Float64))

	outputs, err := t.MaxBetween(inputs, zeros)
	handleErr(err)

	r.Outputs = outputs
}
