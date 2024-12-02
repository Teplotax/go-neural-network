package ml

import (
	t "gorgonia.org/tensor"
)

type ActivationFunction interface {
	Forward(inputs t.Tensor)
	GetOutputs() t.Tensor
}

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

func (r *ActivationReLU) GetOutputs() t.Tensor {
	return r.Outputs
}

type ActivationSoftMax struct {
	Outputs t.Tensor
}

func NewActivationSoftMax() ActivationSoftMax {
	return ActivationSoftMax{Outputs: t.New(t.Of(t.Float64))}
}

func (s *ActivationSoftMax) Forward(inputs t.Tensor) {

	outputs, err := t.SoftMax(inputs, 1)
	handleErr(err)

	s.Outputs = outputs
}

func (s *ActivationSoftMax) GetOutputs() t.Tensor {
	return s.Outputs
}
