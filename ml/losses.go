package ml

import (
	t "gorgonia.org/tensor"
)

func CategoricalCrossEntropyLoss(predictions t.Tensor, targetOutput []float64) (t.Tensor, float64) {

	predictedValsBacking := make([]float64, predictions.Shape()[0])

	it := predictions.Iterator()

	for _, err := it.Start(); err == nil; _, err = it.Next() {

		c := it.Coord()

		if c[1] == int(targetOutput[c[0]]) {
			tempVal, _ := predictions.At(it.Coord()...)
			val, _ := tempVal.(float64)

			predictedValsBacking[c[0]] = val
		}
	}

	predictedVals := t.New(t.WithBacking(predictedValsBacking))
	lower := float32(1e-7)
	upper := float32(1 - 1e-7)
	clippedPredictedVals, _ := t.Clamp(predictedVals, lower, upper)

	losses, err := t.Log(clippedPredictedVals)
	handleErr(err)
	losses, err = t.Mul(losses, float64(-1))
	handleErr(err)

	sum, err := t.Sum(losses)
	handleErr(err)

	averageLoss, _ := t.Div(sum, float64(losses.Size()))

	return losses, averageLoss.ScalarValue().(float64)
}
