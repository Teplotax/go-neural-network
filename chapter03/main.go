package main

import (
	"fmt"
	"log"
	"time"

	ml "github.com/Teplotax/go-neural-network/ml"

	timing "github.com/Teplotax/go-neural-network/Timing"
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

	dense2.Forward(dense1.Outputs)
	activation2.Forward(dense2.Outputs, 1)

	loss, avgLoss := ml.CategoricalCrossEntropyLoss(activation2.Outputs, rawY)

	fmt.Println(activation2.Outputs)
	fmt.Println(loss)
	fmt.Println(avgLoss)

	//---------------------------------------

	// data plotting
	// p.PlotData(X, Y, CMap, "plot.png")
	//---------------------------------------

	// SoftMax activation function
	// outputs := [][]float64{
	// 	{4.8, 1.21, 2.385},
	// 	{8.9, -1.81, 0.2},
	// 	{1.41, 1.051, 0.026},
	// }

	// expVals := make([][]float64, len(outputs))
	// normVals := make([][]float64, len(expVals))

	// for i, output := range outputs {

	// 	var normBase float64 = 0
	// 	var max float64 = output[0]
	// 	expVals[i] = make([]float64, len(output))
	// 	normVals[i] = make([]float64, len(output))

	// 	for _, val := range output {
	// 		max = math.Max(max, val)
	// 	}

	// 	for j, val := range output {
	// 		expVals[i][j] = math.Pow(math.E, val-max)
	// 		max = math.Max(max, expVals[i][j])
	// 		normBase += expVals[i][j]
	// 	}

	// 	for l, val := range expVals[i] {
	// 		normVals[i][l] = val / normBase
	// 	}
	// }

	// fmt.Println(normVals)
	//---------------------------------------

	// Calculating Network Error with Loss

	// single output

	// softmaxOutputs := []float64{0.7, 0.1, 0.2}

	// desired := []float64{1, 0, 0}

	// var sum float64
	// for i, val := range softmaxOutputs {
	// 	sum += desired[i] * math.Log(val)
	// }

	// loss := -sum

	// fmt.Println(loss)

	// batch outputs onHand encoded
	// softmaxOutputs := [][]float64{
	// 	{0.7, 0.1, 0.2},
	// 	{0.1, 0.5, 0.4},
	// 	{0.02, 0.9, 0.08},
	// }

	// desired := [][]float64{
	// 	{1, 0, 0},
	// 	{0, 1, 0},
	// 	{0, 1, 0},
	// }

	// loss := make([]float64, len(softmaxOutputs))

	// for i, output := range softmaxOutputs {
	// 	for j, val := range output {
	// 		loss[i] -= math.Log(val) * desired[i][j]
	// 	}
	// }

	// fmt.Println(loss)
	//---------------------------------------

	// batch outputs not one-hot encoded

	// softmaxOutputs := [][]float64{
	// 	{0.7, 0.1, 0.2},
	// 	{0.1, 0.5, 0.4},
	// 	{0.02, 0.9, 0.08},
	// }

	// targetOutput := []int{0, 1, 1}

	// loss := make([]float64, len(targetOutput))

	// for i, output := range softmaxOutputs {

	// 	val := output[targetOutput[i]]
	// 	fmt.Println(val)
	// 	loss[i] -= math.Log(val)
	// }

	// fmt.Println(loss)
	// fmt.Println(stat.Mean(loss, nil))

	//---------------------------------------

	// batch outputs not one-hot encoded with tensors

	// softmaxOutputsBacking := []float64{
	// 	0.7, 0.1, 0.2,
	// 	0.1, 0.5, 0.4,
	// 	0.02, 0.9, 0.08,
	// }
	// softmaxOutputs := t.New(t.WithShape(3, 3), t.WithBacking(softmaxOutputsBacking))

	// predictedValsBacking := make([]float64, softmaxOutputs.Shape()[0])

	// targetOutput := []float64{0, 1, 1}

	// it := softmaxOutputs.Iterator()

	// for _, err := it.Start(); err == nil; _, err = it.Next() {

	// 	c := it.Coord()

	// 	if c[1] == int(targetOutput[c[0]]) {
	// 		tempVal, _ := softmaxOutputs.At(it.Coord()...)
	// 		val, _ := tempVal.(float64)

	// 		predictedValsBacking[c[0]] = val
	// 	}
	// }

	// predictedVals := t.New(t.WithBacking(predictedValsBacking))

	// losses, err := t.Log(predictedVals)
	// handleErr(err)
	// losses, err = t.Mul(losses, float64(-1))
	// handleErr(err)

	// sum, err := t.Sum(losses)
	// handleErr(err)

	// avgLoss, _ := t.Div(sum, float64(losses.Size()))

	// fmt.Println(softmaxOutputs)
	// fmt.Println(loss)
	// fmt.Println(avgLoss)

	// testing one-hot to sparce conversion

	// oneHot := [][]float64{
	// 	{0, 1, 0},
	// 	{1, 0, 0},
	// 	{1, 0, 0},
	// 	{0, 0, 1},
	// }

	// sparse := ml.Sparse(oneHot)

	// fmt.Println(sparse)

}

func handleErr(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
