package ml

import (
	t "gorgonia.org/tensor"
	"math"
	"math/rand"
	"time"
)

func AdjustLayers(layers []LayerDense, base float64) []LayerDense {
	newLayers := make([]LayerDense, len(layers))
	copy(newLayers[:], layers[:])

	for i, _ := range newLayers {
		newLayers[i].Weights = AdjustWeights(layers[i].Weights, base)
		newLayers[i].Biases = AdjustBiases(layers[i].Biases, base)
	}

	return newLayers
}

func AdjustWeights(weights t.Tensor, base float64) t.Tensor {
	randDistribution := t.New(
		t.WithShape(weights.Shape()...),
		t.WithBacking(normalDistribution(weights.Size(), base)),
	)
	newWeights, _ := t.Add(weights, randDistribution)
	return newWeights
}

func AdjustBiases(biases []float64, base float64) []float64 {

	nd := normalDistribution(len(biases), base)

	newBiases := make([]float64, len(biases))
	for i := range biases {
		newBiases[i] = biases[i] + nd[i]
	}

	return newBiases
}

func normalDistribution(size int, base float64) []float64 {
	// Set the random seed
	rand.Seed(time.Now().UnixNano())

	randNumbers := make([]float64, size)

	// Generate and print 10 random numbers following the standard normal distribution
	// We will print 2 numbers (z0, z1) per iteration
	counter := 0
	for i := 0; i < size/2; i++ { // 5 iterations, because each iteration gives 2 numbers
		z0, z1 := boxMuller()

		randNumbers[counter] = base * z0
		counter++

		if counter <= size {
			randNumbers[counter] = base * z1
			counter++
		}
	}
	return randNumbers
}

func boxMuller() (float64, float64) {
	// Generate two independent uniform random numbers u1 and u2
	u1 := rand.Float64()
	u2 := rand.Float64()

	// Box-Muller transform to get two normally distributed random variables
	z0 := math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
	z1 := math.Sqrt(-2*math.Log(u1)) * math.Sin(2*math.Pi*u2)

	// Return the two independent standard normal variables
	return z0, z1
}
