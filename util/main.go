package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// BoxMuller generates two independent standard normal distributed numbers (z0 and z1)
// based on two uniformly distributed random variables u1 and u2.
func BoxMuller() (float64, float64) {
	// Generate two independent uniform random numbers u1 and u2
	u1 := rand.Float64()
	u2 := rand.Float64()

	// Box-Muller transform to get two normally distributed random variables
	z0 := math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
	z1 := math.Sqrt(-2*math.Log(u1)) * math.Sin(2*math.Pi*u2)

	// Return the two independent standard normal variables
	return z0, z1
}

func main() {
	fmt.Println(NormalDistribution(3, 2, 0.005))
}

func NormalDistribution(i, j int, base float64) []float64 {
	// Set the random seed
	rand.Seed(time.Now().UnixNano())

	length := i * j

	randNumbers := make([]float64, length)

	// Generate and print 10 random numbers following the standard normal distribution
	// We will print 2 numbers (z0, z1) per iteration
	counter := 0
	for i := 0; i < length/2; i++ { // 5 iterations, because each iteration gives 2 numbers
		z0, z1 := BoxMuller()

		randNumbers[counter] = base * z0
		counter++

		if counter <= length {
			randNumbers[counter] = base * z1
			counter++
		}
	}
	return randNumbers
}
