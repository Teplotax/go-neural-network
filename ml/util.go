package ml

func Sparse(targets [][]float64) []float64 {
	sparse := make([]float64, len(targets))

	for i, oneHot := range targets {
		var maxIndex float64 = 0
		max := oneHot[0]
		for i := 1; i < len(oneHot); i++ {
			if oneHot[i] > max {
				maxIndex = float64(i)
				max = oneHot[i]
			}
		}
		sparse[i] = maxIndex
	}

	return sparse
}
