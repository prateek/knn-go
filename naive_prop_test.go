package knn

import (
	"reflect"
	"sort"
	"testing"

	"github.com/leanovate/gopter"
	"github.com/leanovate/gopter/gen"
	"github.com/leanovate/gopter/prop"
)

func TestNaiveKNNProperty(t *testing.T) {
	parameters := gopter.DefaultTestParameters()
	parameters.MinSuccessfulTests = 1000

	properties := gopter.NewProperties(parameters)
	genVectors := gen.SliceOfN(100, gen.Float64Range(-1e6, 1e6)).Map(func(v []float64) Vector {
		return Vector{Point: v}
	})

	genK := gen.IntRange(1, 10)

	genVectorsSlice := gen.SliceOf(genVectors)

	properties.Property("NaiveKNN.KNN should return k vectors", prop.ForAll(
		func(vecs []Vector, k int) bool {
			if len(vecs) < k {
				return true // Not enough vectors to test this property
			}
			naiveKNN, _ := NewNaiveKNN(vecs, CosineDistance)
			return len(naiveKNN.KNN(k, vecs[0])) == k
		},
		genVectorsSlice,
		genK,
	))

	properties.Property("NaiveKNN.KNN should return the same vectors for the same input", prop.ForAll(
		func(vecs []Vector, k int) bool {
			if len(vecs) < k {
				return true // Not enough vectors to test this property
			}
			naiveKNN, _ := NewNaiveKNN(vecs, CosineDistance)
			return reflect.DeepEqual(naiveKNN.KNN(k, vecs[0]), naiveKNN.KNN(k, vecs[0]))
		},
		genVectorsSlice,
		genK,
	))

	properties.Property("NaiveKNN.KNN should only return vectors from the initial set", prop.ForAll(
		func(vecs []Vector, k int) bool {
			if len(vecs) < k {
				return true // Not enough vectors to test this property
			}
			naiveKNN, _ := NewNaiveKNN(vecs, CosineDistance)
			kNearest := naiveKNN.KNN(k, vecs[0])

			for _, v := range kNearest {
				found := false
				for _, initialVec := range vecs {
					if reflect.DeepEqual(v, initialVec) {
						found = true
						break
					}
				}
				if !found {
					return false
				}
			}
			return true
		},
		genVectorsSlice,
		genK,
	))

	properties.Property("NaiveKNN.KNN should return k closest vectors to the target", prop.ForAll(
		func(vecs []Vector, k int, target Vector) bool {
			if len(vecs) < k {
				return true // Not enough vectors to test this property
			}
			naiveKNN, _ := NewNaiveKNN(vecs, CosineDistance)
			kNearest := naiveKNN.KNN(k, target)

			// Compute all distances and sort them
			allDistances := make([]float64, len(vecs))
			for i, v := range vecs {
				allDistances[i] = CosineDistance(target, v)
			}
			sort.Float64s(allDistances)

			// Compute distances to kNearest vectors
			kDistances := make([]float64, k)
			for i, v := range kNearest {
				kDistances[i] = CosineDistance(target, v)
			}
			sort.Float64s(kDistances)

			// Verify that each kDistance is in the shortest k distances
			for _, kd := range kDistances {
				found := false
				for i := 0; i < k; i++ {
					if kd >= allDistances[i]-_tolerance && kd <= allDistances[i]+_tolerance {
						found = true
						break
					}
				}
				if !found {
					return false
				}
			}

			return true
		},
		genVectorsSlice,
		genK,
		genVectors,
	))

	properties.TestingRun(t)
}
