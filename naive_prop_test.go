package knn

import (
	"fmt"
	"reflect"
	"sort"
	"testing"

	"github.com/leanovate/gopter"
	"github.com/leanovate/gopter/gen"
	"github.com/leanovate/gopter/prop"
	"github.com/stretchr/testify/require"
)

func TestNaiveKNNProperty(t *testing.T) {
	parameters := gopter.DefaultTestParameters()
	parameters.MinSuccessfulTests = 1000

	properties := gopter.NewProperties(parameters)
	genVectors := gen.SliceOfN(100, gen.Float64Range(-1e6, 1e6)).Map(func(v []float64) Vector {
		return Vector{Point: v}
	})

	genK := gen.IntRange(1, 10)

	genVectorsSlice := gen.SliceOf(genVectors).SuchThat(func(v []Vector) bool {
		return len(v) > 0
	})

	properties.Property("NaiveKNN.Search should return k vectors", prop.ForAll(
		func(vecs []Vector, k int) bool {
			if len(vecs) < k {
				return true // Not enough vectors to test this property
			}
			naiveKNN, _ := NewNaiveKNN(vecs, CosineDistance)
			return len(naiveKNN.Search(k, vecs[0])) == k
		},
		genVectorsSlice,
		genK,
	))

	properties.Property("NaiveKNN.Search should return the same vectors for the same input", prop.ForAll(
		func(vecs []Vector, k int) bool {
			if len(vecs) < k {
				return true // Not enough vectors to test this property
			}
			naiveKNN, _ := NewNaiveKNN(vecs, CosineDistance)
			return reflect.DeepEqual(naiveKNN.Search(k, vecs[0]), naiveKNN.Search(k, vecs[0]))
		},
		genVectorsSlice,
		genK,
	))

	properties.Property("NaiveKNN.Search should only return vectors from the initial set", prop.ForAll(
		func(vecs []Vector, k int) bool {
			if len(vecs) < k {
				return true // Not enough vectors to test this property
			}
			naiveKNN, _ := NewNaiveKNN(vecs, CosineDistance)
			kNearest := naiveKNN.Search(k, vecs[0])

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

	properties.Property("NaiveKNN.Search should return k closest vectors to the target", prop.ForAll(
		func(vecs []Vector, k int, target Vector) (bool, error) {
			if len(vecs) < k {
				return true, nil // Not enough vectors to test this property
			}
			naiveKNN, err := NewNaiveKNN(vecs, CosineDistance)
			if err != nil {
				return false, err
			}
			kNearest := naiveKNN.Search(k, target)

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
					return false, fmt.Errorf("did not find expected distance %f in shortest %d distances", kd, k)
				}
			}

			return true, nil
		},
		genVectorsSlice,
		genK,
		genVectors,
	))

	properties.Property("NaiveKNN.KNN should return values ordered closest to furthest", prop.ForAll(
		func(vecs []Vector, k int, target Vector) (bool, error) {
			naiveKNN, err := NewNaiveKNN(vecs, CosineDistance)
			if err != nil {
				return false, err
			}
			results := naiveKNN.Search(k, target)

			// Calculate the distance from target to each result vector
			distances := make([]float64, len(results))
			for i, result := range results {
				distances[i] = CosineDistance(target, result)
			}

			// Check if the distances are in ascending order
			for i := 0; i < len(results)-1; i++ {
				if distances[i] > distances[i+1]+_tolerance {
					return false, fmt.Errorf("distances not in ascending order")
				}
			}

			return true, nil
		},
		genVectorsSlice,
		genK,
		genVectors,
	))

	properties.TestingRun(t)
}

func TestNaiveKNNFailedOrderCheck(t *testing.T) {
	vecs := []Vector{
		{ID: "1", Point: []float64{-336385.56231209706, -209830.96853725868}},
		{ID: "2", Point: []float64{-334181.08546641236, 430176.9241516462}},
	}

	targetVector := Vector{
		ID: "3", Point: []float64{79021.84705465887, 748382.334697546},
	}

	knn, err := NewNaiveKNN(vecs, CosineDistance)
	require.NoError(t, err)

	result := knn.Search(2, targetVector)

	require.Len(t, result, 2)
	require.Contains(t, result, vecs[0])
	require.Contains(t, result, vecs[1])
	require.Equal(t, "2", string(result[0].ID))
	require.Equal(t, "1", string(result[1].ID))
}
