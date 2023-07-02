package knn

import (
	"fmt"
	"math"
	"reflect"
	"sort"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"pgregory.net/rapid"
)

var genVector = rapid.Custom(func(t *rapid.T) Vector {
	return Vector{Point: rapid.SliceOfN(rapid.Float64Range(-1e6, 1e6), 100, 100).Draw(t, "point")}
})

var genVectors = rapid.SliceOf(genVector).Filter(func(vecs []Vector) bool {
	return len(vecs) > 0
})

func TestNaiveKNNReturnsKVectors(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		target := genVector.Draw(t, "target")
		vecs := genVectors.Draw(t, "vecs")
		k := rapid.IntRange(1, 10).Draw(t, "k")

		if len(vecs) < k {
			t.Skip("Not enough vectors to test this property")
		}

		naiveKNN, err := NewNaiveKNN(vecs, CosineDistance)
		assert.NoError(t, err)
		assert.Equal(t, k, len(naiveKNN.Search(k, target)), "NaiveKNN.Search should return k vectors")
	})
}

func TestNaiveKNNReturnsSameVectors(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		target := genVector.Draw(t, "target")
		vecs := genVectors.Draw(t, "vecs")
		k := rapid.IntRange(1, 10).Draw(t, "k")

		if len(vecs) < k {
			t.Skip("Not enough vectors to test this property")
		}

		naiveKNN, err := NewNaiveKNN(vecs, CosineDistance)
		assert.NoError(t, err)

		assert.Equal(t,
			naiveKNN.Search(k, target),
			naiveKNN.Search(k, target),
			"NaiveKNN.Search should return the same vectors for the same input",
		)
	})
}

func TestNaiveKNNReturnsInitialSetVectors(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		target := genVector.Draw(t, "target")
		vecs := genVectors.Draw(t, "vecs")
		k := rapid.IntRange(1, 10).Draw(t, "k")

		if len(vecs) < k {
			t.Skip("Not enough vectors to test this property")
		}

		naiveKNN, err := NewNaiveKNN(vecs, CosineDistance)
		assert.NoError(t, err)
		kNearest := naiveKNN.Search(k, target)

		for _, v := range kNearest {
			found := false
			for _, initialVec := range vecs {
				if reflect.DeepEqual(v, initialVec) {
					found = true
					break
				}
			}
			assert.True(t, found, "NaiveKNN.Search should only return vectors from the initial set")
		}
	})
}

func TestNaiveKNNReturnsClosestVectors(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		target := genVector.Draw(t, "target")
		vecs := genVectors.Draw(t, "vecs")
		k := rapid.IntRange(1, 10).Draw(t, "k")

		if len(vecs) < k {
			t.Skip("Not enough vectors to test this property")
		}

		naiveKNN, err := NewNaiveKNN(vecs, CosineDistance)
		require.NoError(t, err)
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
				if math.Abs(kd-allDistances[i]) <= _tolerance {
					found = true
					break
				}
			}
			assert.True(t, found, fmt.Sprintf("did not find expected distance %f in shortest %d distances", kd, k))
		}
	})
}

func TestNaiveKNNReturnsOrderedVectors(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		target := genVector.Draw(t, "target")
		vecs := genVectors.Draw(t, "vecs")
		k := rapid.IntRange(1, 10).Draw(t, "k")

		if len(vecs) < k {
			t.Skip("Not enough vectors to test this property")
		}

		naiveKNN, err := NewNaiveKNN(vecs, CosineDistance)
		require.NoError(t, err)
		results := naiveKNN.Search(k, target)

		// Calculate the distance from target to each result vector
		distances := make([]float64, len(results))
		for i, result := range results {
			distances[i] = CosineDistance(target, result)
		}

		// Check if the distances are in ascending order
		for i := 0; i < len(results)-1; i++ {
			assert.True(t, distances[i] <= distances[i+1]+_tolerance, "distances not in ascending order")
		}
	})
}
