package knn

import (
	"testing"

	"github.com/leanovate/gopter"
	"github.com/leanovate/gopter/gen"
	"github.com/leanovate/gopter/prop"
)

func TestCosineDistanceProperty(t *testing.T) {
	parameters := gopter.DefaultTestParameters()
	parameters.MinSuccessfulTests = 10_000

	properties := gopter.NewProperties(parameters)
	genVectors := gen.SliceOfN(100, gen.Float64Range(-1e6, 1e6)).Map(func(v []float64) Vector {
		return Vector{Point: v}
	})

	properties.Property("CosineDistance of identical vectors should be zero", prop.ForAll(
		func(a Vector) bool {
			distance := CosineDistance(a, a)
			return distance >= -_tolerance && distance <= _tolerance
		},
		genVectors,
	))

	const tolerance = 1e-9

	properties.Property("CosineDistance should be non-negative", prop.ForAll(
		func(a, b Vector) bool {
			return CosineDistance(a, b) >= 0
		},
		genVectors,
		genVectors,
	))

	properties.Property("CosineDistance should be symmetric", prop.ForAll(
		func(a, b Vector) bool {
			distanceAB := CosineDistance(a, b)
			distanceBA := CosineDistance(b, a)
			return distanceAB >= distanceBA-tolerance && distanceAB <= distanceBA+tolerance
		},
		genVectors,
		genVectors,
	))

	properties.TestingRun(t)
}
