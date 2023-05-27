package knn

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCosineDistance(t *testing.T) {
	a := Vector{Point: []float64{1, 0}}
	b := Vector{Point: []float64{0, 1}}
	expected := 1.0

	assert.Equal(t, expected, CosineDistance(a, b))
}

func TestCosineDistanceEmptyPoint(t *testing.T) {
	a := Vector{Point: []float64{0, 0}}
	b := Vector{Point: []float64{0, 0}}

	d := CosineDistance(a, b)
	assert.True(t, math.IsNaN(d), d)
}
