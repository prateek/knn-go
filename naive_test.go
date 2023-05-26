package knn

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestNewNaiveKNN(t *testing.T) {
	vecs := []Vector{
		{Point: []float64{1, 0}},
		{Point: []float64{0, 1}},
	}

	knn, err := NewNaiveKNN(vecs, CosineDistance)
	require.NoError(t, err)
	require.NotNil(t, knn)

	_, err = NewNaiveKNN([]Vector{}, CosineDistance)
	require.Error(t, err)

	vecs = append(vecs, Vector{Point: []float64{0, 1, 2}})
	_, err = NewNaiveKNN(vecs, CosineDistance)
	require.Error(t, err)
}

func TestKNN(t *testing.T) {
	vecs := []Vector{
		{ID: "1", Point: []float64{1, 0}},
		{ID: "2", Point: []float64{0, 1}},
		{ID: "3", Point: []float64{-1, 0}},
		{ID: "4", Point: []float64{0, -1}},
	}

	knn, err := NewNaiveKNN(vecs, CosineDistance)
	require.NoError(t, err)

	target := Vector{Point: []float64{1, 1}}
	result := knn.KNN(2, target)

	require.Len(t, result, 2)
	require.Contains(t, result, vecs[0])
	require.Contains(t, result, vecs[1])
}
