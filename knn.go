package knn

import (
	"math"
)

type ID string

type Payload []byte

type DistanceFn func(a, b Vector) float64

type Vector struct {
	ID      ID
	Payload Payload
	Point   []float64
}

func CosineDistance(a, b Vector) float64 {
	var dotProduct, normA, normB float64
	for i := range a.Point {
		dotProduct += a.Point[i] * b.Point[i]
		normA += math.Pow(a.Point[i], 2)
		normB += math.Pow(b.Point[i], 2)
	}
	return 1 - dotProduct/(math.Sqrt(normA)*math.Sqrt(normB))
}

type KNN interface {
	Search(k int, targetVector Vector) []Vector
}
