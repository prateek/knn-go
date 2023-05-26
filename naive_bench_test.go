package knn

import (
	"fmt"
	"math/rand"
	"testing"
)

/*
- Results on my M2 Max MBP as of 2023-05-26:
Λ go test -bench=. -run '^$' .
goos: darwin
goarch: arm64
pkg: github.com/prateek/knn-go
BenchmarkNaiveKNN/dimension-1000_k-100_numVec-2000-12                 13          78137103 ns/op
BenchmarkNaiveKNN/dimension-1000_k-100_numVec-3000-12                  9         116915051 ns/op
BenchmarkNaiveKNN/dimension-1000_k-100_numVec-4000-12                  7         154007631 ns/op
BenchmarkNaiveKNN/dimension-1000_k-100_numVec-5000-12                  6         193157847 ns/op
BenchmarkNaiveKNN/dimension-1000_k-200_numVec-2000-12                 14          77251137 ns/op
BenchmarkNaiveKNN/dimension-1000_k-200_numVec-3000-12                  9         117093866 ns/op
BenchmarkNaiveKNN/dimension-1000_k-200_numVec-4000-12                  7         155256643 ns/op
BenchmarkNaiveKNN/dimension-1000_k-200_numVec-5000-12                  6         192989180 ns/op
BenchmarkNaiveKNN/dimension-1000_k-500_numVec-2000-12                 14          77250690 ns/op
BenchmarkNaiveKNN/dimension-1000_k-500_numVec-3000-12                  9         118173732 ns/op
BenchmarkNaiveKNN/dimension-1000_k-500_numVec-4000-12                  7         154997946 ns/op
BenchmarkNaiveKNN/dimension-1000_k-500_numVec-5000-12                  6         193248389 ns/op
BenchmarkNaiveKNN/dimension-1000_k-1000_numVec-2000-12                14          78970607 ns/op
BenchmarkNaiveKNN/dimension-1000_k-1000_numVec-3000-12                 9         117289245 ns/op
BenchmarkNaiveKNN/dimension-1000_k-1000_numVec-4000-12                 7         155286923 ns/op
BenchmarkNaiveKNN/dimension-1000_k-1000_numVec-5000-12                 6         198260312 ns/op
BenchmarkNaiveKNN/dimension-1500_k-100_numVec-2000-12                  9         118312208 ns/op
BenchmarkNaiveKNN/dimension-1500_k-100_numVec-3000-12                  6         176361944 ns/op
BenchmarkNaiveKNN/dimension-1500_k-100_numVec-4000-12                  5         234593858 ns/op
BenchmarkNaiveKNN/dimension-1500_k-100_numVec-5000-12                  4         293964562 ns/op
BenchmarkNaiveKNN/dimension-1500_k-200_numVec-2000-12                  9         115968056 ns/op
BenchmarkNaiveKNN/dimension-1500_k-200_numVec-3000-12                  6         173565688 ns/op
BenchmarkNaiveKNN/dimension-1500_k-200_numVec-4000-12                  5         235831692 ns/op
BenchmarkNaiveKNN/dimension-1500_k-200_numVec-5000-12                  4         294094552 ns/op
BenchmarkNaiveKNN/dimension-1500_k-500_numVec-2000-12                  9         119243403 ns/op
BenchmarkNaiveKNN/dimension-1500_k-500_numVec-3000-12                  6         175594597 ns/op
BenchmarkNaiveKNN/dimension-1500_k-500_numVec-4000-12                  5         235258425 ns/op
BenchmarkNaiveKNN/dimension-1500_k-500_numVec-5000-12                  4         291697938 ns/op
BenchmarkNaiveKNN/dimension-1500_k-1000_numVec-2000-12                 9         119453181 ns/op
BenchmarkNaiveKNN/dimension-1500_k-1000_numVec-3000-12                 6         178747861 ns/op
BenchmarkNaiveKNN/dimension-1500_k-1000_numVec-4000-12                 5         238084617 ns/op
BenchmarkNaiveKNN/dimension-1500_k-1000_numVec-5000-12                 4         294619188 ns/op
*/

func BenchmarkNaiveKNN(b *testing.B) {
	dimensions := []int{1000, 1500}
	kValues := []int{100, 200, 500, 1000}
	numVectors := []int{2000, 3000, 4000, 5000}

	for _, dim := range dimensions {
		for _, k := range kValues {
			for _, numVec := range numVectors {
				vectors := make([]Vector, numVec)
				for i := range vectors {
					vectors[i] = Vector{
						ID:      ID(fmt.Sprintf("vec_%d", i)),
						Payload: nil,
						Point:   randomVector(dim),
					}
				}
				targetVector := Vector{
					ID:      "target",
					Payload: nil,
					Point:   randomVector(dim),
				}

				b.Run(fmt.Sprintf("dimension-%d_k-%d_numVec-%d", dim, k, numVec), func(b *testing.B) {
					knn, _ := NewNaiveKNN(vectors, CosineDistance)
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						knn.KNN(k, targetVector)
					}
				})
			}
		}
	}
}

func randomVector(dim int) []float64 {
	v := make([]float64, dim)
	for i := range v {
		v[i] = rand.Float64()
	}
	return v
}
