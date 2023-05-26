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
BenchmarkNaiveKNN/dimension-1000_k-100_numVec-2000-12                 13          78,137,103 ns/op
BenchmarkNaiveKNN/dimension-1000_k-100_numVec-3000-12                  9         116,915,051 ns/op
BenchmarkNaiveKNN/dimension-1000_k-100_numVec-4000-12                  7         154,007,631 ns/op
BenchmarkNaiveKNN/dimension-1000_k-100_numVec-5000-12                  6         193,157,847 ns/op
BenchmarkNaiveKNN/dimension-1000_k-200_numVec-2000-12                 14          77,251,137 ns/op
BenchmarkNaiveKNN/dimension-1000_k-200_numVec-3000-12                  9         117,093,866 ns/op
BenchmarkNaiveKNN/dimension-1000_k-200_numVec-4000-12                  7         155,256,643 ns/op
BenchmarkNaiveKNN/dimension-1000_k-200_numVec-5000-12                  6         192,989,180 ns/op
BenchmarkNaiveKNN/dimension-1000_k-500_numVec-2000-12                 14          77,250,690 ns/op
BenchmarkNaiveKNN/dimension-1000_k-500_numVec-3000-12                  9         118,173,732 ns/op
BenchmarkNaiveKNN/dimension-1000_k-500_numVec-4000-12                  7         154,997,946 ns/op
BenchmarkNaiveKNN/dimension-1000_k-500_numVec-5000-12                  6         193,248,389 ns/op
BenchmarkNaiveKNN/dimension-1000_k-1000_numVec-2000-12                14          78,970,607 ns/op
BenchmarkNaiveKNN/dimension-1000_k-1000_numVec-3000-12                 9         117,289,245 ns/op
BenchmarkNaiveKNN/dimension-1000_k-1000_numVec-4000-12                 7         155,286,923 ns/op
BenchmarkNaiveKNN/dimension-1000_k-1000_numVec-5000-12                 6         198,260,312 ns/op
BenchmarkNaiveKNN/dimension-1500_k-100_numVec-2000-12                  9         118,312,208 ns/op
BenchmarkNaiveKNN/dimension-1500_k-100_numVec-3000-12                  6         176,361,944 ns/op
BenchmarkNaiveKNN/dimension-1500_k-100_numVec-4000-12                  5         234,593,858 ns/op
BenchmarkNaiveKNN/dimension-1500_k-100_numVec-5000-12                  4         293,964,562 ns/op
BenchmarkNaiveKNN/dimension-1500_k-200_numVec-2000-12                  9         115,968,056 ns/op
BenchmarkNaiveKNN/dimension-1500_k-200_numVec-3000-12                  6         173,565,688 ns/op
BenchmarkNaiveKNN/dimension-1500_k-200_numVec-4000-12                  5         235,831,692 ns/op
BenchmarkNaiveKNN/dimension-1500_k-200_numVec-5000-12                  4         294,094,552 ns/op
BenchmarkNaiveKNN/dimension-1500_k-500_numVec-2000-12                  9         119,243,403 ns/op
BenchmarkNaiveKNN/dimension-1500_k-500_numVec-3000-12                  6         175,594,597 ns/op
BenchmarkNaiveKNN/dimension-1500_k-500_numVec-4000-12                  5         235,258,425 ns/op
BenchmarkNaiveKNN/dimension-1500_k-500_numVec-5000-12                  4         291,697,938 ns/op
BenchmarkNaiveKNN/dimension-1500_k-1000_numVec-2000-12                 9         119,453,181 ns/op
BenchmarkNaiveKNN/dimension-1500_k-1000_numVec-3000-12                 6         178,747,861 ns/op
BenchmarkNaiveKNN/dimension-1500_k-1000_numVec-4000-12                 5         238,084,617 ns/op
BenchmarkNaiveKNN/dimension-1500_k-1000_numVec-5000-12                 4         294,619,188 ns/op
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
						knn.Search(k, targetVector)
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
