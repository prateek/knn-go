package knn

import (
	"fmt"
)

func ExampleNaiveKNN() {
	// Define the input vectors
	vecs := []Vector{
		{ID: "vec1", Point: []float64{1.0, 2.0, 3.0}},
		{ID: "vec2", Point: []float64{4.0, 5.0, 6.0}},
		{ID: "vec3", Point: []float64{7.0, 8.0, 9.0}},
	}

	// Create a new NaiveKNN instance
	knn, err := NewNaiveKNN(vecs, CosineDistance)
	if err != nil {
		fmt.Println("Failed to create NaiveKNN:", err)
		return
	}

	// Define the target vector
	target := Vector{ID: "target", Point: []float64{0.0, 0.0, 0.0}}

	// Perform the KNN search
	k := 2
	result := knn.KNN(k, target)

	// Print the result
	for _, vec := range result {
		fmt.Println(vec.ID, vec.Point)
	}

	// Output:
	// vec1 [1 2 3]
	// vec2 [4 5 6]
}
