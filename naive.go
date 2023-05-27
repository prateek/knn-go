package knn

import (
	"container/heap"
	"fmt"
)

type pqItem struct {
	Vector Vector
	Dist   float64
}

type pQueue []*pqItem

func (pq pQueue) Len() int { return len(pq) }

func (pq pQueue) Less(i, j int) bool {
	return pq[i].Dist > pq[j].Dist
}

func (pq pQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
}

func (pq *pQueue) Push(x interface{}) {
	item := x.(*pqItem)
	*pq = append(*pq, item)
}

func (pq *pQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	*pq = old[0 : n-1]
	return item
}

var _ KNN = (*NaiveKNN)(nil)

type NaiveKNN struct {
	vectors    []Vector
	distanceFn DistanceFn
}

func NewNaiveKNN(vecs []Vector, fn DistanceFn) (*NaiveKNN, error) {
	if len(vecs) == 0 {
		return nil, fmt.Errorf("no vectors provided")
	}

	if fn == nil {
		return nil, fmt.Errorf("no distance function provided")
	}

	// ensure every vector has the same dimension
	for i := 0; i < len(vecs)-1; i++ {
		if len(vecs[i].Point) != len(vecs[i+1].Point) {
			return nil, fmt.Errorf("vectors must have the same dimension")
		}
	}

	return &NaiveKNN{
		vectors:    vecs,
		distanceFn: fn,
	}, nil
}

func (n *NaiveKNN) Search(k int, targetVector Vector) []Vector {
	if k >= len(n.vectors) {
		k = len(n.vectors)
	}

	pq := make(pQueue, 0, k)
	for _, vector := range n.vectors {
		dist := CosineDistance(targetVector, vector)
		item := &pqItem{Vector: vector, Dist: dist}

		if pq.Len() < k {
			heap.Push(&pq, item)
		} else if dist < pq[0].Dist {
			heap.Pop(&pq)
			heap.Push(&pq, item)
		}
	}

	results := make([]Vector, k)
	for i := k - 1; i >= 0; i-- {
		results[i] = heap.Pop(&pq).(*pqItem).Vector
	}

	return results
}
