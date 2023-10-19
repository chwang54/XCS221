from typing import List, Dict, Tuple
import random

def kmeans(
    examples: List[Dict[str, float]], K: int, maxEpochs: int
) -> Tuple[List[Dict[str, float]], List[int], float]:
    
    # Initialize cluster centers by randomly selecting K examples
    centers = random.sample(examples, K)
    
    # Initialize assignments and previous assignments
    assignments = [-1 for _ in range(len(examples))]
    prev_assignments = None
    
    for epoch in range(maxEpochs):
        # Assign each example to the nearest center
        for i, example in enumerate(examples):
            min_distance = float('inf')
            for j, center in enumerate(centers):
                distance = -2 * dotProduct(example, center) + dotProduct(center, center)
                if distance < min_distance:
                    min_distance = distance
                    assignments[i] = j
        
        # Check for convergence
        if prev_assignments == assignments:
            break
        prev_assignments = assignments[:]
        
        # Update centers
        centers = [{} for _ in range(K)]
        counts = [0 for _ in range(K)]
        
        for i, example in enumerate(examples):
            increment(centers[assignments[i]], 1, example)
            counts[assignments[i]] += 1
        
        for j in range(K):
            if counts[j] > 0:
                for key in centers[j]:
                    centers[j][key] /= counts[j]
                    
    # Calculate final reconstruction loss
    loss = 0.0
    for i, example in enumerate(examples):
        loss += -2 * dotProduct(example, centers[assignments[i]]) + dotProduct(centers[assignments[i]], centers[assignments[i]])

    return centers, assignments, loss

# Test the function with sample data
examples = generateClusteringExamples(100, 5, 5)
centers, assignments, loss = kmeans(examples, 3, 100)

print("Centers:", centers)
print("Assignments:", assignments)
print("Final reconstruction loss:", loss)
