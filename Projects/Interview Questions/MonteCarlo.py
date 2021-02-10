# Solution to this Interview Query problem: https://www.interviewquery.com/questions/nightly-job

import random
import numpy as np
import matplotlib.pyplot as plt


def cost_estimate(n=10, days=365, cost=1000):
    total_overlaps = []

    for i in range(n):
        first_job = random.randint(0, 300)  # minutes between 7p and 12a: 300 min
        second_job = random.randint(0, 300)

        overlap = 0

        first_overlap = first_job < second_job <= first_job + 60
        second_overlap = second_job < first_job <= second_job + 60

        if first_overlap or second_overlap:
            overlap = 1
        total_overlaps.append(overlap)

    probability = np.mean(total_overlaps)
    est_cost = probability * days * cost

    return probability


# run Monte Carlo simulation
probabilities = []
simulations = list(range(1, 50000, 100))
for i in simulations:
    prob = cost_estimate(n=i)
    probabilities.append(prob)


plt.plot(simulations, probabilities)
plt.title('Probabilities by number of simulations')
plt.xlabel('Number of simulations')
plt.ylabel('Probability')
plt.show()