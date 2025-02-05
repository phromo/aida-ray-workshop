import matplotlib.pyplot as plt
import random


random_state = random.Random(1729)
lmbd = 0.5

n = 10000
wait_times = [random_state.expovariate(lmbd) for i in range(n)]

fig = plt.figure()
plt.hist(wait_times, bins=100)
plt.xlabel('Wait times')
plt.ylabel('Frequency')
fig.savefig(f'wait_times_lambda[{lmbd}].png')

