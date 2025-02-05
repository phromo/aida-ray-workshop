import time
import random

import ray
ray.init()  # Initialize ray, so this script will use the runtime


@ray.remote  # By decorating the function with ray.remote, it can now be executed as a ray task
def do_some_work(x, wait_time):
    time.sleep(wait_time) # Simulate actual work
    return x

def main():
    n = 4
    random_state = random.Random(1729)
    wait_times = [random_state.expovariate(0.5) for i in range(n)]
    start = time.time()
    # By using the .remote method on the function object, we invoke it as a ray task
    results = [do_some_work.remote(x, wait_time) for x, wait_time in enumerate(wait_times)] 
    
    # Don't get the items in order, instead take them as they become ready
    unfinished = results
    actual_results = []
    while unfinished:
        # ray.wait takes the list of references (unfinished) and divides it 
        # into two list of the ones which are finished and the ones which aren't
        finished_tasks, unfinished = ray.wait(unfinished, num_returns=1)
        for finished_task in finished_tasks:
            # get the actual result from the reference
            result = ray.get(finished_task)  
            actual_results.append(result)
    print("Actual duration =", time.time() - start, "\nresults = ", actual_results)
    
if __name__ == '__main__':
    main()