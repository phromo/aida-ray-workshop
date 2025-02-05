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
    print("First duration =", time.time() - start, "\nresults = ", results)
    
    actual_results = [ray.get(x) for x in results]  # We need to actually wait for the results to complete, the tasks we submitted are executed asynchronysly from this process
    print("Actual duration =", time.time() - start, "\nresults = ", actual_results)
    
if __name__ == '__main__':
    main()