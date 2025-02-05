import time
import random


def do_some_work(x, wait_time):
    time.sleep(wait_time) # Simulate actual work
    return x


def main():
    n = 4
    random_state = random.Random(1729)
    wait_times = [random_state.expovariate(0.5) for i in range(n)]
    start = time.time()
    
    results = [do_some_work(x, wait_time) for x, wait_time in enumerate(wait_times)]
    print("duration =", time.time() - start, "\nresults = ", results)
    
    
if __name__ == '__main__':
    main()