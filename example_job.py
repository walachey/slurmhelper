#!/usr/bin/env python3
"""
Usage:
example_job.py --clean --createjobs --run --status
"""
from slurmhelper import SLURMJob
import datetime

# Note that the method's arguments are all passed as strings.
def some_job(x=None):
	x = int(x)
	import time, math
	time.sleep(10)
	print("starting job...")
	return math.pow(x, 32)

def generate_jobs():
	import random
	for _ in range(100):
        # Yield values as strings or rely on their default string representation.
		yield dict(x=random.randint(2, 5))

job = SLURMJob("SLURMtest", "/home/foobar/slurm/")
job.qos = "medium"
job.max_memory = "64MB"
job.n_cpus = 1
job.time_limit = datetime.timedelta(minutes=1)

job.map(some_job, generate_jobs())

job()
