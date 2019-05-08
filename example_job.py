#!/usr/bin/env python3
"""
Usage:
example_job.py --clean --createjobs --run --status
"""
from slurmhelper import SLURMJob
import datetime

def some_job(x=None):
	import time, math
	time.sleep(10)
	print("starting job...")
	# The results have to be serializable via dill.
	return math.pow(x, 32)

def generate_jobs():
	import random
	for _ in range(100):
		# The kwargs dictionary has to be serializable via dill.
		yield dict(x=random.randint(2, 5))

job = SLURMJob("SLURMtest", "/home/foobar/slurm/")
job.qos = "medium"
job.max_memory = "64MB"
job.n_cpus = 1
job.time_limit = datetime.timedelta(minutes=1)

job.map(some_job, generate_jobs())

job()
