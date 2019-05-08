# slurmhelper

This project is a small python 3.7 wrapper around the SLURM command line commands.

You get a map-like function that is converted to SLURM batches. You generate/run/view jobs using a command line interface.
SLURM options are set in python.

## Installation

`pip3 install git+https://github.com/walachey/slurmhelper.git`

## Quick How-To

* Have an iterable and a function that you want to map to the iterable.
  * Both the result of the function and the arguments provided by the iterable have to be serializable by dill.
* Create a SLURMJob object, setting the name of the SLURM job and a filesystem location to create files.
* `slurm_job.map` your function to your iterable.
* Set SLURM options such as the max. run time, memory, CPUs, ...
* Call `slurm_job()` to create a command line interface.
* Make your python script executable (`chmod +x my_script.py`).
* Run `./my_script --clean --create --run --status`.
* The executing machine has to have the SLURM commands available.
  * Usually, you run this on your login node to the SLURM system.
* Check error log with `./my_script --log`.
* Check results with `./my_script --results` (if they are printable).
* When everything is finished you can iterate over `kwargs, results` with `slurm_job.items()`.
  * This also works from another machine as long as you have the output files available (e.g. per sshfs mount).

## Example
See the provided [example job](https://github.com/walachey/slurmhelper/blob/master/example_job.py).

## FAQ
### My script had a timeout and I want to increase the limit (or other options).
Just modify your python file and increase the time limit/memory limit/whatever.
You can run the job again with `./my_script --run` and it won't delete old results and just execute the jobs that have not finished yet with the new settings.

### Can I modify/use/sell this project?
Sure. This project is licensed under the [MIT license](https://github.com/walachey/slurmhelper/blob/master/license.md).