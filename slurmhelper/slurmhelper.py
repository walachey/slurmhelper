import os
import itertools
import datetime
import zipfile
import dill
import subprocess
import errno

progress_bar = lambda x, **kwargs: x

try:
    import tqdm
    progress_bar = tqdm.tqdm
except:
    pass

def try_make_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

class SLURMJob():
    name = "randomjob"
    job_root = "./"    
    additional_output_dirs = None

    command = "python3"
    n_nodes = 1
    n_tasks = None
    n_cpus = 1
    n_gpus = 0
    max_memory = "2GB"
    time_limit = datetime.timedelta(hours=1)
    partition = None
    qos = None

    _job_file = None
    _job_fun_code = None
    _job_fun_name = None

    def __init__(self, name, job_root):
        self.additional_output_dirs = []
        self.name = name
        self.job_root = job_root

    def set_command(self, command="python3"):
        self.command = command

    def set_job_arguments(self, iterable):
        self.job_args = iterable

    def set_job_file(self, filename=None):
        self._job_file = filename

    def set_job_fun(self, fun):
        import inspect
        lines, _ = inspect.getsourcelines(fun)
        lines = (line.replace("\t", "    ") for line in lines)
        self._job_fun_code = "    ".join(lines)
        self._job_fun_name = [f for n, f in inspect.getmembers(fun) if n == "__name__"][0]

    def map(self, fun, args):
        self.set_job_arguments(args)
        self.set_job_fun(fun)

    def add_output_dir(self, dir):
        self.additional_output_dirs.append(dir)

    def get_original_job_path_filename(self):
        assert self._job_file
        return os.path.split(self._job_file)

    @property
    def job_filename(self):
        return self.job_dir + "run_job.py"

    def write_job_file(self):
        if self._job_fun_name is not None:
            job_definition = self._job_fun_code
            job_fun_name = self._job_fun_name
        else:
            if self._job_file is None:
                print("ERROR! Must call either map, set_job_fun, set_job_file.")
                exit(1)
            job_fun_name = "job.run"
            job_definition = 'sys.path.append("{}")\nimport {} as job'.format(*self.get_original_job_path_filename(), self._job_file)
        job_file = """
import sys, warnings, zipfile, dill
results_filename = sys.argv[1]
args = sys.argv[2::]
if len(args) % 2 != 0:
    print("Wrong number of args!")
    exit(1)
kwargs = dict()
for i in range(0, len(args) - 1, 2):
    kwargs[args[i]] = args[i+1]
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    {}
    results = {}(**kwargs)

with zipfile.ZipFile(results_filename, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
    with zf.open("results.dill", "w") as f:
            dill.dump(results, f)
    with zf.open("kwargs.dill", "w") as f:
            dill.dump(kwargs, f)
""".format(job_definition, job_fun_name)

        with open(self.job_filename, "w") as f:
            f.write(job_file)

    @property
    def job_dir(self):
        return self.job_root + "/" + self.name + "/"
    
    @property
    def output_dir(self):
        return self.job_dir + "output/"

    @property
    def log_dir(self):
        return self.job_dir + "log/"

    @property   
    def batch_dir(self):
        return self.job_dir + "jobs/"

    def get_open_job_count(self):
        return len([f for f in os.listdir(self.batch_dir) if f.endswith(".sbatch")])

    def get_finished_job_directories(self):
        return itertools.chain((self.output_dir,), self.additional_output_dirs)

    def get_finished_job_count(self):
        count = 0
        for dir in self.get_finished_job_directories():
            count += len([f for f in os.listdir(dir) if f.endswith(".zip")])
        return count

    def get_running_jobs(self):
        from subprocess import Popen, PIPE
        output, _ = Popen([f'squeue --format="%j" | grep {self.name}'], stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True).communicate()
        return [f for f in output.decode("ascii").split("\n") if f]

    def get_running_job_count(self):
        return len(self.get_running_jobs())

    def cancel_running_jobs(self):
        jobs = self.get_running_jobs()
        if len(jobs) == 0:
            print("No jobs are currently running.")
            return
        print("Cancelling running jobs...")
        from subprocess import Popen, PIPE
        output, _ = Popen(["scancel", "--name", ",".join(jobs)], stdin=PIPE, stdout=PIPE, stderr=PIPE).communicate()
        if output:
            print(output.decode("ascii"))

    def run_jobs(self):
        if self.get_running_job_count() != 0:
            print("Jobs are currently running! Aborting.")
            return
        jobs = os.listdir(self.batch_dir)
        if len(jobs) == 0:
            print("No prepared jobs to be run.")
            return
            
        from subprocess import Popen, PIPE
        for f in progress_bar(jobs):
            if not f.endswith(".sbatch"):
                continue
            partition = []
            if self.partition is not None:
                partition = ["--partition", self.partition]
            elif self.n_gpus > 0:
                partition = ["--partition", "gpu"]
            command = ["sbatch"] + partition + [self.batch_dir + f]
            p = Popen(command, stdin=PIPE, stdout=PIPE, stderr=PIPE)
            _, error = p.communicate()
            
            if p.returncode != 0:
                print("Running job failed with error: ")
                if error:
                    print(error.decode("ascii"))
                print("Not running more jobs.")
                return

    def clear_batch_files(self):
        for f in os.listdir(self.batch_dir):
            if f.endswith(".sbatch"):
                os.remove(self.batch_dir + f)

    def clear_log_dir(self):
        for f in os.listdir(self.log_dir):
            if f.endswith(".txt"):
                os.remove(self.log_dir + f)

    def ensure_directories(self):
        try_make_dir(self.job_dir)
        try_make_dir(self.output_dir)
        try_make_dir(self.log_dir)
        try_make_dir(self.batch_dir)

    def clear_helper_directories(self):
        self.clear_batch_files()
        self.clear_log_dir()

    def write_batch_files(self):
        created = 0
        skipped = 0
        self.clear_batch_files()
        time_limit = self.time_limit.total_seconds()
        H, M, S = time_limit // 3600, time_limit // 60, time_limit % 60
        time_limit = "{:02d}:{:02d}:{:02d}".format(int(H), int(M), int(S))
        for idx, args in enumerate(self.job_args):
            batch_filename = self.batch_dir + "job_{:04d}.sbatch".format(idx)
            output_filename_tail = f"job_{idx:04d}.zip"
            output_filename = f"{self.output_dir}/{output_filename_tail}"

            found = False
            # Check if valid results exist.
            for dir in self.get_finished_job_directories():
                local_filename = dir + "/" + output_filename_tail
                if os.path.isfile(local_filename):
                    # Check if not corrupt.
                    try:
                        with zipfile.ZipFile(local_filename, mode="r", compression=zipfile.ZIP_DEFLATED) as _:
                            found = True
                            break
                    except:
                        pass
            if found:
                skipped += 1
                continue
            else:
                created += 1

            args_string = ""
            for arg, val in args.items():
                args_string += '"{}" "{}"'.format(arg, val)
            qos_string = ""
            if self.qos:
                qos_string = f"#SBATCH --qos={self.qos}"
            gpu_string = ""
            if self.n_gpus > 0:
                gpu_string = f"#SBATCH --gres=gpu:{self.n_gpus}\nmodule load CUDA"
            task_limit_string = ""
            if self.n_tasks is not None and self.n_tasks > 0:
                task_limit_string = f"#SBATCH --ntasks-per-node={self.n_tasks}"
            sbatch = f"""#!/bin/bash
#SBATCH --job-name={self.name}_{idx:04d}
#SBATCH --error={self.log_dir}job_{idx:04d}.error.txt
#SBATCH --output={self.log_dir}job_{idx:04d}.log.txt
#SBATCH --time={time_limit}
#SBATCH --mem={self.max_memory}
#SBATCH --nodes={self.n_nodes}
#SBATCH --cpus-per-task={self.n_cpus}
{qos_string}
{gpu_string}
{task_limit_string}

cd {self.job_dir}
{self.command} run_job.py {output_filename} {args_string} && rm {batch_filename}
"""
            with open(batch_filename, "w") as f:
                f.write(sbatch)

        print("Files created: {}, skipped: {}.".format(created, skipped))

    def print_status(self):
        print("Job {}, residing in {}".format(self.name, self.job_dir))
        print("Working on callable {}.".format(self._job_file or self._job_fun_name))
        if self.partition is None:
            print("Using default partition (see scontrol show partition).")
        else:
            print("Using partition: {}".format(self.partition))
        if self.qos is None:
            print("Using default QOS (see sqos).")
        else:
            print("Using QOS: {}".format(self.qos))
        print(f"Resources per task: nodes: {self.n_nodes}, tasks: {self.n_tasks}, cpus: {self.n_cpus}, memory: {self.max_memory}")
        print("Time limit per job: {}".format(self.time_limit))
        if os.path.isdir(self.job_dir):
            n_open, n_done, n_running = self.get_open_job_count(), self.get_finished_job_count(), self.get_running_job_count()
            print("Jobs prepared: {} (running: {})".format(n_open, n_running))
            print("Jobs done: {}".format(n_done))
        else:
            print("Jobs not created. Try --createjobs.")

    def print_log(self):
        files = list(sorted((f for f in os.listdir(self.log_dir) if f.endswith(".txt"))))
        for filename in files:
            with open(self.log_dir + filename, 'r') as f:
                print(f.read())

    def print_results(self):
        files = list(sorted((f for f in os.listdir(self.output_dir) if f.endswith(".zip"))))
        for filename in files:
            with zipfile.ZipFile(self.output_dir + filename, mode="r", compression=zipfile.ZIP_DEFLATED) as zf:
                with zf.open("results.dill", "r") as f:
                    results = dill.load(f)
                with zf.open("kwargs.dill", "r") as f:
                    kwargs = dill.load(f)
                print("== Results {} {}".format(filename[:-4], "=" * 20))
                print("    \t{}\n--->\t{}".format(str(kwargs), str(results)))


    def __call__(self):
        self.check_program_arguments()

    def check_program_arguments(self):
        import argparse
        parser = argparse.ArgumentParser(description='Easy SLURM job management.')
        parser.add_argument("--createjobs", action='store_true', help="Create batch scripts to be used with SLURM.")
        parser.add_argument("--cancel", action='store_true', help="Cancel running jobs.")
        parser.add_argument("--run", action='store_true', help="Run remaining jobs.")
        parser.add_argument("--clean", action='store_true', help="Clean up log/batch directory (not results, though.).")
        parser.add_argument("--status", action='store_true', help="Print status.")
        parser.add_argument("--log", action='store_true', help="Prints the last log messages.")
        parser.add_argument("--results", action='store_true', help="Naively prints the result output of the last jobs.")
        args = parser.parse_args()

        if not any(vars(args).values()):
            parser.error('No action requested, try --help.')
        
        if args.clean:
            self.clear_helper_directories()

        if args.createjobs:
            self.ensure_directories()
            self.write_job_file()
            self.write_batch_files()

        if args.run:
            self.run_jobs()

        if args.status:
            self.print_status()
        if args.log:
            self.print_log()
        if args.results:
            self.print_results()






