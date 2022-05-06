import os
import itertools
import datetime
import zipfile
import dill
import getpass
import time
import subprocess
import errno

progress_bar = lambda x, **kwargs: x

try:
    import tqdm.auto
    progress_bar = tqdm.auto.tqdm
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
    partition = "main"
    qos = "standard"
    nice = None
    concurrent_job_limit = None
    # If set, jobs that exceed the max job array size will the split into multiple arrays.
    # Should be at most MaxArraySize.
    max_job_array_size = "auto"
    # Additional environment variables to set.
    # E.g. exports = "OMP_NUM_THREADS=2,MKL_NUM_THREADS=2"
    exports = ""
    # Can contain the names of additional modules to load (e.g. "OpenBLAS").
    modules = None
    # Will be added after the other #SBATCH commands in the .sbatch file.
    # This can be used for other settings that not yet have an own property.
    custom_preamble = ""
    # Whether to stream the results into a file instead of saving everything in the end.
    # Might save some RAM but needs a generator function.
    save_as_stream = False

    _job_file = None
    _job_fun_code = None
    _job_fun_name = None
    # The username used in the slurm system. This is usually the system user name and is retrieved automatically if not set.
    username = None

    def __init__(self, name, job_root, daemon_mount_dir=None):
        self.additional_output_dirs = []
        self.name = name
        self.job_root = job_root
        # The daemon mount dir is the local path where the slurm-accessible file system is mounted.
        self.daemon_mount_dir = daemon_mount_dir
        if self.is_daemon_client():
            self.map = self.map_blocking
        else:
            self.map = self.map_async

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

    def is_daemon_client(self):
        return self.daemon_mount_dir is not None

    def get_daemon_mount_root(self):
        return self.daemon_mount_dir + "/" + self.name

    def map_async(self, fun, args):
        self.set_job_arguments(args)
        self.set_job_fun(fun)

    def map_blocking(self, fun, args):
        assert self.is_daemon_client()

        if self.get_open_job_count() > 0:
            raise ValueError("Jobs are still running and/or failed.")
        self.clear_helper_directories()
        self.clear_result_files()

        self.set_job_arguments(args)
        self.set_job_fun(fun)

        import tqdm.auto

        self.write_job_file()
        self.write_batch_file()
        self.write_input_files()

        n_jobs = self.get_open_job_count()
        if n_jobs == 0:
            raise ValueError("No jobs were created.")

        trange = tqdm.auto.tqdm(total=n_jobs)
        currently_done = 0
        while True:
            done = self.get_finished_job_count()

            if done > currently_done:
                trange.update(done - currently_done)
                currently_done = done

            if done >= n_jobs:
                break

            time.sleep(2)
        trange.close()

        yield from self.items()

    def add_output_dir(self, dir):
        self.additional_output_dirs.append(dir)

    def get_original_job_path_filename(self):
        assert self._job_file
        return os.path.split(self._job_file)

    def get_job_filename(self, local_path=False):
        return self.get_job_dir(local_path=local_path) + "run_job.py"

    def get_username(self):
        if self.username is None:
            self.username = getpass.getuser()
        return self.username

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

        # Default: Just save everything into a file in the end
        saving_strategy = """
with zipfile.ZipFile(results_filename, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
    with zf.open("results.dill", "w", force_zip64=True) as f:
        dill.dump(results, f)
    with zf.open("kwargs.dill", "w") as f:
        dill.dump(kwargs, f)
        """

        if self.save_as_stream:
            saving_strategy = """
with zipfile.ZipFile(results_filename, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
    with zf.open("results.dill", "w", force_zip64=True) as f:
        for result in results:
            dill.dump(result, f)
    with zf.open("kwargs.dill", "w") as f:
        dill.dump(kwargs, f)
            """

        job_file = """
import sys, warnings, zipfile, dill
job_id = int(sys.argv[1])
results_filename = "{}".format(job_id)
input_filename = "{}".format(job_id)

if len(sys.argv) > 2:
    print("Too many arguments.")
    exit(1)

with open(input_filename, "rb") as f:
    kwargs = dill.load(f)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    {}
    results = {}(**kwargs)

{}

""".format(self._get_output_filename_format_string(),
            self._get_input_filename_format_string(),
            job_definition, job_fun_name, saving_strategy)

        with open(self.get_job_filename(local_path=True), "w") as f:
            f.write(job_file)

    def get_job_root(self, local_path=False):
        if local_path and self.is_daemon_client():
            return self.daemon_mount_dir
        return self.job_root

    def get_job_dir(self, local_path=False):
        return self.get_job_root(local_path=local_path) + "/" + self.name + "/"
    
    def get_output_dir(self, local_path=False):
        return self.get_job_dir(local_path=local_path) + "output/"

    def get_log_dir(self, local_path=False):
        return self.get_job_dir(local_path=local_path) + "log/"

    def get_input_dir(self, local_path=False):
        return self.get_job_dir(local_path=local_path) + "jobs/"

    def get_open_job_count(self):
        try:
            return len([f for f in os.listdir(self.get_input_dir(local_path=True)) if f.endswith(".dill")])
        except FileNotFoundError:
            return 0

    def get_finished_job_directories(self, local_path=False):
        return itertools.chain((self.get_output_dir(local_path=local_path),), self.additional_output_dirs)

    def get_finished_job_count(self):
        count = 0
        for dir in self.get_finished_job_directories(local_path=True):
            try:
                count += len([f for f in os.listdir(dir) if f.endswith(".zip")])
            except FileNotFoundError:
                pass
        return count

    def get_running_jobs(self, state=""):
        from subprocess import Popen, PIPE
        state = "" if not state else f"-t {state}"
        output, _ = Popen([f'squeue -u {self.get_username()} -r --format="%A %F %j" {state} --name="{self.name}"'], stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True).communicate()
        output = [f for f in output.decode("ascii").split("\n")[1:] if f] # Skip the header line.
        output = [f.split(" ") for f in output]
        return output

    def get_running_job_count(self, state=""):
        return len(self.get_running_jobs(state=state))

    def cancel_running_jobs(self):
        jobs = self.get_running_jobs()
        if len(jobs) == 0:
            print("No jobs are currently running.")
            return
        unique_job_array_ids = set([job[1] for job in jobs])
        print("Cancelling {} running jobs from {} array(s)...".format(len(jobs), len(unique_job_array_ids)))

        from subprocess import Popen, PIPE
        output, _ = Popen(["scancel", ",".join(unique_job_array_ids)], stdin=PIPE, stdout=PIPE, stderr=PIPE).communicate()
        if output:
            print(output.decode("ascii"))

    def run_jobs(self, max_jobs=None, write_job_files=True):
        if self.get_running_job_count() != 0:
            print("Jobs are currently in the queue or running! Aborting.")
            return
        jobs = os.listdir(self.get_input_dir(local_path=True))
        if len(jobs) == 0:
            print("No prepared jobs to be run.")
            return
        
        if write_job_files:
            self.write_job_file()
            self.write_batch_file()

        # Collect job array indices to run.
        indices = []
        for f in jobs:
            if not f.endswith(".dill"):
                continue
            idx = int(f[:-5].split("_")[-1])
            indices.append(idx)
        indices = list(sorted(indices))
        total_job_count = len(indices)

        # Combine consecutive sequences into one index pair (i.e. 1, 2, 3 becomes "1-3").
        def format_consecutive_sequences(indices):
            from itertools import groupby
            from operator import itemgetter
            indices = list(sorted(indices))
            formatted_indices = []
            for _, g in groupby(enumerate(indices), lambda t: t[0]-t[1]):
                g = list(map(itemgetter(1), g))
                del g[1:-1]
                g = "-".join(map(str, g))
                formatted_indices.append(g)
            return ",".join(formatted_indices)

        if self.max_job_array_size == "auto":
            try:
                from subprocess import Popen, PIPE
                output, _ = Popen(["scontrol show config | sed -n '/^MaxArraySize/s/.*= *//p'"], stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True).communicate()
                max_job_array_size = int(output.decode("ascii")) - 1
            except Exception as e:
                print("Error while querying scontrol for 'MaxArraySize': {}".format(str(e)))
        else:
            max_job_array_size = self.max_job_array_size        
        # User can overwrite max. job limit from command line.
        if max_jobs is not None and max_jobs > 0:
            max_job_array_size = max_jobs

        # Split very large jobs into multiple job arrays.
        index_groups = [indices]
        if max_job_array_size is not None:
            index_groups = []
            jobs_remaining = max_job_array_size
            while len(indices) > 0:
                # Split into chunks where each chunk's index differences are bounded.
                first_index_value = indices[0]
                n_indices_in_chunk = max_job_array_size
                for idx, index_value in enumerate(indices):
                    if (index_value - first_index_value) > max_job_array_size:
                        n_indices_in_chunk = idx
                        break
                if jobs_remaining > 0:
                    n_indices_in_chunk = min(jobs_remaining, n_indices_in_chunk)
                jobs_remaining -= n_indices_in_chunk
                one_job_indices = indices[:n_indices_in_chunk]
                del indices[:n_indices_in_chunk]
                index_groups.append(one_job_indices)
        
        total_submitted_jobs = 0
        for idx, indices in enumerate(index_groups):
            min_index = min(indices)
            indices = [i-min_index for i in indices]

            limit_concurrent_flag = f"%{self.concurrent_job_limit}" if self.concurrent_job_limit else ""
            array_command = "--array=" + format_consecutive_sequences(indices) + limit_concurrent_flag
            environment_vars = "--export=ALL,JOB_ARRAY_OFFSET={}".format(min_index)
            if self.exports:
                environment_vars = environment_vars + "," + self.exports
                
            command = ["sbatch"] + [array_command] + [environment_vars] + [self.get_batch_filename()]

            from subprocess import Popen, PIPE
            p = Popen(command, stdin=PIPE, stdout=PIPE, stderr=PIPE)
            _, error = p.communicate()
                
            if p.returncode != 0:
                print("Running jobs failed with error: ")
                if error:
                    print(error.decode("ascii"))
                print("Job command was: " + " ".join(command))
            else:
                print("Submitted job array {} of {} (i.e. {} of {} jobs).".format(
                    idx + 1, len(index_groups),
                    len(indices), total_job_count))
                total_submitted_jobs += len(indices)
                if total_submitted_jobs >= max_job_array_size:
                    break
    
    def clear_directory(self, directory, file_ending):
        try:
            for f in os.listdir(directory):
                if f.endswith(file_ending):
                    os.remove(directory + f)
        except FileNotFoundError:
            pass

    def clear_input_files(self):
        self.clear_directory(self.get_input_dir(local_path=True), ".dill")

    def clear_result_files(self):
        # Currently not part of the public interface to prevent accidental deletion.
        # So, assert we are running as a daemon.
        assert self.is_daemon_client()
        self.clear_directory(self.get_output_dir(local_path=True), ".zip")

    def clear_log_dir(self):
        self.clear_directory(self.get_log_dir(local_path=True), ".txt")

    def ensure_directories(self):
        try_make_dir(self.get_job_dir(local_path=True))
        try_make_dir(self.get_output_dir(local_path=True))
        try_make_dir(self.get_log_dir(local_path=True))
        try_make_dir(self.get_input_dir(local_path=True))

    def clear_helper_directories(self):
        self.clear_input_files()
        self.clear_log_dir()

    def _get_output_filename_format_string(self, local_path=False):
        return self.get_output_dir(local_path=local_path) + "/job_{:04d}.zip"

    def _get_output_filename_for_job_id(self, job_id, local_path=False):
        output_filename_tail = f"job_{job_id:04d}.zip"
        output_filename = f"{self.get_output_dir(local_path=local_path)}/{output_filename_tail}"
        return output_filename, output_filename_tail

    def _get_input_filename_format_string(self, local_path=False):
        return self.get_input_dir(local_path=local_path) + "/job_{:04d}.dill"

    def write_input_files(self):
        created = 0
        skipped = 0
        self.clear_input_files()

        iterable = enumerate(self.job_args)
        if not self.is_daemon_client():
            iterable = progress_bar(iterable)

        for idx, args in iterable:
            input_filename = self.get_input_dir(local_path=True) + "job_{:04d}.dill".format(idx)
            _, output_filename_tail = self._get_output_filename_for_job_id(idx, local_path=True)

            found = False
            # Check if valid results exist.
            for dir in self.get_finished_job_directories(local_path=True):
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
            
            with open(input_filename, "wb") as f:
                dill.dump(args, f)

        if not self.is_daemon_client():
            print("Files created: {}, skipped: {}.".format(created, skipped))

    def createjobs(self):
        self.ensure_directories()
        self.write_input_files()

    def get_batch_filename(self, local_path=False):
        return self.get_job_dir(local_path=local_path) + "/job_definition.sbatch"

    def get_module_loading_string(self):
        modules = self.modules or []
        if self.n_gpus > 0 and "CUDA" not in modules:
            modules.append("CUDA")
        return "\n".join(["module load {}".format(m) for m in modules])

    def write_batch_file(self):
        time_limit = self.time_limit.total_seconds()
        H, M, S = time_limit // 3600, time_limit % (60 * 60) // 60, time_limit % 60
        time_limit = "{:02d}:{:02d}:{:02d}".format(int(H), int(M), int(S))

        qos_string = ""
        if self.qos:
            qos_string = f"#SBATCH --qos={self.qos}"
        gpu_string = ""
        if self.n_gpus > 0:
            gpu_string = f"#SBATCH --gres=gpu:{self.n_gpus}"
        task_limit_string = ""
        if self.n_tasks is not None and self.n_tasks > 0:
            task_limit_string = f"#SBATCH --ntasks-per-node={self.n_tasks}"
        nice_value_string = ""
        if self.nice is not None:
            nice_value_string = f"#SBATCH --nice={self.nice}"
        partition_string = ""
        if self.partition is not None:
            partition_string = f"#SBATCH --partition={self.partition}"
        elif self.n_gpus > 0:
            partition_string = "#SBATCH --partition=gpu"
        
        sbatch = f"""#!/bin/bash
#SBATCH --job-name={self.name}
#SBATCH --error={self.get_log_dir()}job_%a_%A.error.txt
#SBATCH --output={self.get_log_dir()}job_%a_%A.log.txt
#SBATCH --time={time_limit}
#SBATCH --mem={self.max_memory}
#SBATCH --nodes={self.n_nodes}
#SBATCH --cpus-per-task={self.n_cpus}
{partition_string}
{qos_string}
{gpu_string}
{task_limit_string}
{nice_value_string}
{self.custom_preamble}
{self.get_module_loading_string()}
# JOB_ARRAY_OFFSET needs to be passed to sbatch (e.g. --export=ALL,JOB_ARRAY_OFFSET=0).
sub_job_id=$(($SLURM_ARRAY_TASK_ID + $JOB_ARRAY_OFFSET))
formatted_job_id=`printf %04d ${{sub_job_id}}`
cd {self.get_job_dir()}
{self.command} run_job.py ${{sub_job_id}} && rm {self.get_input_dir()}job_${{formatted_job_id}}.dill
"""
        with open(self.get_batch_filename(local_path=True), "w") as f:
            f.write(sbatch)


    def print_status(self):
        print("Job {}, residing in {}".format(self.name, self.get_job_dir(local_path=True)))
        print("Working on callable {}.".format(self._job_file or self._job_fun_name))
        if self.partition is None:
            print("Using default partition (see scontrol show partition).")
        else:
            print("Using partition: {}".format(self.partition))
        if self.qos is None:
            print("Using default QOS (see sqos).")
        else:
            print("Using QOS: {}".format(self.qos))
        if self.max_job_array_size == "auto":
            print("Automatically chunking large jobs into smaller job arrays.")
        else:
            print(f"Splitting up job arrays larger than {self.max_job_array_size}")
        print(f"Resources per task: nodes: {self.n_nodes}, task limit: {self.n_tasks}, cpus: {self.n_cpus}, memory: {self.max_memory}")
        print("Time limit per job: {}, concurrent job limit: {}".format(self.time_limit, self.concurrent_job_limit))
        if os.path.isdir(self.get_job_dir(local_path=True)):
            n_open, n_done, n_submitted = self.get_open_job_count(), self.get_finished_job_count(), self.get_running_job_count()
            n_pending, n_running = self.get_running_job_count("PD"), self.get_running_job_count("R")
            print("Jobs prepared: {} (running: {}, pending: {}, total queued: {})".format(n_open, n_running, n_pending, n_submitted))
            print("Jobs done: {}".format(n_done))
        else:
            print("Jobs not created. Try --createjobs.")

    def print_log(self):
        directory = self.get_log_dir(local_path=True)
        files = list(sorted((f for f in os.listdir(directory) if f.endswith(".txt"))))
        for filename in files:
            with open(directory + filename, 'r') as f:
                print(f.read())

    def get_result_filenames(self):
        return list(sorted((f for f in os.listdir(self.get_output_dir(local_path=True)) if f.endswith(".zip"))))

    def load_kwargs_results_from_result_file(self, filename, only_load_kwargs=False):
        with zipfile.ZipFile(self.get_output_dir(local_path=True) + filename, mode="r", compression=zipfile.ZIP_DEFLATED) as zf:
            with zf.open("kwargs.dill", "r") as f:
                kwargs = dill.load(f)
            if only_load_kwargs:
                return kwargs, None
                
            if not self.save_as_stream:
                with zf.open("results.dill", "r") as f:
                    results = dill.load(f)
            else:
                results = []
                with zf.open("results.dill", "r") as f:
                    while True:
                        try:
                            result = dill.load(f)
                            results.append(result)
                        except EOFError:
                            break
            
            return kwargs, results

    def print_results(self):
        files = self.get_result_filenames()
        for filename in files:
                kwargs, results = self.load_kwargs_results_from_result_file(filename)
                print("== Results {} {}".format(filename[:-4], "=" * 20))
                print("    \t{}\n--->\t{}".format(str(kwargs), str(results)))
    
    def items(self, ignore_open_jobs=False):
        if not ignore_open_jobs and (self.get_open_job_count() > 0):
            raise ValueError("Some jobs are not completed yet.")
        if self.get_finished_job_count() == 0:
            raise ValueError("There are no finished jobs.")
        for filename in self.get_result_filenames():
            try:
                kwargs, results = self.load_kwargs_results_from_result_file(filename)
            except (zipfile.BadZipFile, KeyError, EOFError) as e:
                # This probably means that a job failed while writing the zipfile (and is thus still open).
                if not ignore_open_jobs:
                    raise
                else:
                    continue
            yield kwargs, results

    def __call__(self):
        self.check_program_arguments()

    def check_program_arguments(self):
        import argparse
        parser = argparse.ArgumentParser(description='Easy SLURM job management.')
        parser.add_argument("--createjobs", action='store_true', help="Create batch scripts to be used with SLURM.")
        parser.add_argument("--cancel", action='store_true', help="Cancel running jobs.")
        parser.add_argument("--run", action='store_true', help="Run remaining jobs.")
        parser.add_argument("--clean", action='store_true', help="Clean up log/batch directory (not results, though.).")
        parser.add_argument("--removelogs", action='store_true', help="Clean up just the log directory.")
        parser.add_argument("--status", action='store_true', help="Print status.")
        parser.add_argument("--log", action='store_true', help="Prints the last log messages.")
        parser.add_argument("--results", action='store_true', help="Naively prints the result output of the last jobs.")
        parser.add_argument("--max_jobs", type=int, default=None, help="Used together with --run. Max. jobs to submit.")
        parser.add_argument("--autorun", action='store_true', help="Lingers and automatically submits next job array when current one is finished.")
        parser.add_argument("--daemon", action='store_true', help="Same as autorun but does not exit when all jobs are finished.")
        parser.add_argument("--stats", action='store_true', help="Print statistics about finished jobs.")
        args = parser.parse_args()

        if not any(vars(args).values()):
            parser.error('No action requested, try --help.')
        
        if args.cancel:
            self.cancel_running_jobs()

        if args.removelogs:
            self.clear_log_dir()

        if args.clean:
            self.clear_helper_directories()

        if args.createjobs:
            self.createjobs()

        max_jobs = args.max_jobs or None
        if args.run:
            self.run_jobs(max_jobs=max_jobs)

        if args.status:
            self.print_status()
        if args.stats:
            from . import stats
            stats.analyse_log_file_dir(self.get_log_dir(local_path=True))
        if args.log:
            self.print_log()
        if args.results:
            self.print_results()

        if args.autorun or args.daemon:
            while args.daemon or (self.get_open_job_count() > 0):
                os.system("cls||clear")
                if self.get_running_job_count() == 0:
                    self.run_jobs(max_jobs=max_jobs, write_job_files=not args.daemon)                
                self.print_status()
                if args.stats:
                    from . import stats
                    stats.analyse_log_file_dir(self.get_log_dir(local_path=True))

                if args.daemon:
                    time.sleep(10)
                else:
                    time.sleep(60 * 2)






