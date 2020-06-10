from collections import defaultdict
import datetime
import humanfriendly
import numpy as np
import os
import pandas
import pytz
import shutil
import sys

def parse_cpu_time(s: str):
    # Additional parsing for 24h periods..
    days = 0
    if "-" in s:
        days, s = s.split("-")
        days = int(days)
    h, m, s = [int(i) for i in s.split(":")]
    return datetime.timedelta(days=days, hours=h, minutes=m, seconds=s)

def parse_memory(s: str):
    s = s.replace(" ", "")
    return humanfriendly.parse_size(s)

def analyse_log_file_dir(directory):
    filenames = [f for f in os.listdir(directory) if f.endswith(".log.txt")]

    all_jobs = []

    infos_to_parse = {
        "State": lambda x: x,
        "CPU Utilized": parse_cpu_time,
        "Job Wall-clock time": parse_cpu_time,
        "Memory Utilized": parse_memory
    }

    for filename in filenames:
        full_filename = os.path.join(directory, filename)

        last_modified_time = pytz.UTC.localize(datetime.datetime.fromtimestamp(os.path.getmtime(full_filename)))
        job = dict(end_time=last_modified_time)

        with open(full_filename, "r") as f:
            for line in f:
                try:
                    first, second = line.split(": ")
                    second = second[:-1] # strip newline.
                except:
                    continue
                if first in infos_to_parse:
                    second = infos_to_parse[first](second)
                else:
                    continue

                job[first] = second
        
        if "Job Wall-clock time" in job:
            job["begin_time"] = job["end_time"] - job["Job Wall-clock time"]
        
        if len(job) > 1:
            all_jobs.append(job)

    all_jobs = pandas.DataFrame(all_jobs)
    if all_jobs.shape[0] == 0:
        print("No log files available.")
        return
        
    terminal_width = shutil.get_terminal_size((80, 20)).columns

    first_date, last_date = all_jobs.begin_time.min(), all_jobs.end_time.max()
    total_seconds_runtime = (last_date - first_date).total_seconds()
    seconds_bins = np.linspace(0, total_seconds_runtime + 0.5, num=terminal_width, endpoint=True)

    all_states = all_jobs["State"].unique()
    states_over_time = defaultdict(list)
    for bin_idx in range(len(seconds_bins) - 1):
        bin_start, bin_end = seconds_bins[bin_idx], seconds_bins[bin_idx + 1]
        begin_dt = first_date + datetime.timedelta(seconds=bin_start)
        end_dt = first_date + datetime.timedelta(seconds=bin_end)
        for state in all_states:

            df = all_jobs[all_jobs["State"] == state]
            df = df[(df["end_time"] >= begin_dt) & (df["end_time"] < end_dt)]
            states_over_time[state].append(df.shape[0])
    states_over_time = pandas.DataFrame(states_over_time)
    states_over_time = states_over_time.div(states_over_time.sum(axis=1), axis=0)
    first_date_string, last_date_string = first_date.isoformat(), last_date.isoformat()
    space_len = terminal_width - len(first_date_string) - len(last_date_string)
    
    print("Timelines: " + ", ".join(all_states))
    print("{}{}{}".format(first_date_string, " " * space_len, last_date_string))

    for state in all_states:
        for val in states_over_time[state].values:
            if pandas.isnull(val) or val == 0.0:
                val = "."
            else:
                val = int((100 * val) // 10)
                if val >= 10:
                    val = "#"
                else:
                    val = str(val)
            sys.stdout.write(val)
        sys.stdout.write("\n")
    
    sys.stdout.write("\n")
    for state, df in all_jobs.groupby("State"):
        def get_stats_for(col, strip_outliers=False):
            _df = df
            if strip_outliers:
                _df = _df[_df[col].between(*np.percentile(_df[col], (5, 95)))]
            return (_df[col].mean(), _df[col].min(), _df[col].max(), _df[col].median())

        def print_stats(title, vals):
            vals = [str(v) for v in vals]
            sys.stdout.write("{:20s}: ".format(title))
            value_descriptors = ("Mean", "Min", "Max", "Median")
            for what, v in zip(value_descriptors, vals):
                space = (terminal_width - max(len(title), 20) - 2) // len(value_descriptors) - len(what) - len(v) - 2
                sys.stdout.write("{}: {}{}".format(what, v, " " * space))
            sys.stdout.write("\n")

        def print_times(title, times):
            def format(t):
                t = t.components
                def subformat(args):
                    unit, val = args
                    if val == 0:
                        return "    " if unit != "d" else ""
                    return "{:2}{} ".format(val, unit)
                return "".join(map(subformat, zip("dhms", (t.days, t.hours, t.minutes, t.seconds))))
            print_stats(title, map(format, times))
        
        print("{} (N={})".format(state, df.shape[0]))
        cpu_times = get_stats_for("CPU Utilized")
        print_times("CPU Utilized", cpu_times)
        wall_clock_times = get_stats_for("Job Wall-clock time")
        print_times("Job Wall-clock time", wall_clock_times)
        memory = get_stats_for("Memory Utilized")
        print_stats("Memory Utilized", map(humanfriendly.format_size, memory))
        

