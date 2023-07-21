import os
import sys
import json
import csv
import re
import subprocess
import pandas as pd
from halo import Halo
from termcolor import colored

from ruamel.yaml import YAML

sve_header = [
    "sve_length",
    "l1_core_bw",
    "matrix_size",
    "itrs",
    "ipc",
    "total_cycles",
    "roi_cycles",
    "total_insns",
    "l1_hits",
    "l1_misses",
    "l2_hits",
    "l2_misses",
    "mips",
]

sme_header = [
    "sme_length",
    "l1_core_bw",
    "matrix_size",
    "itrs",
    "ipc",
    "total_cycles",
    "roi_cycles",
    "total_insns",
    "l1_hits",
    "l1_misses",
    "l2_hits",
    "l2_misses",
    "mips",
]

neon_header = [
    "matrix_size",
    "itrs",
    "ipc",
    "total_cycles",
    "roi_cycles",
    "total_insns",
    "l1_hits",
    "l1_misses",
    "l2_hits",
    "l2_misses",
    "mips",
]

# vector_lengths = [128, 256, 512, 1024, 2048]

# matrix_sizes = [64, 256]

fpws = ["32", "64"]


class ModsimBenchmarks:
    def __init__(self):
        json_cfg_file = open(self.get_config_file_path(sys.argv[1:]), "r")
        json_cfg = json.load(json_cfg_file)
        self.validate_json_config(json_cfg)
        self.json_cfg = json_cfg
        self.sim_dir_path = "%s/sim" % os.path.abspath(os.getcwd())
        self.stats_dir_path = "%s/stats" % os.path.abspath(os.getcwd())
        self.sim_sst_stat_csv_path = "%s/stat.csv" % self.sim_dir_path
        self.generate_benchmarks()

    # Function which gets the JSON config file path from CLI
    def get_config_file_path(self, cli_args):
        if len(cli_args) == 0:
            raise Exception("Please pass in the JSON config file path.")
        for arg in cli_args:
            key, value = arg.split("=")
            if key == "--config":
                return value
        raise Exception("Unknown cli argument passed to data_script.py")

    def validate_file_existance(self, path):
        if not (os.path.exists(path) and os.path.isfile(path)):
            raise Exception(("File at path: %s does not exist" % path))

    def validate_dir_existance(self, path):
        if not (os.path.exists(path) and os.path.isdir(path)):
            raise Exception(("Directory at path: %s does not exist" % path))

    # Validate JSON config to check if valid paths as passed
    def validate_json_config(self, json_dict):
        keys = [
            "benchmarks",
            "yaml_configs_dir_path",
            "bin_dir_path",
            "sst_config_path",
            "sst_core_path",
            "matrix_sizes",
            "vector_lengths",
            "print_sim_output",
        ]
        keys_found = []
        for json_key in json_dict:
            if json_key in keys_found:
                raise Exception(("Duplicate property in JSON config: %s" % json_key))
            if json_key in keys:
                keys_found.append(json_key)
            else:
                raise Exception(
                    ("Unknown property found in JSON config: %s" % json_key)
                )

        self.sst_config_path = json_dict["sst_config_path"]
        self.validate_file_existance(json_dict["sst_config_path"])

        self.validate_dir_existance(json_dict["bin_dir_path"])

        self.sst_core_path = json_dict["sst_core_path"]
        self.validate_dir_existance(self.sst_core_path)

        self.input_files_dir_path = "%s/Input_files" % json_dict["bin_dir_path"]
        self.validate_dir_existance(self.input_files_dir_path)
        for fpw in fpws:
            for size in json_dict["matrix_sizes"]:
                input_file_path = "%s/fp%s_input_%s_%s_%s.dat" % (
                    self.input_files_dir_path,
                    fpw,
                    size,
                    size,
                    size,
                )
                self.validate_file_existance(input_file_path)

        self.output_files_dir_path = "%s/Ref_output" % json_dict["bin_dir_path"]
        self.validate_dir_existance(self.output_files_dir_path)
        for fpw in fpws:
            for size in json_dict["matrix_sizes"]:
                output_file_path = "%s/fp%s_ref_output_%s_%s_%s.dat" % (
                    self.output_files_dir_path,
                    fpw,
                    size,
                    size,
                    size,
                )
                self.validate_file_existance(output_file_path)

        self.validate_dir_existance(json_dict["yaml_configs_dir_path"])
        self.yaml_config_paths = {}
        for key in ["neon", "sve", "sme"]:
            yaml_cfg_name = "a64fx_%s.yaml" % key
            path = "%s/%s" % (json_dict["yaml_configs_dir_path"], yaml_cfg_name)
            self.yaml_config_paths[key] = path
            self.validate_file_existance(path)

        for benchmark in json_dict["benchmarks"]:
            path = "%s/%s" % (json_dict["bin_dir_path"], benchmark)
            self.validate_file_existance(path)

    def generate_benchmark_exec_args(self, fpw, matrix_size):
        input_file_path = "%s/fp%s_input_%s_%s_%s.dat" % (
            self.input_files_dir_path,
            fpw,
            matrix_size,
            matrix_size,
            matrix_size,
        )
        output_file_path = "%s/fp%s_ref_output_%s_%s_%s.dat" % (
            self.output_files_dir_path,
            fpw,
            matrix_size,
            matrix_size,
            matrix_size,
        )
        iterations = 10 if (matrix_size > 256) else 100
        exec_args = "%s %s %s" % (input_file_path, output_file_path, iterations)
        return [exec_args, iterations]

    def generate_sst_cli_args(
        self, se_yaml_path, exec_path, exec_args, bandwidth, sim_stat_file_path
    ):
        sst_args = (
            "--|yaml_path=%s|exec_path=%s|exec_args=%s|l1_core_bw=%sB|stat_fpath=%s"
            % (se_yaml_path, exec_path, exec_args, bandwidth, sim_stat_file_path)
        )
        return sst_args.split("|")

    def generate_neon_benchmarks(self, name, fp_width):
        matrix_sizes = self.json_cfg["matrix_sizes"]
        benchmarks = []
        file_benchmark = {
            "stat_file_path": ("%s/%s.csv" % (self.stats_dir_path, name)),
        }
        for msize in matrix_sizes:
            exec_args, itrs = self.generate_benchmark_exec_args(fp_width, msize)
            path = "%s/%s" % (self.json_cfg["bin_dir_path"], name)
            l1_core_bw = 128
            benchmark = {
                "path": path,
                "desc": (
                    "%s input_size=%s itrs=%s l1/core_bw=%sB"
                    % (name, msize, itrs, l1_core_bw)
                ),
                "itrs": itrs,
                "sst_cli_args": self.generate_sst_cli_args(
                    self.yaml_config_paths["neon"],
                    path,
                    exec_args,
                    l1_core_bw,
                    self.sim_sst_stat_csv_path,
                ),
                "matrix_size": msize,
            }
            benchmarks.append(benchmark)
        file_benchmark["benchmarks"] = benchmarks
        return file_benchmark

    def generate_sve_benchmarks(self, name, fp_width):
        matrix_sizes = self.json_cfg["matrix_sizes"]
        vector_lengths = self.json_cfg["vector_lengths"]
        benchmarks = []
        file_benchmark = {
            "stat_file_path": ("%s/%s.csv" % (self.stats_dir_path, name)),
            "sim_yaml_path": ("%s/sim.yaml" % (self.sim_dir_path)),
        }
        for msize in matrix_sizes:
            exec_args, itrs = self.generate_benchmark_exec_args(fp_width, msize)
            path = "%s/%s" % (self.json_cfg["bin_dir_path"], name)
            for vector_length in vector_lengths:
                l1_core_bw = self.scale_core_and_l1_bandwidth(vector_length)
                benchmark = {
                    "path": path,
                    "desc": (
                        "%s input_size=%s sve_vl=%s itrs=%s l1/core_bw=%sB"
                        % (name, msize, vector_length, itrs, l1_core_bw)
                    ),
                    "sve_length": vector_length,
                    "l1_core_bw": l1_core_bw,
                    "itrs": itrs,
                    "sst_cli_args": self.generate_sst_cli_args(
                        file_benchmark["sim_yaml_path"],
                        path,
                        exec_args,
                        l1_core_bw,
                        self.sim_sst_stat_csv_path,
                    ),
                    "matrix_size": msize,
                }
                benchmarks.append(benchmark)
        file_benchmark["benchmarks"] = benchmarks
        return file_benchmark

    def generate_sme_benchmarks(self, name, fp_width):
        matrix_sizes = self.json_cfg["matrix_sizes"]
        vector_lengths = self.json_cfg["vector_lengths"]
        benchmarks = []
        file_benchmark = {
            "stat_file_path": ("%s/%s.csv" % (self.stats_dir_path, name)),
            "sim_yaml_path": ("%s/sim.yaml" % (self.sim_dir_path)),
        }
        for msize in matrix_sizes:
            exec_args, itrs = self.generate_benchmark_exec_args(fp_width, msize)
            path = "%s/%s" % (self.json_cfg["bin_dir_path"], name)
            for vector_length in vector_lengths:
                l1_core_bw = self.scale_core_and_l1_bandwidth(vector_length)
                benchmark = {
                    "path": path,
                    "desc": (
                        "%s input_size=%s sme_svl=%s itrs=%s l1/core_bw=%sB"
                        % (name, msize, vector_length, itrs, l1_core_bw)
                    ),
                    "sme_svl": vector_length,
                    "l1_core_bw": l1_core_bw,
                    "itrs": itrs,
                    "sst_cli_args": self.generate_sst_cli_args(
                        file_benchmark["sim_yaml_path"],
                        path,
                        exec_args,
                        l1_core_bw,
                        self.sim_sst_stat_csv_path,
                    ),
                    "matrix_size": msize,
                }
                benchmarks.append(benchmark)
        file_benchmark["benchmarks"] = benchmarks
        return file_benchmark

    def generate_benchmarks(self):
        otp_dict = {
            "neon": {"count": 0, "file_benchmarks": []},
            "sve": {"count": 0, "file_benchmarks": []},
            "sme": {"count": 0, "file_benchmarks": []},
        }
        for benchmark_name in self.json_cfg["benchmarks"]:
            name_split = benchmark_name.split("_")
            fp_wdith = self.get_fp_width(name_split[1])
            if "neon" in name_split:
                file_benchmark = self.generate_neon_benchmarks(benchmark_name, fp_wdith)
                otp_dict["neon"]["count"] += len(file_benchmark["benchmarks"])
                otp_dict["neon"]["file_benchmarks"].append(file_benchmark)
            elif "sme" in name_split:
                file_benchmark = self.generate_sme_benchmarks(benchmark_name, fp_wdith)
                otp_dict["sme"]["count"] += len(file_benchmark["benchmarks"])
                otp_dict["sme"]["file_benchmarks"].append(file_benchmark)
            else:
                file_benchmark = self.generate_sve_benchmarks(benchmark_name, fp_wdith)
                otp_dict["sve"]["count"] += len(file_benchmark["benchmarks"])
                otp_dict["sve"]["file_benchmarks"].append(file_benchmark)

        self.benchmarks = otp_dict

    # Get fp width
    def get_fp_width(self, fp_str):
        return int(fp_str[2:])

    # Scale core and L1 bandwidth value per vector length
    def scale_core_and_l1_bandwidth(self, vector_length):
        def_value = 128
        return (
            def_value
            if (vector_length <= 512)
            else (def_value * int((vector_length / 512)))
        )

    def get_yaml_data(self, path):
        with open(path) as yaml_file:
            yaml = YAML()
            return yaml.load(yaml_file)

    def save_yaml_data(self, file_path, yaml_data):
        with open(file_path, "w") as yaml_file:
            yaml = YAML()
            yaml.dump(yaml_data, yaml_file)

    def generate_sve_yaml_with_scalebw(self, sve_yaml_path, output_path, sve_vl):
        ldbw = self.scale_core_and_l1_bandwidth(sve_vl)
        stbw = int(ldbw / 2)

        sve_yaml_data = self.get_yaml_data(sve_yaml_path)
        sve_yaml_data["Core"]["Vector-Length"] = sve_vl
        sve_yaml_data["LSQ-L1-Interface"]["Load-Bandwidth"] = ldbw
        sve_yaml_data["LSQ-L1-Interface"]["Store-Bandwidth"] = stbw

        self.save_yaml_data(output_path, sve_yaml_data)

    def generate_sme_yaml_with_scalebw(self, sve_yaml_path, output_path, sme_svl):
        ldbw = self.scale_core_and_l1_bandwidth(sme_svl)
        stbw = int(ldbw / 2)

        sme_yaml_data = self.get_yaml_data(sve_yaml_path)
        sme_yaml_data["Core"]["Vector-Length"] = sme_svl
        sme_yaml_data["Core"]["Streaming-Vector-Length"] = sme_svl
        sme_yaml_data["LSQ-L1-Interface"]["Load-Bandwidth"] = ldbw
        sme_yaml_data["LSQ-L1-Interface"]["Store-Bandwidth"] = stbw

        self.save_yaml_data(output_path, sme_yaml_data)

    def generate_sst_subprocess_cmd(self, sst_cli_args):
        return [("%s/sst" % self.sst_core_path), self.sst_config_path] + sst_cli_args

    def run_subprocess(self, cmd):
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout

    def print_benchmark_category(self, category, count):
        print(
            colored("Running %s %s benchmarks", attrs=["bold"])
            % (count, category.upper())
        )

    def generate_csv_row(self, category, benchmark, stdout):
        simeng_stats = self.parse_output(stdout)
        sst_stats = self.parse_sim_sst_stats()
        csv_row = {}
        csv_row["matrix_size"] = benchmark["matrix_size"]
        csv_row["itrs"] = benchmark["itrs"]
        if category == "sve":
            csv_row["sve_length"] = benchmark["sve_length"]
            csv_row["l1_core_bw"] = benchmark["l1_core_bw"]
        elif category == "sme":
            csv_row["sme_length"] = benchmark["sme_svl"]
            csv_row["l1_core_bw"] = benchmark["l1_core_bw"]
        return csv_row | (simeng_stats | sst_stats)

    def generate_stat_file_from_df(self, df, stat_fpath):
        df.to_csv(stat_fpath, index=False)
        print(df)
        print("Statistics written to: %s" % colored(stat_fpath, "light_blue"))
        return

    def exit_benchmark_on_keyboard_interrupt(self, df, stat_fpath):
        print(
            colored(
                "\nBenchmarking interrupted collecting stats for completed benchmarks",
                "red",
                attrs=["bold"],
            )
        )
        self.generate_stat_file_from_df(df, stat_fpath)
        exit(1)

    def parse_output(self, output: str):
        roi_cycles_reg = re.compile("Done in ([0-9]+) cycles!")
        total_cycles_reg = re.compile("\[SimEng\] cycles: ([0-9]+)")
        ipc_reg = re.compile("\[SimEng\] ipc: ([0-9]+\.[0-9]+)")
        insn_reg = re.compile("\[SimEng\] retired: ([0-9]+)")
        mips_reg = re.compile(
            "\[SimEng\] Finished [0-9]+ ticks in [0-9]+ms \([0-9]+ kHz, ([0-9]+\.[0-9]+) MIPS\)"
        )

        lines = output.splitlines()
        parsed_output = {"roi_cycles": -1, "total_cycles": -1, "ipc": 0}
        for line in lines:
            if re.match(roi_cycles_reg, line):
                search = re.search(roi_cycles_reg, line)
                parsed_output["roi_cycles"] = int(search.group(1))
            elif re.match(insn_reg, line):
                search = re.search(insn_reg, line)
                parsed_output["total_insns"] = int(search.group(1))
            elif re.match(ipc_reg, line):
                search = re.search(ipc_reg, line)
                parsed_output["ipc"] = float(search.group(1))
            elif re.match(total_cycles_reg, line):
                search = re.search(total_cycles_reg, line)
                parsed_output["total_cycles"] = int(search.group(1))
            elif re.match(mips_reg, line):
                search = re.search(mips_reg, line)
                parsed_output["mips"] = float(search.group(1))
        return parsed_output

    def parse_sim_sst_stats(self):
        csv_pd = pd.read_csv(self.sim_sst_stat_csv_path)
        l1h, l1m, l2h, l2m = csv_pd[" Sum.u64"]
        parsed_csv = {
            "l1_hits": int(l1h),
            "l1_misses": int(l1m),
            "l2_hits": int(l2h),
            "l2_misses": int(l2m),
        }
        return parsed_csv

    def start_benchmark_spinner(self, b_text):
        bench_str = "Running %s" % b_text
        self.b_spinner = {
            "spinner": Halo(text=bench_str, spinner="dots"),
            "run": True,
            "desc": b_text,
        }
        self.b_spinner["spinner"].start()

    def stop_and_persist_benchmark_spinner(self):
        if self.b_spinner and self.b_spinner["run"]:
            bench_str = "Completed %s" % self.b_spinner["desc"]
            self.b_spinner["desc"] = ""
            self.b_spinner["run"] = False
            self.b_spinner["spinner"].stop_and_persist(
                symbol=colored("✔", "light_green", attrs=["bold"]), text=bench_str
            )

    def run_benchmark(self, subproc_args, total_count, current_count, desc):
        spinner_text = "[%s/%s] %s" % (
            current_count,
            total_count,
            desc,
        )
        self.start_benchmark_spinner(spinner_text)
        cmd = self.generate_sst_subprocess_cmd(subproc_args)
        stdout = self.run_subprocess(cmd)
        if self.json_cfg["print_sim_output"]:
            print(stdout)
        self.stop_and_persist_benchmark_spinner()
        return stdout

    def run_neon(self, b_suite):
        file_benchmarks = b_suite["file_benchmarks"]
        total_benchmark_count = b_suite["count"]
        completed_count = 0

        self.print_benchmark_category("neon", total_benchmark_count)

        for file_benchmark in file_benchmarks:
            benchmarks = file_benchmark["benchmarks"]

            row_df = pd.DataFrame(columns=neon_header)
            current_count = completed_count + 1
            stat_file_path = file_benchmark["stat_file_path"]
            try:
                for benchmark in benchmarks:
                    stdout = self.run_benchmark(
                        benchmark["sst_cli_args"],
                        total_benchmark_count,
                        current_count,
                        benchmark["desc"],
                    )
                    csv_row = self.generate_csv_row("neon", benchmark, stdout)
                    row_df = pd.concat(
                        [row_df, pd.DataFrame([csv_row])], ignore_index=True
                    )
                    current_count += 1
            except KeyboardInterrupt:
                self.exit_benchmark_on_keyboard_interrupt(row_df, stat_file_path)
            else:
                completed_count = current_count
                self.generate_stat_file_from_df(row_df, stat_file_path)

    def run_sve(self, b_suite):
        file_benchmarks = b_suite["file_benchmarks"]
        total_benchmark_count = b_suite["count"]
        completed_count = 0

        self.print_benchmark_category("sve", total_benchmark_count)

        for file_benchmark in file_benchmarks:
            benchmarks = file_benchmark["benchmarks"]

            row_df = pd.DataFrame(columns=sve_header)
            current_count = completed_count + 1
            stat_file_path = file_benchmark["stat_file_path"]
            sim_yaml_file_path = file_benchmark["sim_yaml_path"]
            sve_yaml_path = self.yaml_config_paths["sve"]

            try:
                for benchmark in benchmarks:
                    sve_vl = benchmark["sve_length"]
                    self.generate_sve_yaml_with_scalebw(
                        sve_yaml_path, sim_yaml_file_path, sve_vl
                    )
                    stdout = self.run_benchmark(
                        benchmark["sst_cli_args"],
                        total_benchmark_count,
                        current_count,
                        benchmark["desc"],
                    )
                    csv_row = self.generate_csv_row("sve", benchmark, stdout)
                    row_df = pd.concat(
                        [row_df, pd.DataFrame([csv_row])], ignore_index=True
                    )
                    current_count += 1
            except KeyboardInterrupt:
                self.exit_benchmark_on_keyboard_interrupt(row_df, stat_file_path)
            else:
                completed_count = current_count
                self.generate_stat_file_from_df(row_df, stat_file_path)

    def run_sme(self, b_suite):
        file_benchmarks = b_suite["file_benchmarks"]
        total_benchmark_count = b_suite["count"]
        completed_count = 0

        self.print_benchmark_category("sme", total_benchmark_count)

        for file_benchmark in file_benchmarks:
            benchmarks = file_benchmark["benchmarks"]

            row_df = pd.DataFrame(columns=sme_header)
            current_count = completed_count + 1
            stat_file_path = file_benchmark["stat_file_path"]
            sim_yaml_file_path = file_benchmark["sim_yaml_path"]
            sme_yaml_path = self.yaml_config_paths["sme"]

            try:
                for benchmark in benchmarks:
                    sme_svl = benchmark["sme_svl"]
                    self.generate_sme_yaml_with_scalebw(
                        sme_yaml_path, sim_yaml_file_path, sme_svl
                    )
                    stdout = self.run_benchmark(
                        benchmark["sst_cli_args"],
                        total_benchmark_count,
                        current_count,
                        benchmark["desc"],
                    )
                    csv_row = self.generate_csv_row("sme", benchmark, stdout)
                    row_df = pd.concat(
                        [row_df, pd.DataFrame([csv_row])], ignore_index=True
                    )
                    current_count += 1
            except KeyboardInterrupt:
                self.exit_benchmark_on_keyboard_interrupt(row_df, stat_file_path)
            else:
                completed_count = current_count
                self.generate_stat_file_from_df(row_df, stat_file_path)

    def run(self):
        self.run_neon(self.benchmarks["neon"])
        self.run_sve(self.benchmarks["sve"])
        self.run_sme(self.benchmarks["sme"])


ModsimBenchmarks().run()
