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

simeng_stats__ = []

sve_header = [
    "sve_length",
    "l1_core_bw",
    "matrix_size",
    "itrs",
    "ipc",
    "cycles",
    "retired",
    "l1_hits",
    "l1_misses",
    "l2_hits",
    "l2_misses",
    "branch.executed",
    "branch.mispredict",
    "branch.missrate",
    "decode.earlyFlushes",
    "dispatch.rsStalls",
    "fetch.branchStalls",
    "flushes",
    "issue.backendStalls",
    "issue.frontendStalls",
    "issue.portBusyStalls",
    "lsq.loadViolations",
    "rename.allocationStalls",
    "rename.lqStalls",
    "rename.robStalls",
    "rename.sqStalls",
    "rob.smesCommitted",
    "rob.svesCommitted",
]

sme_header = [
    "sme_length",
    "l1_core_bw",
    "matrix_size",
    "itrs",
    "ipc",
    "cycles",
    "retired",
    "l1_hits",
    "l1_misses",
    "l2_hits",
    "l2_misses",
    "branch.executed",
    "branch.mispredict",
    "branch.missrate",
    "decode.earlyFlushes",
    "dispatch.rsStalls",
    "fetch.branchStalls",
    "flushes",
    "issue.backendStalls",
    "issue.frontendStalls",
    "issue.portBusyStalls",
    "lsq.loadViolations",
    "rename.allocationStalls",
    "rename.lqStalls",
    "rename.robStalls",
    "rename.sqStalls",
    "rob.smesCommitted",
    "rob.svesCommitted",
]

neon_header = [
    "matrix_size",
    "itrs",
    "ipc",
    "cycles",
    "retired",
    "l1_hits",
    "l1_misses",
    "l2_hits",
    "l2_misses",
    "branch.executed",
    "branch.mispredict",
    "branch.missrate",
    "decode.earlyFlushes",
    "dispatch.rsStalls",
    "fetch.branchStalls",
    "flushes",
    "issue.backendStalls",
    "issue.frontendStalls",
    "issue.portBusyStalls",
    "lsq.loadViolations",
    "rename.allocationStalls",
    "rename.lqStalls",
    "rename.robStalls",
    "rename.sqStalls",
    "rob.smesCommitted",
    "rob.svesCommitted",
]

fpws = ["32", "64"]


class ModsimBenchmarks:
    def __init__(self):
        json_cfg_file = open(self.get_config_file_path(sys.argv[1:]), "r")
        json_cfg = json.load(json_cfg_file)
        self.sim_dir_path = "%s/sim" % os.path.abspath(os.getcwd())
        self.stats_dir_path = "%s/stats" % os.path.abspath(os.getcwd())
        self.output_dir_path = "%s/output" % os.path.abspath(os.getcwd())
        if not os.path.exists(self.sim_dir_path):
            os.mkdir(self.sim_dir_path)

        if not os.path.exists(self.stats_dir_path):
            os.mkdir(self.stats_dir_path)

        if not os.path.exists(self.output_dir_path):
            os.mkdir(self.output_dir_path)

        self.validate_json_config(json_cfg)
        self.json_cfg = json_cfg
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
            "vector_lengths",
            "print_sim_output",
            "vanilla",
            "simeng_path",
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

        self.vanilla_se = json_dict["vanilla"]
        self.simeng_path = json_dict["simeng_path"]
        self.validate_file_existance(self.simeng_path)

    def generate_vanilla_simeng_args(self, bin_path, yaml_path, simeng_path):
        return [simeng_path, yaml_path, bin_path]

    def generate_sst_cli_args(
        self,
        se_yaml_path,
        exec_path,
        bandwidth,
        sim_stat_file_path,
        simeng_stat_fpath,
    ):
        sst_args = (
            "--|yaml_path=%s|exec_path=%s|l1_core_bw=%sB|stat_fpath=%s|simeng_stat_csv_path=%s"
            % (
                se_yaml_path,
                exec_path,
                bandwidth,
                sim_stat_file_path,
                simeng_stat_fpath,
            )
        )
        return sst_args.split("|")

    def generate_neon_benchmarks(
        self, benchmark_name, fp_width, matrix_size, iterations
    ):
        prefix = "se" if self.vanilla_se else "sst"
        l1_core_bw = 128
        yaml_path = "%s/%s_%s_neon.yaml" % (self.sim_dir_path, prefix, fp_width)
        file_benchmark = {
            "output_stat_file_path": "%s/%s_%s.csv"
            % (self.stats_dir_path, prefix, benchmark_name)
        }
        benchmark = {
            "sst_stat_file_path": (
                "%s/sst_%s.csv" % (self.sim_dir_path, benchmark_name)
            ),
            "se_stat_file_path": ("%s/se_%s.csv" % (self.sim_dir_path, benchmark_name)),
            "path": "%s/%s" % (self.json_cfg["bin_dir_path"], benchmark_name),
            "output_path": (
                "%s/%s_%s.output" % (self.output_dir_path, prefix, benchmark_name)
            ),
            "l1_core_bw": l1_core_bw,
            "itrs": iterations,
            "yaml_path": yaml_path,
            "matrix_size": matrix_size,
            "fp_width": fp_width,
            "desc": ("%s itrs=%s msize=%s" % (benchmark_name, iterations, matrix_size)),
        }
        benchmark["simeng_cli_args"] = self.generate_vanilla_simeng_args(
            benchmark["path"], benchmark["yaml_path"], self.simeng_path
        )
        benchmark["sst_cli_args"] = self.generate_sst_cli_args(
            benchmark["yaml_path"],
            benchmark["path"],
            l1_core_bw,
            benchmark["sst_stat_file_path"],
            benchmark["se_stat_file_path"],
        )
        file_benchmark["benchmarks"] = [benchmark]
        return file_benchmark

    def generate_sve_benchmarks(
        self, benchmark_name, fp_width, matrix_size, iterations
    ):
        prefix = "se" if self.vanilla_se else "sst"
        file_benchmark = {
            "output_stat_file_path": "%s/%s_%s.csv"
            % (self.stats_dir_path, prefix, benchmark_name),
            "benchmarks": [],
        }
        for vector_length in self.json_cfg["vector_lengths"]:
            l1_core_bw = self.scale_core_and_l1_bandwidth(vector_length)
            yaml_path = "%s/%s_%s_vl%s_sve.yaml" % (
                self.sim_dir_path,
                prefix,
                fp_width,
                vector_length,
            )
            benchmark = {
                "sst_stat_file_path": (
                    "%s/sst_vl%s_%s.csv"
                    % (self.sim_dir_path, vector_length, benchmark_name)
                ),
                "se_stat_file_path": (
                    "%s/se_vl%s_%s.csv"
                    % (self.sim_dir_path, vector_length, benchmark_name)
                ),
                "path": "%s/%s" % (self.json_cfg["bin_dir_path"], benchmark_name),
                "output_path": (
                    "%s/%s_vl%s_%s.output"
                    % (self.output_dir_path, prefix, vector_length, benchmark_name)
                ),
                "sve_vl": vector_length,
                "l1_core_bw": l1_core_bw,
                "itrs": iterations,
                "yaml_path": yaml_path,
                "matrix_size": matrix_size,
                "fp_width": fp_width,
                "desc": (
                    "%s vl=%s itrs=%s msize=%s l1_core_bw=%sB"
                    % (
                        benchmark_name,
                        vector_length,
                        iterations,
                        matrix_size,
                        l1_core_bw,
                    )
                ),
            }
            benchmark["simeng_cli_args"] = self.generate_vanilla_simeng_args(
                benchmark["path"], benchmark["yaml_path"], self.simeng_path
            )
            benchmark["sst_cli_args"] = self.generate_sst_cli_args(
                benchmark["yaml_path"],
                benchmark["path"],
                l1_core_bw,
                benchmark["sst_stat_file_path"],
                benchmark["se_stat_file_path"],
            )
            file_benchmark["benchmarks"].append(benchmark)
        return file_benchmark

    def generate_sme_benchmarks(
        self, benchmark_name, fp_width, matrix_size, iterations
    ):
        prefix = "se" if self.vanilla_se else "sst"
        file_benchmark = {
            "output_stat_file_path": "%s/%s_%s.csv"
            % (self.stats_dir_path, prefix, benchmark_name),
            "benchmarks": [],
        }
        for vector_length in self.json_cfg["vector_lengths"]:
            l1_core_bw = self.scale_core_and_l1_bandwidth(vector_length)
            yaml_path = "%s/%s_%s_svl%s_sme.yaml" % (
                self.sim_dir_path,
                prefix,
                fp_width,
                vector_length,
            )
            benchmark = {
                "sst_stat_file_path": (
                    "%s/sst_svl%s_%s.csv"
                    % (self.sim_dir_path, vector_length, benchmark_name)
                ),
                "se_stat_file_path": (
                    "%s/se_svl%s_%s.csv"
                    % (self.sim_dir_path, vector_length, benchmark_name)
                ),
                "path": "%s/%s" % (self.json_cfg["bin_dir_path"], benchmark_name),
                "output_path": (
                    "%s/%s_svl%s_%s.output"
                    % (self.output_dir_path, prefix, vector_length, benchmark_name)
                ),
                "sme_svl": vector_length,
                "l1_core_bw": l1_core_bw,
                "itrs": iterations,
                "yaml_path": yaml_path,
                "matrix_size": matrix_size,
                "fp_width": fp_width,
                "desc": (
                    "%s svl=%s itrs=%s msize=%s l1_core_bw=%sB"
                    % (
                        benchmark_name,
                        vector_length,
                        iterations,
                        matrix_size,
                        l1_core_bw,
                    )
                ),
            }
            benchmark["simeng_cli_args"] = self.generate_vanilla_simeng_args(
                benchmark["path"], benchmark["yaml_path"], self.simeng_path
            )
            benchmark["sst_cli_args"] = self.generate_sst_cli_args(
                benchmark["yaml_path"],
                benchmark["path"],
                l1_core_bw,
                benchmark["sst_stat_file_path"],
                benchmark["se_stat_file_path"],
            )
            file_benchmark["benchmarks"].append(benchmark)
        return file_benchmark

    def generate_benchmarks(self):
        otp_dict = {
            "neon": {"count": 0, "file_benchmarks": []},
            "sve": {"count": 0, "file_benchmarks": []},
            "sme": {"count": 0, "file_benchmarks": []},
        }
        for benchmark_name in self.json_cfg["benchmarks"]:
            name_split = benchmark_name.split("_")
            fp_wdith = name_split[0]
            btype = name_split[1]
            matrix_size = int(name_split[2])
            iterations = int(name_split[3][:-7])
            if "neon" == btype:
                file_benchmark = self.generate_neon_benchmarks(
                    benchmark_name, fp_wdith, matrix_size, iterations
                )
                otp_dict["neon"]["count"] += len(file_benchmark["benchmarks"])
                otp_dict["neon"]["file_benchmarks"].append(file_benchmark)
            elif "sme" == btype:
                file_benchmark = self.generate_sme_benchmarks(
                    benchmark_name, fp_wdith, matrix_size, iterations
                )
                otp_dict["neon"]["count"] += len(file_benchmark["benchmarks"])
                otp_dict["neon"]["file_benchmarks"].append(file_benchmark)
            else:
                file_benchmark = self.generate_sve_benchmarks(
                    benchmark_name, fp_wdith, matrix_size, iterations
                )
                otp_dict["neon"]["count"] += len(file_benchmark["benchmarks"])
                otp_dict["neon"]["file_benchmarks"].append(file_benchmark)
        self.benchmarks = otp_dict

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

    def generate_neon_yaml_file(self, output_path, is_vanilla=False, se_stat_path=""):
        neon_yaml_path = self.yaml_config_paths["neon"]
        neon_yaml_data = self.get_yaml_data(neon_yaml_path)
        if is_vanilla:
            neon_yaml_data["CSV-Stats-Path"] = se_stat_path

        self.save_yaml_data(output_path, neon_yaml_data)

    def generate_sve_yaml_with_scalebw(
        self, output_path, sve_vl, is_vanilla=False, se_stat_path=""
    ):
        sve_yaml_path = self.yaml_config_paths["sve"]
        ldbw = self.scale_core_and_l1_bandwidth(sve_vl)
        stbw = int(ldbw / 2)

        sve_yaml_data = self.get_yaml_data(sve_yaml_path)
        sve_yaml_data["Core"]["Vector-Length"] = sve_vl
        sve_yaml_data["LSQ-L1-Interface"]["Load-Bandwidth"] = ldbw
        sve_yaml_data["LSQ-L1-Interface"]["Store-Bandwidth"] = stbw

        if is_vanilla:
            sve_yaml_data["CSV-Stats-Path"] = se_stat_path

        self.save_yaml_data(output_path, sve_yaml_data)

    def generate_sme_yaml_with_scalebw(
        self, output_path, sme_svl, is_vanilla=False, se_stat_path=""
    ):
        sme_yaml_path = self.yaml_config_paths["sme"]
        ldbw = self.scale_core_and_l1_bandwidth(sme_svl)
        stbw = int(ldbw / 2)

        sme_yaml_data = self.get_yaml_data(sme_yaml_path)
        sme_yaml_data["Core"]["Vector-Length"] = sme_svl
        sme_yaml_data["Core"]["Streaming-Vector-Length"] = sme_svl
        sme_yaml_data["LSQ-L1-Interface"]["Load-Bandwidth"] = ldbw
        sme_yaml_data["LSQ-L1-Interface"]["Store-Bandwidth"] = stbw

        if is_vanilla:
            sme_yaml_data["CSV-Stats-Path"] = se_stat_path

        self.save_yaml_data(output_path, sme_yaml_data)

    def generate_yaml_config(self, btype, benchmark):
        output_path = benchmark["yaml_path"]
        se_stat_path = benchmark["se_stat_file_path"]
        if btype == "neon":
            self.generate_neon_yaml_file(output_path, self.vanilla_se, se_stat_path)
        elif btype == "sve":
            vl = benchmark["sve_vl"]
            self.generate_sve_yaml_with_scalebw(
                output_path, vl, self.vanilla_se, se_stat_path
            )
        else:
            svl = benchmark["sme_svl"]
            self.generate_sme_yaml_with_scalebw(
                output_path, svl, self.vanilla_se, se_stat_path
            )

    def generate_sst_subprocess_cmd(self, sst_cli_args):
        return [("%s/sst" % self.sst_core_path), self.sst_config_path] + sst_cli_args

    def run_subprocess(self, cmd):
        print(cmd)
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout

    def print_benchmark_category(self, category, count):
        print(
            colored("Running %s %s benchmarks", attrs=["bold"])
            % (count, category.upper())
        )

    def generate_csv_row(self, category, benchmark, stdout):
        se_csv_fpath = benchmark["se_stat_file_path"]
        se_csv_stats = self.get_simeng_csv_stats(se_csv_fpath)
        # simeng_stats = self.parse_output(stdout)
        sst_stats = {"l1_hits": 0, "l1_misses": 0, "l2_hits": 0, "l2_misses": 0}
        if not self.vanilla_se:
            sst_stats = self.parse_sim_sst_stats(benchmark["sst_stat_file_path"])
        csv_row = {}
        csv_row["matrix_size"] = benchmark["matrix_size"]
        csv_row["itrs"] = benchmark["itrs"]
        if category == "sve":
            csv_row["sve_length"] = benchmark["sve_length"]
            csv_row["l1_core_bw"] = benchmark["l1_core_bw"]
        elif category == "sme":
            csv_row["sme_length"] = benchmark["sme_svl"]
            csv_row["l1_core_bw"] = benchmark["l1_core_bw"]
        return {**csv_row, **{**sst_stats, **se_csv_stats}}

    def generate_stat_file_from_df(self, df, stat_fpath):
        df.to_csv(stat_fpath, index=False)
        print(df.iloc[[-1]])
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
        roi_cycles_reg = re.compile("Done in (-?[0-9]+) cycles!")
        lines = output.splitlines()
        parsed_output = {"roi_cycles": -1}
        for line in lines:
            if roi_search := re.search(roi_cycles_reg, line):
                parsed_output["roi_cycles"] = int(roi_search.group(1))
        return parsed_output

    def parse_sim_sst_stats(self, sst_csv_fpath):
        csv_pd = pd.read_csv(sst_csv_fpath)
        if self.json_cfg["print_sim_output"]:
            print(csv_pd)
        l1h, l1m, l2h, l2m = csv_pd[" Sum.u64"]
        parsed_csv = {
            "l1_hits": int(l1h),
            "l1_misses": int(l1m),
            "l2_hits": int(l2h),
            "l2_misses": int(l2m),
        }
        return parsed_csv

    def get_simeng_csv_stats(self, path):
        csv_pd = csv.DictReader(open(path, "r"))
        return list(csv_pd).pop()

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

    def run_sst_benchmark(self, benchmark, total_count, current_count):
        desc = benchmark["desc"]
        spinner_text = "[%s/%s] %s" % (
            current_count,
            total_count,
            desc,
        )
        subproc_args = benchmark["sst_cli_args"]
        output_fpath = benchmark["output_path"]
        self.start_benchmark_spinner(spinner_text)
        cmd = self.generate_sst_subprocess_cmd(subproc_args)
        stdout = self.run_subprocess(cmd)
        if self.json_cfg["print_sim_output"]:
            print(stdout)
        ofile = open(output_fpath, "w")
        ofile.write(stdout)
        ofile.close()
        self.stop_and_persist_benchmark_spinner()
        return stdout

    def run_se_benchmark(self, benchmark, total_count, current_count):
        desc = benchmark["desc"]
        spinner_text = "[%s/%s] %s" % (
            current_count,
            total_count,
            desc,
        )
        output_fpath = benchmark["output_path"]
        self.start_benchmark_spinner(spinner_text)
        stdout = self.run_subprocess(benchmark["simeng_cli_args"])
        if self.json_cfg["print_sim_output"]:
            print(stdout)
        ofile = open(output_fpath, "w")
        ofile.write(stdout)
        ofile.close()
        self.stop_and_persist_benchmark_spinner()
        return stdout

    def get_header(self, btype):
        if btype == "neon":
            return neon_header
        elif btype == "sve":
            return sve_header
        else:
            return sme_header

    def run_benchmark_suite(self, btype):
        suite = self.benchmarks[btype]
        file_benchmarks = suite["file_benchmarks"]
        total_benchmark_count = suite["count"]
        completed_count = 0
        self.print_benchmark_category(btype, total_benchmark_count)
        for file_benchmark in file_benchmarks:
            output_stat_file_path = file_benchmark["output_stat_file_path"]
            row_df = pd.DataFrame(columns=self.get_header(btype))
            current_count = completed_count
            for benchmark in file_benchmark["benchmarks"]:
                try:
                    current_count += 1
                    stdout = ""
                    self.generate_yaml_config(btype, benchmark)
                    if self.vanilla_se:
                        stdout = self.run_se_benchmark(
                            benchmark,
                            total_benchmark_count,
                            current_count,
                        )
                    else:
                        stdout = stdout = self.run_sst_benchmark(
                            benchmark, total_benchmark_count, current_count
                        )
                    csv_row = self.generate_csv_row(btype, benchmark, stdout)
                    row_df = pd.concat(
                        [row_df, pd.DataFrame([csv_row])], ignore_index=True
                    )
                    self.generate_stat_file_from_df(row_df, output_stat_file_path)
                except KeyboardInterrupt:
                    self.exit_benchmark_on_keyboard_interrupt(
                        row_df, output_stat_file_path
                    )
                else:
                    completed_count = current_count

    def run(self):
        self.run_benchmark_suite("neon")
        self.run_benchmark_suite("sve")
        self.run_benchmark_suite("sme")


ModsimBenchmarks().run()
