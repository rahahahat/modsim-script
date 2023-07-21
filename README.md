## Requirements

- pip
- python

## Setup

- Create a python virtual environment by running the command: `<python|python3> -m venv .env`
  - This will create a python virtual environment of with the name `.env`
- Activate the virtual environment using the command: `source .env/bin/active`
- Install all dependencies using the command: `<pip|pip3> -r install requirements.txt`

## Running the script

- The script can be run using the command: `<python|python3> data_script.py --config=<path>`
  - `<path>` is the path to a JSON file with the following properties:
    - `bin_dir_path`: `<string> -- (Path to Binaries directory)`
    - `yaml_configs_dir_path`: `<string> -- (Path to the SimEng Yaml config files directory)`
    - `benchmarks`: `<Array(string)> -- (name of benchmark files to run)`
    - `sst_config_path`: `<string> -- (Path to sst config.py used for all benchmark runs)`
    - `sst_core_path`: `<string> -- (Path to the SST core installation)`
    - `matrix_sizes`: `<Array(int)> -- (Matrix sizes used for benchmarks)`
    - `vector_lengths`:`<Array(int)> -- (Vector lengths used for SVE and SME benchmarks)`
    - `print_sim_output`: `<boolean> --  (Print SimEng simulation output to stdout)`
- Statistics are generated per benchmark file in the stats directory in the root directory with the name format: `<benchmark_name>.csv`

#### Example `config.json`

```
{
  "bin_dir_path": "/home/Binaries",
  "yaml_configs_dir_path": "/home/yaml_configs",
  "benchmarks": ["gemm_fp32_neon", "gemm_fp64_sve128"],
  "sst_config_path": "modsim-a64fx-config.py",
  "sst_core_path": "/sst-core/bin",
  "matrix_sizes": [64, 256, 1024],
  "vector_lengths": [128, 256],
  "print_sim_output": false
}
## The above configuration options will do 9 simulation runs i.e.
gemm_fp32_neon inp=64 vl=512 itrs=100
gemm_fp32_neon inp=256 vl=512 itrs=100
gemm_fp32_neon inp=1024 vl=512 itrs=10

gemm_fp32_sve128 inp=64 vl=128 itrs=100
gemm_fp32_sve128 inp=64 vl=256 itrs=100
gemm_fp32_sve128 inp=256 vl=128 itrs=100
gemm_fp32_sve128 inp=256 vl=256 itrs=100
gemm_fp32_sve128 inp=1024 vl=128 itrs=10
gemm_fp32_sve128 inp=1024 vl=256 itrs=10
```
