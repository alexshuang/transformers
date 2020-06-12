import argparse
import numpy as np
import pandas as pd


def gemm_stats(model_name:str, input_dir:str, do_train:bool=True):
    model_type = 'training' if do_train else 'inference'
    gpu_stats_fpath = f"{input_dir}/{model_name}_{model_type}_gpu_res.stats.csv"
    df = pd.read_csv(gpu_stats_fpath)

    gemm_idx = [o.startswith('Cijk_A') for o in df.Name.values]
    kernel_total_duration_ns = df['TotalDurationNs'].sum()
    gemm_total_duration_ns = df[gemm_idx]['TotalDurationNs'].sum()
    gemm_percentage = df[gemm_idx]['Percentage'].sum()

    df['GEMM_Percentage'] = np.nan
    df['GEMM_TotalDurationNs'] = np.nan
    df.iloc[0, df.columns.get_loc("GEMM_Percentage")] = df[gemm_idx]['Percentage'].sum()
    df.iloc[0, df.columns.get_loc("GEMM_TotalDurationNs")] = df[gemm_idx]['TotalDurationNs'].sum()
    df.to_csv(gpu_stats_fpath, index=False)

    torch_stats_fpath = f"{input_dir}/{model_name}_{model_type}_torch_res.csv"
    df = pd.read_csv(torch_stats_fpath)
    df['Kernel_Percentage'] = np.nan
    df.iloc[0, df.columns.get_loc("Kernel_Percentage")] = kernel_total_duration_ns / (df.elapsed.to_list()[0] * 1e9)
    df.to_csv(torch_stats_fpath, index=False)


def kernel_stats(model_name:str, input_dir:str, do_train:bool=True):
    model_type = 'training' if do_train else 'inference'
    kernel_stats_fpath = f"{input_dir}/{model_name}_{model_type}_gpu_res.csv"
    df = pd.read_csv(kernel_stats_fpath)
    df2 = df[410:]
    df2.to_csv(kernel_stats_fpath, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.description = "Benchmark Data Stats"
    parser.add_argument("-m", "--model_name", type=str, help='model name')
    parser.add_argument("-d", "--input_dir", type=str, help='path of *.csv')
    parser.add_argument("-t", "--do_train", action="store_true", help='run training')
    args = parser.parse_args()

    gemm_stats(**vars(args))


if __name__ == '__main__':
    main()
