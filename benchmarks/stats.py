import argparse
import pandas as pd


def apply_sum(df, new_col, src_col):


def gemm_stats(model_name:str, num_iter:int):
    df = pd.read_csv(f"{model_name}_inference_gpu_res.stats.csv")
    df 


def main():
    parser = argparse.ArgumentParser()
    parser.description = "Benchmark Data Stats"
    parser.add_argument("-m", "--model_name", type=str, help='model name')
    parser.add_argument("-n", "--num_iter", type=int, help='number of iteration')
    args = parser.parse_args()

    gemm_stats(**vars(args))


if __name__ == '__main__':
    main()
