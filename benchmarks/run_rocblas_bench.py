import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
import os
import re
from tqdm import tqdm

#idx = -1

#def prof_rocblas_bench(src_file:str, out_dir:str):
#    tmp_dir = "/tmp/rocblas_bench"
#    if os.path.exists(tmp_dir):
#        shutil.rmtree(tmp_dir)
#    os.makedirs(tmp_dir, exist_ok=True)
#
#    print("profiling rocblas_bench.csv ...")
#    res = []
#    with open(src_file) as fp:
#        for i, o in enumerate(tqdm(fp.readlines())):
#            s = o.split(' ')
##            s.insert(1, '-i 1')
##            s.insert(1, '-j 2')
#            tmp_file = "%s/%d.csv" % (tmp_dir, i)
#            cmd = "/opt/rocm/bin/rocprof -i input.txt --obj-tracking on -o %s %s" % (tmp_file, ' '.join(s))
#            print(cmd)
#            ret = os.popen(cmd).readlines()
#            print(ret)
#            import pdb; pdb.set_trace()
#            if ret == None:
#                print("ERROR: Fail to prof rocblas-bench at %d lines" % i)
#                exit(i)
#            res.extend([o for o in ret if re.match('[T|N],', o)])
#    
#    print("generating results ...")
#    with open(out_dir + '/rocblas_kernel.csv', 'w') as fp:
#        for o in res: fp.write(o)
#
#    print("%s is generated." % (out_dir + '/rocblas_kernel.csv'))
#
#    res = []
#    for i in range(len(open(src_file).readlines())):
#        res.extend(open(tmp_dir + "/%d.csv" % i).readlines()[idx])
#
#    with open(out_dir + '/rocblas_kernel.prof.csv', 'w') as fp:
#        for o in res: fp.write(o)
#
#    print("%s is generated." % (out_dir + '/rocblas_kernel.prof.csv'))


def prof_rocblas_bench(src_file:str, out_dir:str):
    out_file = out_dir + '/rocblas_bench.prof.csv'
    tmp_dir = "/tmp/rocblas_bench"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    print("profiling rocblas_bench.csv ...")
    with open(src_file) as fp:
        for i, o in enumerate(tqdm(fp.readlines())):
            tmp_file = "%s/%d.csv" % (tmp_dir, i)
            cmd = "/opt/rocm/bin/rocprof -i input.txt --obj-tracking on -o %s %s" % (tmp_file, o)
            ret = os.popen(cmd).readlines()
            if ret == None:
                print("ERROR: Fail to prof rocblas-bench at %d lines" % i)
                exit(ret)
    
    print("generating results ...")
    res = []
    for i in range(len(open(src_file).readlines())):
        res.extend(open(tmp_dir + "/%d.csv" % i).readlines()[-1])

    with open(out_file, 'w') as fp:
        for o in res: fp.write(o)

    print("%s is generated." % (out_file))


def main():
    parser = argparse.ArgumentParser()
    parser.description = "prof rocblas-bench"
    parser.add_argument("-f", "--src_file", type=str, help='file path of rocblas_bench.csv')
    parser.add_argument("-o", "--out_dir", type=str, help='output dir')
    args = parser.parse_args()
    prof_rocblas_bench(**vars(args))


if __name__ == '__main__':
    main()
