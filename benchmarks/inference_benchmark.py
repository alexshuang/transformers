# Copyright (C) 2020 THL A29 Limited, a Tencent company.
# All rights reserved.
# Licensed under the BSD 3-Clause License (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at
# https://opensource.org/licenses/BSD-3-Clause
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
# See the AUTHORS file for names of contributors.

import argparse
import benchmark_helper
import torch
import transformers

def benchmark_torch(model_id: str, seq_len: int, batch_size: int, num_iter: int, framework: str):
    if not torch.cuda.is_available():
        print("cuda is not available for torch")
        return

    test_device = torch.device('cuda:0')

    torch.set_grad_enabled(False)

    model = transformers.BertForPreTraining.from_pretrained(model_id)
    model.eval()

    model.to(test_device)

    cfg = model.config  # type: transformers.BertConfig
    input_ids = torch.randint(low=0,
                              high=cfg.vocab_size - 1,
                              size=(batch_size, seq_len),
                              dtype=torch.long,
                              device=test_device)
    benchmark_helper.run_model(lambda: model(input_ids), model_id, num_iter)


def main():
    parser = argparse.ArgumentParser()
    parser.description = "PyTorch Inference Benchmark"
    parser.add_argument("-m", "--model_id", type=str, help='model id')
    parser.add_argument("-sl", "--seq_len", type=int, help='sequnence length')
    parser.add_argument("-bs", "--batch_size", type=int, help='batch size')
    parser.add_argument("-f", "--framework", type=str, help='PyTorch or other')
    parser.add_argument("-n", "--num_iter", type=int, help='number of iteration')
    args = parser.parse_args()

    if args.framework == 'torch':
        benchmark_torch(**vars(args))
    else:
        raise RuntimeError(f"Not supportted framework {args['--framework']}")


if __name__ == '__main__':
    main()
