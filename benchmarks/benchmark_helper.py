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


def run_model(model, model_id, num_iter):
    import torch
    import contexttimer
    import pandas as pd
    # warm up
    model()
    model()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    with contexttimer.Timer() as t:
        for it in range(num_iter):
            model()

    end.record()
    torch.cuda.synchronize()
    torch_elapsed = start.elapsed_time(end)

    data = { 'model': [model_id], 'torch_elapsed': [torch_elapsed], 'num_iter': [num_iter] }
    df = pd.DataFrame(data)
    df.to_csv(f'{model_id}_inference_torch_res.csv', index=False)
