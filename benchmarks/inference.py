import torch
from transformers import AutoTokenizer, BertForPreTraining
import time

bs, seq_len = 12, 512
model_name = 'bert-large-uncased-whole-word-masking'

print("Preparing model and dataset...")
model = BertForPreTraining.from_pretrained(model_name).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)

s = torch.arange(2000, 2000 + seq_len)
s[0] = tokenizer.cls_token_id # add [CLS]
s[seq_len//2] = tokenizer.sep_token_id # add [SEP]
s[-1] = tokenizer.sep_token_id # add [SEP]
inputs = s.unsqueeze(0).repeat(bs, 1).cuda()
print("inputs size: {}".format(inputs.shape))

start = time.perf_counter()
mlm_logits, nsp_logits = model(inputs)
end = time.perf_counter()
print("output sizes: {}, {}".format(mlm_logits.shape, nsp_logits.shape))
print("elapsed time: %.3f Seconds" % (end - start))
