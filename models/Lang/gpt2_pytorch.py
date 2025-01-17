import os, argparse
from typing import Optional, Union
import tiktoken

import torch
from torch import nn
import torch.nn.functional as F

from utils import fetch, getenv, colored

MAX_CONTEXT = getenv("MAX_CONTEXT", 128)
HALF = getenv("HALF")

class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.c_attn = nn.Linear(dim, 3*dim, bias=True)
        self.c_proj = nn.Linear(dim, dim, bias=True)
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads

    def forward(self, x:torch.tensor, start_pos:torch.tensor, mask:Optional[torch.tensor]) -> torch.tensor:
        if mask is not None or start_pos.item() == 0:
            start_pos = start_pos.item()
        
        if HALF: x = x.to(torch.float16)
        xqkv = self.c_attn(x)

        bsz, seqlen, _ = xqkv.shape
        xq = xqkv[:, :, :self.dim].reshape(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xqkv[:, :, self.dim:2*self.dim].reshape(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xqkv[:, :, 2*self.dim:].reshape(bsz, seqlen, self.n_heads, self.head_dim)


        bsz, seqlen, _, _ = xq.shape

        if not hasattr(self, "cache_kv"):
           self.cache_kv = torch.zeros((2, bsz, MAX_CONTEXT, self.n_heads, self.head_dim), dtype=x.dtype, device=x.device).contiguous()
        
        self.cache_kv[0, :, start_pos:start_pos+seqlen, :, :] = xk
        self.cache_kv[1, :, start_pos:start_pos+seqlen, :, :] = xv

        if start_pos > 0:
          keys = self.cache_kv[0][:, 0:start_pos+seqlen, :, :]
          values = self.cache_kv[1][:, 0:start_pos+seqlen, :, :]
        else:
            keys = xk
            values = xv
        
        xq, keys, values = xq.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)

        return self.c_proj(F.scaled_dot_product_attention(xq, keys, values, attn_mask=mask).transpose(1,2).reshape(bsz, seqlen, self.dim))
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.c_fc = nn.Linear(dim, hidden_dim, bias=True)
        self.c_proj = nn.Linear(hidden_dim, dim, bias=True)
    
    def forward(self, x:torch.tensor)->torch.tensor:
        return self.c_proj(F.gelu(self.c_fc(x)))
    

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, norm_eps):
        super().__init__()
        self.attn = Attention(dim, n_heads)
        self.mlp = FeedForward(dim, 4*dim)
        self.ln_1 = nn.LayerNorm(dim, norm_eps)
        self.ln_2 = nn.LayerNorm(dim, norm_eps)
    
    def forward(self, x:torch.tensor, start_pos: torch.tensor, mask:Optional[torch.tensor]):
        h = x + self.attn(self.ln_1(x), start_pos, mask).to(torch.float32)
        return (h + self.mlp(self.ln_2(h)))


class Transformer(nn.Module):
    def __init__(self, dim, n_heads, n_layers, norm_eps, vocab_size, max_seq_len=1024):
      super().__init__()

      self.vocab_size = vocab_size
      self.wte =nn.Embedding(vocab_size, dim)
      self.wpe = nn.Embedding(max_seq_len, dim)
      self.h = nn.ModuleList([TransformerBlock(dim, n_heads, norm_eps) for _ in range(n_layers)])
      self.ln_f = nn.LayerNorm(dim, norm_eps)
      self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens: torch.tensor, start_pos: torch.tensor, temperature: float=0.0):
      if not hasattr(self, 'allpos'): 
         self.allpos = torch.tensor.arange(0, MAX_CONTEXT).reshape(1, -1)
      
      if tokens.dim() == 2:
        seqlen = tokens.shape[1]
        tok_emb = self.wte(tokens)
      elif tokens.dim() == 1:
         seqlen = 1
         tok_emb = self.wte(tokens.unsqueeze(0))

      pos_emb = self.wpe(self.allpos[:, start_pos:start_pos+seqlen])
      h = tok_emb + pos_emb

      if HALF: 
        h = h.to(torch.float16)


      if seqlen > 1:
          start_pos_val = start_pos.item()
          mask = torch.full((1, 1, seqlen, start_pos_val + seqlen),
                            float('-inf'),
                            dtype=torch.float32,
                            device=tokens.device)
          mask = mask.triu(start_pos_val + 1)
      else:
          mask = None


      for hi in self.h: 
         h = hi(h, start_pos, mask)

      logits = self.lm_head(self.ln_f(h))

      if logits.shape[1] == 0:
        # special case for empty prompt
        logits = torch.ones((logits.shape[0], self.vocab_size), dtype=logits.dtype, device=logits.device)
      else:
        logits = logits[:, -1, :]

      if temperature < 1e-6:
        ret = logits.argmax(-1)
      else:
        probs = F.Softmax((logits / temperature), dim=-1)
        ret = torch.multinomial(probs, num_samples=1)
        ret = ret.squeeze(-1)
      return ret
    
VOCAB_SIZE = 50257
MODEL_PARAMS = {
  'gpt2':         dict(n_layers=12, n_heads=12, dim=768, norm_eps=1e-5, vocab_size=VOCAB_SIZE),   # 124M params
  'gpt2-medium':  dict(n_layers=24, n_heads=16, dim=1024, norm_eps=1e-5, vocab_size=VOCAB_SIZE),  # 350M params
  'gpt2-large':   dict(n_layers=36, n_heads=20, dim=1280, norm_eps=1e-5, vocab_size=VOCAB_SIZE),  # 774M params
  'gpt2-xl':      dict(n_layers=48, n_heads=25, dim=1600, norm_eps=1e-5, vocab_size=VOCAB_SIZE),  # 1558M params
}


# class GPT2:
# How to get gguf file without using Tinygrad function ?? 
class GPT2:
  @staticmethod
  def build(model_size="gpt2"):
    tokenizer = tiktoken.get_encoding("gpt2")

    model = Transformer(**MODEL_PARAMS[model_size])
    weights = torch.load(fetch(f'https://huggingface.co/{model_size}/resolve/main/pytorch_model.bin'))
    transposed = ('attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight')

    for k in weights:
      if k.endswith(transposed):
        weights[k] = weights[k].T
    
    weights['lm_head.weight'] = weights['wte.weight']
    model.load_state_dict(weights)

    if HALF:
      for l in model.state_dict().values():
        l = l.to(torch.float16)
    
    return GPT2(model, tokenizer)
  
  def __init__(self, model, tokenizer):
    self.model = model
    self.tokenizer = tokenizer

  def generate(self, prompt:str, max_length:int, temperature:float, timing:bool=False, batch_size:int=1):
    prompt_tokens = self.tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    toks = [prompt_tokens[:] for _ in range(batch_size)]
    start_pos = 0
    
    for _ in range(max_length):
      if batch_size == 1 and len(toks[0][start_pos:]) == 1:
        tokens = torch.tensor([x[start_pos:] for x in toks], dtype=torch.long)
      else:
        tokens = torch.tensor([x[start_pos:] for x in toks])
      tok = self.model(tokens, torch.tensor(start_pos), temperature).tolist()
      start_pos = len(toks[0])
      for i,t in enumerate(tok): toks[i].append(t)
    return [self.tokenizer.decode(x) for x in toks]


if __name__ == "__main__":
  
  default_prompt = "What is the answer to life, the universe, and everything?"

  parser = argparse.ArgumentParser(description='Run GPT2 in tinygrad', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--prompt', type=str, default=default_prompt, help="Phrase to start with")
  parser.add_argument('--count', type=int, default=100, help="Max number of tokens to generate")
  parser.add_argument('--temperature', type=float, default=0.8, help="Temperature in the softmax")
  parser.add_argument('--model_size', type=str, default="gpt2-medium", help="Size of model to use [gpt2, gpt2-medium, gpt2-large, gpt2-xl]")
  parser.add_argument('--timing', action='store_true', help="Print timing per token")
  parser.add_argument('--seed', type=int, help="Set the random seed")
  parser.add_argument('--batch_size', type=int, default=1, help="Set the input batch size")
  parser.add_argument('--noshow', action='store_true', help="Don't show the output")
  args = parser.parse_args()

  if args.seed is not None:
     torch.manual_seed(args.seed)
    
  print(f"Using {args.model_size}")
  gpt2 = GPT2.build(args.model_size)

  texts = gpt2.generate(args.prompt, args.count, args.temperature, timing=args.timing, batch_size=args.batch_size)
  if not args.noshow:
    print('Generating text...')
    if len(texts) == 1: 
       print(texts[0])
    else:
      for i,text in enumerate(texts): 
        print(colored(f"Response {i}:", "green"), text)

    if args.temperature == 0 and args.model_size == "gpt2-medium" and args.count == 10:
      expected = {
        default_prompt: "What is the answer to life, the universe, and everything?\n\nThe answer is that we are all one",
        "Hello.": "Hello. I'm a little late to the party, but",
      }
      try:
        assert texts[0] == expected[args.prompt]
        print(colored("output validated", "green"))
      except KeyError:
        pass

