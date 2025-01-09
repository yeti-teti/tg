import torch 
from torch import nn 
import torch.nn.functional as F

from typing import List, Optional, Union, Tuple, Dict
from abc import ABC, abstractmethod
from functools import lru_cache
from PIL import Image
import numpy as np
import re, gzip

import urllib.request

@lru_cache()
def default_bpe():
    return urllib.request.urlretrieve("https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz", "bpe_simple_vocab_16e6.txt.gz")


class Tokenizer:

    @staticmethod
    def get_pairs(word):
        return set(zip(word, word[1:]))
    @staticmethod
    def whitespace_clean(text):
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    @staticmethod
    def bytes_to_unicode():
        bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))
    
    class ClipTokenizer:
        def __init__(self):
            self.byte_encoder = Tokenizer.bytes_to_unicode()
            merges = gzip.open(default_bpe()).read().decode("utf-8").split('\n')
            merges = merges[1:49152-256-2+1]
            merges = [tuple(merge.split()) for merge in merges]
            vocab = list(Tokenizer.bytes_to_unicode().values())
            vocab = vocab + [v+'</w>' for v in vocab]
            for merge in merges:
                vocab.append(''.join(merge))
            vocab.extend(['<|startoftext|>', '<|endoftext|>'])
            self.encoder = dict(zip(vocab, range(len(vocab))))
            self.bpe_ranks = dict(zip(merges, range(len(merges))))
            self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
            self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[^\s]+""", re.IGNORECASE)
    
        def bpe(self, token):
            if token in self.cache:
                return self.cache[token]
            word = tuple(token[:-1]) + ( token[-1] + '</w>',)
            pairs = Tokenizer.get_pairs(word)

            if not pairs:
                return token+'</w>'

            while True:
                bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
                if bigram not in self.bpe_ranks:
                    break
                first, second = bigram
                new_word = []
                i = 0
                while i < len(word):
                    try:
                        j = word.index(first, i)
                        new_word.extend(word[i:j])
                        i = j
                    except Exception:
                        new_word.extend(word[i:])
                        break

                    if word[i] == first and i < len(word)-1 and word[i+1] == second:
                        new_word.append(first+second)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word = tuple(new_word)
                word = new_word
                if len(word) == 1:
                    break
                pairs = Tokenizer.get_pairs(word)
            word = ' '.join(word)
            self.cache[token] = word
            return word

        def encode(self, text:str, pad_with_zeros:bool=False) -> List[int]:
            bpe_tokens: List[int] = []
            text = Tokenizer.whitespace_clean(text.strip()).lower()
            for token in re.findall(self.pat, text):
                token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
                bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
            # Truncation, keeping two slots for start and end tokens.
            if len(bpe_tokens) > 75:
                bpe_tokens = bpe_tokens[:75]
            return [49406] + bpe_tokens + [49407] + ([0] if pad_with_zeros else [49407]) * (77 - len(bpe_tokens) - 2)

class Embedder(ABC):
  input_key: str
  @abstractmethod
  def __call__(self, x:Union[str,List[str],torch.Tensor]):
    pass


class Closed:

    class ClipMlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(768, 3072)
            self.fc2 = nn.Linear(3072, 768)
        
        def forward(self, h:torch.tensor):
            h = self.fc1(h)
            h = F.gelu(h)
            h = self.fc2(h)
            return h
        
    class ClipAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_dim = 768
            self.num_heads = 12
            self.head_dim = self.embed_dim // self.num_heads
            self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
            self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
            self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
            self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
        def forward(self, hidden_states, causal_attention_mask):
            bsz, tgt_len, embed_dim = hidden_states.shape
            q,k,v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
            q,k,v = [x.reshape(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2) for x in (q,k,v)]
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_attention_mask)
            return self.out_proj(attn_output.transpose(1, 2).reshape(bsz, tgt_len, embed_dim))
    
    class ClipEncoderLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Closed.ClipAttention()
            self.layer_norm1 = nn.LayerNorm(768)
            self.mlp = Closed.ClipMlp()
            self.layer_norm2 = nn.LayerNorm(768)

        def forward(self, hidden_states:torch.tensor, causal_attention_mask:torch.tensor):
            residual = hidden_states
            hidden_states = self.layer_norm1(hidden_states)
            hidden_states = self.self_attn(hidden_states, causal_attention_mask)
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self.layer_norm2(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
        
            return hidden_states
    
    class ClipTextEmbeddings:
        def __init__(self):
            self.token_embedding  = nn.Embedding(49408, 768)
            self.position_embedding = nn.Embedding(77, 768)

        def __call__(self, input_ids:torch.tensor, position_ids:torch.tensor):
            return self.token_embedding(input_ids) + self.position_embedding(position_ids)

    class ClipEncoder:
        def __init__(self, layer_count:int=12):
            self.layers = [Closed.ClipEncoderLayer() for _ in range(layer_count)]

        def __call__(self, x:torch.tensor, causal_attention_mask:torch.tensor, ret_layer_idx:Optional[int]=None):
            layers = self.layers if ret_layer_idx is None else self.layers[:ret_layer_idx]
            for l in layers:
                x = l(x, causal_attention_mask)
            return x

    class ClipTextTransformer:
        def __init__(self, ret_layer_idx:Optional[int]=None):
            self.embeddings       = Closed.ClipTextEmbeddings()
            self.encoder          = Closed.ClipEncoder()
            self.final_layer_norm = nn.LayerNorm(768)
            self.ret_layer_idx    = ret_layer_idx

        def __call__(self, input_ids:torch.tensor):
            x = self.embeddings(input_ids, torch.tensor.arange(input_ids.shape[1]).reshape(1, -1))
            x = self.encoder(x, torch.tensor.full((1, 1, 77, 77), float("-inf")).triu(1), self.ret_layer_idx)
            return self.final_layer_norm(x) if (self.ret_layer_idx is None) else x

    class ClipTextModel:
        def __init__(self, ret_layer_idx:Optional[int]):
            self.text_model = Closed.ClipTextTransformer(ret_layer_idx=ret_layer_idx)

class FrozenClosedClipEmbedder(Embedder):
  def __init__(self, ret_layer_idx:Optional[int]=None):
    self.tokenizer   = Tokenizer.ClipTokenizer()
    self.transformer = Closed.ClipTextModel(ret_layer_idx)
    self.input_key   = "txt"

  def __call__(self, texts:Union[str,List[str],torch.tensor]):
    if isinstance(texts, str): texts = [texts]
    assert isinstance(texts, (list,tuple)), f"expected list of strings, got {type(texts).__name__}"
    tokens = torch.tensor.cat(*[torch.tensor(self.tokenizer.encode(text)) for text in texts], dim=0)
    return self.transformer.text_model(tokens.reshape(len(texts),-1))



class Open:

  class MultiheadAttention(nn.Module):
    def __init__(self, dims:int, n_heads:int):
      super.__init__()
      self.dims    = dims
      self.n_heads = n_heads
      self.d_head  = self.dims // self.n_heads

      self.in_proj_bias   = nn.tensor.empty(3*dims)
      self.in_proj_weight = nn.tensor.empty(3*dims, dims)
      self.out_proj = nn.Linear(dims, dims)

    def forward(self, x:torch.tensor, attn_mask:Optional[torch.tensor]=None):
      T,B,C = x.shape

      proj = x.linear(self.in_proj_weight.T, self.in_proj_bias)
      proj = proj.unflatten(-1, (3,C)).unsqueeze(0).transpose(0, -2)

      q,k,v = [y.reshape(T, B*self.n_heads, self.d_head).transpose(0, 1).reshape(B, self.n_heads, T, self.d_head) for y in proj.chunk(3)]

      attn_output = torch.tensor.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
      attn_output = attn_output.permute(2, 0, 1, 3).reshape(T*B, C)

      attn_output = self.out_proj(attn_output)
      attn_output = attn_output.reshape(T, B, C)

      return attn_output

  class Mlp(nn.Module):
    def __init__(self, dims, hidden_dims):
      super.__init__()
      self.c_fc   = nn.Linear(dims, hidden_dims)
      self.c_proj = nn.Linear(hidden_dims, dims)

    def forward(self, x:torch.Tensor):
      x = F.gelu(self.c_fc(x))
      x = self.c_proj(x)
      return x


  class ResidualAttentionBlock(nn.Module):
    def __init__(self, dims:int, n_heads:int, mlp_ratio:float):
      super.__init__()
      self.ln_1 = nn.LayerNorm(dims)
      self.attn = Open.MultiheadAttention(dims, n_heads)

      self.ln_2 = nn.LayerNorm(dims)
      self.mlp  = Open.Mlp(dims, int(dims * mlp_ratio))

    def forward(self, x:torch.Tensor, attn_mask:Optional[torch.Tensor]=None, transpose:bool=False) -> torch.Tensor:
      q_x = self.ln_1(x)
      attn_out = self.attn(q_x.transpose(0, 1) if transpose else q_x, attn_mask=attn_mask)
      attn_out = attn_out.transpose(0, 1) if transpose else attn_out
      x = x + attn_out
      x = x + self.mlp(self.ln_2(x))
      return x


  class ClipTransformer(nn.Module):
    def __init__(self, dims:int, layers:int, n_heads:int, mlp_ratio:float=4.0):
      super.__init__()
      self.resblocks = [
        Open.ResidualAttentionBlock(dims, n_heads, mlp_ratio) for _ in range(layers)
      ]

    def forward(self, x:torch.Tensor, attn_mask:Optional[torch.Tensor]=None) -> torch.Tensor:
      for r in self.resblocks:
        x = r(x, attn_mask=attn_mask, transpose=True)
      return x

  class ClipTextTransformer(nn.Module):
    def __init__(self, width:int, n_heads:int, layers:int, vocab_size:int=49408, ctx_length:int=77):
      super.__init__()
      self.token_embedding = nn.Embedding(vocab_size, width)
      self.positional_embedding = torch.tensor.empty(ctx_length, width)
      self.transformer = Open.ClipTransformer(width, layers, n_heads)
      self.ln_final = nn.LayerNorm(width)
      self.text_projection = nn.tensor.empty(width, width)
      self.attn_mask = nn.tensor.full((77, 77), float("-inf")).triu(1)

    def forward(self, text:nn.tensor) -> nn.Tensor:
      seq_len = text.shape[1]

      x = self.token_embedding(text)
      x = x + self.positional_embedding[:seq_len]
      x = self.transformer(x, attn_mask=self.attn_mask)
      x = self.ln_final(x)

      pooled = x[:, text.argmax(dim=-1)] @ self.text_projection
      return pooled

  class ClipVisionTransformer(nn.Module):
    def __init__(self, width:int, layers:int, d_head:int, image_size:int, patch_size:int):
      super().__init__()
      grid_size = image_size // patch_size
      n_heads = width // d_head
      assert n_heads * d_head == width

      self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)

      self.class_embedding = nn.tensor.empty(width)
      self.positional_embedding = nn.tensor.empty(grid_size * grid_size + 1, width)
      self.transformer = Open.ClipTransformer(width, layers, n_heads)
      self.ln_pre  = nn.LayerNorm(width)
      self.ln_post = nn.LayerNorm(width)
      self.proj = nn.tensor.empty(width, 1024)

    def forward(self, x:nn.Tensor) -> nn.Tensor:
      x = self.conv1(x)
      x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
      x = self.class_embedding.reshape(1, 1, -1).expand(x.shape[0], 1, -1).cat(x, dim=1)
      x = x + self.positional_embedding

      x = self.ln_pre(x)
      x = self.transformer(x)
      x = self.ln_post(x)

      pooled = x[:, 0] @ self.proj
      return pooled



class FrozenOpenClipEmbedder(Embedder):
  def __init__(self, dims:int, n_heads:int, layers:int, return_pooled:bool, ln_penultimate:bool=False):
    self.tokenizer = Tokenizer.ClipTokenizer()
    self.model = Open.ClipTextTransformer(dims, n_heads, layers)
    self.return_pooled = return_pooled
    self.input_key = "txt"
    self.ln_penultimate = ln_penultimate

  def tokenize(self, text:str, device:Optional[str]=None) -> torch.tensor:
    return torch.tensor(self.tokenizer.encode(text, pad_with_zeros=True), dtype=torch.int64, device=device).reshape(1,-1)

  def text_transformer_forward(self, x:torch.Tensor, attn_mask:Optional[torch.Tensor]=None):
    for r in self.model.transformer.resblocks:
      x, penultimate = r(x, attn_mask=attn_mask), x
    return x.permute(1, 0, 2), penultimate.permute(1, 0, 2)

  def embed_tokens(self, tokens:torch.Tensor) -> Union[torch.Tensor,Tuple[torch.Tensor,...]]:
    x = self.model.token_embedding(tokens).add(self.model.positional_embedding).permute(1,0,2)
    x, penultimate = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)

    if self.ln_penultimate:
      penultimate = self.model.ln_final(penultimate)

    if self.return_pooled:
      x = self.model.ln_final(x)
      index = tokens.argmax(axis=-1).reshape(-1,1,1).expand(x.shape[0],1,x.shape[-1])
      pooled = x.gather(1, index).squeeze(1) @ self.model.text_projection
      return penultimate, pooled
    else:
      return penultimate

  def __call__(self, texts:Union[str,List[str],torch.Tensor]) -> Union[torch.Tensor,Tuple[torch.Tensor,...]]:
    if isinstance(texts, str): texts = [texts]
    assert isinstance(texts, (list,tuple)), f"expected list of strings, got {type(texts).__name__}"
    tokens = torch.tensor.cat(*[self.tokenize(text) for text in texts], dim=0)
    return self.embed_tokens(tokens)


clip_configs: Dict = {
  "ViT-H-14": {
    "dims": 1024,
    "vision_cfg": {
      "width": 1280,
      "layers": 32,
      "d_head": 80,
      "image_size": 224,
      "patch_size": 14,
    },
    "text_cfg": {
      "width": 1024,
      "n_heads": 16,
      "layers": 24,
      "ctx_length": 77,
      "vocab_size": 49408,
    },
    "return_pooled": False,
    "ln_penultimate": True,
  }
}

class OpenClipEncoder:
  def __init__(self, dims:int, text_cfg:Dict, vision_cfg:Dict, **_):
    self.visual = Open.ClipVisionTransformer(**vision_cfg)

    text = Open.ClipTextTransformer(**text_cfg)
    self.transformer = text.transformer
    self.token_embedding = text.token_embedding
    self.positional_embedding = text.positional_embedding
    self.ln_final = text.ln_final
    self.text_projection = text.text_projection

    self.attn_mask = torch.tensor.full((77, 77), float("-inf")).triu(1)
    self.mean = torch.tensor([0.48145466, 0.45782750, 0.40821073]).reshape(-1, 1, 1)
    self.std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(-1, 1, 1)

  def prepare_image(self, image:Image.Image) -> torch.Tensor:
    SIZE = 224
    w, h = image.size
    scale = min(SIZE / h, SIZE / w)
    image = image.resize((max(int(w*scale),SIZE),max(int(h*scale),SIZE)), Image.Resampling.BICUBIC)
    w, h = image.size
    if w > SIZE:
      left = (w - SIZE) // 2
      image = image.crop((left, left+SIZE, 0, SIZE))
    elif h > SIZE:
      top = (h - SIZE) // 2
      image = image.crop((0, SIZE, top, top+SIZE))

    x = torch.tensor(np.array(image.convert('RGB')), device=self.std.device)
    x = x.permute(2, 0, 1).type(torch.float32) / 255.0
    return (x - self.mean) / self.std

  def encode_tokens(self, tokens:torch.Tensor) -> torch.Tensor:
    x = self.token_embedding(tokens)
    x = x + self.positional_embedding
    x = self.transformer(x, attn_mask=self.attn_mask)
    x = self.ln_final(x)
    x = x[:, tokens.argmax(axis=-1)]
    x = x @ self.text_projection
    return x

  def get_clip_score(self, tokens:torch.Tensor, image:torch.Tensor) -> torch.Tensor:
    image_features: torch.Tensor = self.visual(image)
    image_features /= image_features.square().sum(-1, keepdim=True).sqrt() # Frobenius Norm

    text_features = self.encode_tokens(tokens)
    text_features /= text_features.square().sum(-1, keepdim=True).sqrt() # Frobenius Norm

    return (image_features * text_features).sum(axis=-1)