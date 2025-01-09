import numpy as np

import torch
from torch.nn import nn
import torch.nn.functional as F

import ast
from PIL import Image

from ../utils import fetch, getenv


class TransformerBlock(nn.Module):
  def __init__(self, embed_dim, num_heads, ff_dim, prenorm=False, act=lambda x: x.relu(), dropout=0.1):
    super(self ).__init__()

    assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

    self.num_heads = num_heads
    self.head_size = embed_dim // num_heads
    self.prenorm, self.act = prenorm, act
    self.dropout = nn.Dropout()

    self.query = (torch.uniform_(embed_dim, embed_dim), torch.zeros(embed_dim))
    self.key = (torch.uniform_(embed_dim, embed_dim), torch.zeros(embed_dim))
    self.value = (torch.uniform_(embed_dim, embed_dim), torch.zeros(embed_dim))

    self.out = (torch.uniform_(embed_dim, embed_dim), torch.zeros(embed_dim))

    self.ff1 = (torch.uniform_(embed_dim, ff_dim), torch.zeros(ff_dim))
    self.ff2 = (torch.uniform_(ff_dim, embed_dim), torch.zeros(embed_dim))

    self.ln1 = (torch.ones(embed_dim), torch.zeros(embed_dim))
    self.ln2 = (torch.ones(embed_dim), torch.zeros(embed_dim))

  def attn(self, x):
    # x: (bs, time, embed_dim) -> (bs, time, embed_dim)
    query, key, value = [x.linear(*y).reshape(shape=(x.shape[0], -1, self.num_heads, self.head_size)).transpose(1,2) for y in [self.query, self.key, self.value]]
    attention = F.scaled_dot_product_attention(query, key, value).transpose(1,2)
    return attention.reshape(shape=(x.shape[0], -1, self.num_heads * self.head_size)).linear(*self.out)

  def forward(self, x):
    if self.prenorm:
      x = nn.LayerNorm(x)
      x = nn.Linear(*self.ln1(x))
      x = self.attn(x)
      x = x + self.dropout(x)
      x = nn.LayerNorm(x)
      x = nn.Linear(*self.ln2(x))
      x = nn.Linear(*self.ff1(x))
      x = self.act(x)
      x = nn.Linear(*self.ff2(x))
      x = x + self.dropout(x)
    else:
      x = self.attn(x)
      x = x + self.dropout(x)
      x = nn.LayerNorm(x)
      x = nn.Linear(*self.ln1(x))
      x = nn.Linear(*self.ff1(x))
      x = nn.Linear(*self.ff2(x))
      x = self.act(x)
      x = x + self.dropout(x)
      x = nn.LayerNomr(nn.Linear(*self.ln2(x)))


class ViT(nn.Module):
    def __init__(self, layers=12, embed_dim=192, num_heads=3):
        super().__init__()
      

        # Patch embedding
        self.embedding = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
        self.embed_dim = embed_dim
        
        # Class token and position embedding
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, 197, embed_dim))

        
        self.tbs = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=embed_dim*4,
            prenorm=True, act=F.gelu)
            for _ in range(layers)])
        
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 1000)

    def patch_embed(self, x):
        x = self.embedding(x)
        x = x.flatten(2).transpose(1, 2)
        return x

    def forward(self, x):

      x = self.patch_embed(x)

      ce = self.cls.expand(x.shape[0], -1, -1)
      x= torch.cat((ce, x), dim=1)

      x = x + self.pos_embed

      for block in self.tbs:
          x = block(x)

      x = self.norm(x)
      x = self.head(x[:, 0])
      return x
    
    def load_from_pretrained(m):
      # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
      if m.embed_dim == 192:
        url = "https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz"
      elif m.embed_dim == 768:
        url = "https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz"
      else:
        raise Exception("no pretrained weights for configuration")
      dat = np.load(fetch(url))

      #for x in dat.keys():
      #  print(x, dat[x].shape, dat[x].dtype)

      m.embedding[0] = np.transpose(dat['embedding/kernel'], (3,2,0,1))
      m.embedding[1] = dat['embedding/bias']

      m.cls = dat['cls']

      m.head[0] = dat['head/kernel']
      m.head[1] = dat['head/bias']

      m.pos_embedding = dat['Transformer/posembed_input/pos_embedding']
      m.encoder_norm[0] = dat['Transformer/encoder_norm/scale']
      m.encoder_norm[1] = dat['Transformer/encoder_norm/bias']

      for i in range(12):
        m.tbs[i].query[0] = dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/query/kernel'].reshape(m.embed_dim, m.embed_dim)
        m.tbs[i].query[1] = dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/query/bias'].reshape(m.embed_dim)
        m.tbs[i].key[0] = dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/key/kernel'].reshape(m.embed_dim, m.embed_dim)
        m.tbs[i].key[1] = dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/key/bias'].reshape(m.embed_dim)
        m.tbs[i].value[0] = dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/value/kernel'].reshape(m.embed_dim, m.embed_dim)
        m.tbs[i].value[1] = dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/value/bias'].reshape(m.embed_dim)
        m.tbs[i].out[0] = dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/out/kernel'].reshape(m.embed_dim, m.embed_dim)
        m.tbs[i].out[1] = dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/out/bias'].reshape(m.embed_dim)
        m.tbs[i].ff1[0] = dat[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_0/kernel']
        m.tbs[i].ff1[1] = dat[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_0/bias']
        m.tbs[i].ff2[0] = dat[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_1/kernel']
        m.tbs[i].ff2[1] = dat[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_1/bias']
        m.tbs[i].ln1[0] = dat[f'Transformer/encoderblock_{i}/LayerNorm_0/scale']
        m.tbs[i].ln1[1] = dat[f'Transformer/encoderblock_{i}/LayerNorm_0/bias']
        m.tbs[i].ln2[0] = dat[f'Transformer/encoderblock_{i}/LayerNorm_2/scale']
        m.tbs[i].ln2[1] = dat[f'Transformer/encoderblock_{i}/LayerNorm_2/bias']


#Example
if getenv("LARGE", 0) == 1:
  m = ViT(embed_dim=768, num_heads=12)
else:
  # tiny
  m = ViT(embed_dim=192, num_heads=3)
m.load_from_pretrained()
m.eval()

# category labels
lbls = ast.literal_eval(fetch("https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt").read_text())

#url = "https://upload.wikimedia.org/wikipedia/commons/4/41/Chicken.jpg"
url = "https://repository-images.githubusercontent.com/296744635/39ba6700-082d-11eb-98b8-cb29fb7369c0"

# junk
img = Image.open(fetch(url))
aspect_ratio = img.size[0] / img.size[1]
img = img.resize((int(224*max(aspect_ratio,1.0)), int(224*max(1.0/aspect_ratio,1.0))))
img = np.array(img)
y0,x0=(np.asarray(img.shape)[:2]-224)//2
img = img[y0:y0+224, x0:x0+224]
img = np.moveaxis(img, [2,0,1], [0,1,2])
img = img.astype(np.float32)[:3].reshape(1,3,224,224)
img /= 255.0
img -= 0.5
img /= 0.5

out = m.forward(torch.tensor(img))
outnp = out.numpy().ravel()
choice = outnp.argmax()
print(out.shape, choice, outnp[choice], lbls[choice])

