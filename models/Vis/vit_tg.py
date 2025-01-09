import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.helpers import fetch
from extra.models.transformer import TransformerBlock

import ast
from PIL import Image
from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv, fetch


# Model
class TransformerBlock:
  def __init__(self, embed_dim, num_heads, ff_dim, prenorm=False, act=lambda x: x.relu(), dropout=0.1):
    assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

    self.num_heads = num_heads
    self.head_size = embed_dim // num_heads
    self.prenorm, self.act = prenorm, act
    self.dropout = dropout

    self.query = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))
    self.key = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))
    self.value = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))

    self.out = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))

    self.ff1 = (Tensor.scaled_uniform(embed_dim, ff_dim), Tensor.zeros(ff_dim))
    self.ff2 = (Tensor.scaled_uniform(ff_dim, embed_dim), Tensor.zeros(embed_dim))

    self.ln1 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))
    self.ln2 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))

  def attn(self, x):
    # x: (bs, time, embed_dim) -> (bs, time, embed_dim)
    query, key, value = [x.linear(*y).reshape(shape=(x.shape[0], -1, self.num_heads, self.head_size)).transpose(1,2) for y in [self.query, self.key, self.value]]
    attention = Tensor.scaled_dot_product_attention(query, key, value).transpose(1,2)
    return attention.reshape(shape=(x.shape[0], -1, self.num_heads * self.head_size)).linear(*self.out)

  def __call__(self, x):
    if self.prenorm:
      x = x + self.attn(x.layernorm().linear(*self.ln1)).dropout(self.dropout)
      x = x + self.act(x.layernorm().linear(*self.ln2).linear(*self.ff1)).linear(*self.ff2).dropout(self.dropout)
    else:
      x = x + self.attn(x).dropout(self.dropout)
      x = x.layernorm().linear(*self.ln1)
      x = x + self.act(x.linear(*self.ff1)).linear(*self.ff2).dropout(self.dropout)
      x = x.layernorm().linear(*self.ln2)


class ViT:
  def __init__(self, layers=12, embed_dim=192, num_heads=3):
    self.embedding = (Tensor.uniform(embed_dim, 3, 16, 16), Tensor.zeros(embed_dim))
    self.embed_dim = embed_dim
    self.cls = Tensor.ones(1, 1, embed_dim)
    self.pos_embedding = Tensor.ones(1, 197, embed_dim)
    self.tbs = [
      TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=embed_dim*4,
        prenorm=True, act=lambda x: x.gelu())
      for i in range(layers)]
    self.encoder_norm = (Tensor.uniform(embed_dim), Tensor.zeros(embed_dim))
    self.head = (Tensor.uniform(embed_dim, 1000), Tensor.zeros(1000))

  def patch_embed(self, x):
    x = x.conv2d(*self.embedding, stride=16)
    x = x.reshape(shape=(x.shape[0], x.shape[1], -1)).permute(order=(0,2,1))
    return x

  def forward(self, x):
    ce = self.cls.add(Tensor.zeros(x.shape[0],1,1))
    pe = self.patch_embed(x)
    x = ce.cat(pe, dim=1)
    x = x.add(self.pos_embedding).sequential(self.tbs)
    x = x.layernorm().linear(*self.encoder_norm)
    return x[:, 0].linear(*self.head)

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

    m.embedding[0].assign(np.transpose(dat['embedding/kernel'], (3,2,0,1)))
    m.embedding[1].assign(dat['embedding/bias'])

    m.cls.assign(dat['cls'])

    m.head[0].assign(dat['head/kernel'])
    m.head[1].assign(dat['head/bias'])

    m.pos_embedding.assign(dat['Transformer/posembed_input/pos_embedding'])
    m.encoder_norm[0].assign(dat['Transformer/encoder_norm/scale'])
    m.encoder_norm[1].assign(dat['Transformer/encoder_norm/bias'])

    for i in range(12):
      m.tbs[i].query[0].assign(dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/query/kernel'].reshape(m.embed_dim, m.embed_dim))
      m.tbs[i].query[1].assign(dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/query/bias'].reshape(m.embed_dim))
      m.tbs[i].key[0].assign(dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/key/kernel'].reshape(m.embed_dim, m.embed_dim))
      m.tbs[i].key[1].assign(dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/key/bias'].reshape(m.embed_dim))
      m.tbs[i].value[0].assign(dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/value/kernel'].reshape(m.embed_dim, m.embed_dim))
      m.tbs[i].value[1].assign(dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/value/bias'].reshape(m.embed_dim))
      m.tbs[i].out[0].assign(dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/out/kernel'].reshape(m.embed_dim, m.embed_dim))
      m.tbs[i].out[1].assign(dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/out/bias'].reshape(m.embed_dim))
      m.tbs[i].ff1[0].assign(dat[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_0/kernel'])
      m.tbs[i].ff1[1].assign(dat[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_0/bias'])
      m.tbs[i].ff2[0].assign(dat[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_1/kernel'])
      m.tbs[i].ff2[1].assign(dat[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_1/bias'])
      m.tbs[i].ln1[0].assign(dat[f'Transformer/encoderblock_{i}/LayerNorm_0/scale'])
      m.tbs[i].ln1[1].assign(dat[f'Transformer/encoderblock_{i}/LayerNorm_0/bias'])
      m.tbs[i].ln2[0].assign(dat[f'Transformer/encoderblock_{i}/LayerNorm_2/scale'])
      m.tbs[i].ln2[1].assign(dat[f'Transformer/encoderblock_{i}/LayerNorm_2/bias'])


# Example
Tensor.training = False
if getenv("LARGE", 0) == 1:
  m = ViT(embed_dim=768, num_heads=12)
else:
  # tiny
  m = ViT(embed_dim=192, num_heads=3)
m.load_from_pretrained()

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

out = m.forward(Tensor(img))
outnp = out.numpy().ravel()
choice = outnp.argmax()
print(out.shape, choice, outnp[choice], lbls[choice])