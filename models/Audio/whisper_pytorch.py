import sys, base64, multiprocessing, itertools, collections, os, io
from typing import Optional, Union, Literal, List

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import librosa
import requests


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state, n_head, kv_caching: Literal['cross', 'self']=None, max_self_attn_cache_len=None):
        super().__init__()
        assert n_state % n_head == 0, f"n_state {n_state} must be divisible by n_head {n_head}"

        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

        self.kv_caching = kv_caching
        self.max_self_attn_cache_len = max_self_attn_cache_len
        
    
    def forward(self, x:torch.Tensor, xa:Optional[torch.Tensor]=None, mask:Optional[torch.Tensor]=None, len: Optional[int]=None):
        if self.kv_caching == 'cross':
            if xa is not None:
                k, v = self.key(xa), self.value(xa)
                if not hasattr(self, 'cache_k'):
                    self.cache_k, self.cache_v = k, v
                else:
                    self.cache_k = k
                    self.cache_v = v 
            else:
                k, v = self.cache_k, self.cache_v
        else:
            k, v = self.key(x), self.value(x)
            if self.kv_caching == 'self':
                if not hasattr(self, 'cache_k'):
                    self.cache_k = torch.zeros(x.shape[0], self.max_self_attn_cache_len, x.shape[2], device=x.device)
                    self.cache_v = torch.zeros(x.shape[0], self.max_self_attn_cache_len, x.shape[2], device=x.device)
                k = torch.cat([self.cache_k[:, :len], k], dim=1)
                v = torch.cat([self.cache_v[:, :len], v], dim=1)
                padding = (0, 0, 0, self.max_self_attn_cache_len - len - x.shape[1])
                self.cache_k = F.pad(k, padding).contiguous()
                self.cache_v = F.pad(v, padding).contiguous()
            
        q = self.query(x)
        n_ctx = q.shape[1]
        assert(q.shape[-1] == k.shape[-1] == v.shape[-1])
        head_dim = q.shape[-1] // self.n_head
        q = q.reshape(q.shape[0], q.shape[1], self.n_head, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(k.shape[0], k.shape[1], self.n_head, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(v.shape[0], v.shape[1], self.n_head, head_dim).permute(0, 2, 1, 3)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, :n_ctx, :n_ctx] if mask is not None else None)
        wv = attn.permute(0, 2, 1, 3).flatten(start_dim=2)
        return self.out(wv)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state, n_head, is_decoder_block=False, max_self_attn_cache_len=None):
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head, kv_caching='self' if is_decoder_block else None, max_self_attn_cache_len=max_self_attn_cache_len)
        self.attn_ln = nn.LayerNorm(n_state)

        self.cross_attn = MultiHeadAttention(n_state, n_head, kv_caching='cross') if is_decoder_block else None
        self.cross_attn_ln = nn.LayerNorm(n_state) if is_decoder_block else None

        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_state*4), 
            nn.GELU(), 
            nn.Linear(n_state*4, n_state)
        )
        self.mlp_ln = nn.LayerNorm(n_state)
    
    def forward(self, x, xa=None, mask=None, len: Optional[int]=None):
        x = x + self.attn(self.attn_ln(x), mask=mask, len=len)
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa)
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
  def __init__(self, n_mels, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer, **_):
    super().__init__()
    self.conv1 = nn.Conv1d(n_mels, n_audio_state, kernel_size=3, padding=1)
    self.conv2 = nn.Conv1d(n_audio_state, n_audio_state, kernel_size=3, stride=2, padding=1)
    modules =  [ResidualAttentionBlock(n_audio_state, n_audio_head) for _ in range(n_audio_layer)]
    self.blocks = nn.Sequential(*modules)
    self.ln_post = nn.LayerNorm(n_audio_state)
    self.positional_embedding = torch.empty(n_audio_ctx, n_audio_state)

  def forward(self, x):
    x = F.gelu(self.conv1(x))
    x = F.gelu(self.conv2(x))
    x = x.permute(0, 2, 1)
    x = x + self.positional_embedding[:x.shape[1]]
    x = self.blocks(x)
    x = self.ln_post(x)
    return x

class TextDecoder(nn.Module):
  def __init__(self, n_vocab, n_text_ctx, n_text_state, n_text_head, n_text_layer, **_):
    super().__init__()
    self.max_tokens_to_sample = n_text_ctx // 2
    self.max_self_attn_cache_len = self.max_tokens_to_sample * 2 + 5  # roughly prompt + start toks + max_tokens_to_sample

    self.token_embedding = nn.Embedding(n_vocab, n_text_state)
    self.positional_embedding = torch.empty(n_text_ctx, n_text_state)
    modules = [ResidualAttentionBlock(n_text_state, n_text_head, is_decoder_block=True, max_self_attn_cache_len=self.max_self_attn_cache_len) for _ in range(n_text_layer)]
    self.blocks = nn.Sequential(*modules)
    self.ln = nn.LayerNorm(n_text_state)
    self.mask = torch.full((n_text_ctx, n_text_ctx), -np.inf).triu(1)

  def forward(self, x:torch.Tensor, pos:Union[Literal[0]], encoded_audio:torch.Tensor):
    seqlen = x.shape[-1]
    x = self.token_embedding(x) + self.positional_embedding[pos:pos+seqlen]
    for block in self.blocks: x = block(x, xa=encoded_audio, mask=self.mask, len=pos)
    return self.output_tok(x)

  def output_tok(self, x):
    return (self.ln(x) @ self.token_embedding.weight.T)

class Whisper(nn.Module):
  def __init__(self, dims, batch_size=1):
    super().__init__()
    self.encoder = AudioEncoder(**dims)
    self.decoder = TextDecoder(**dims)
    self.is_multilingual = dims["n_vocab"] == 51865
    self.batch_size = batch_size

    
RATE = 16000
SEGMENT_SECONDS=30
SAMPLES_PER_SEGMENT = RATE * SEGMENT_SECONDS # 480000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
FRAMES_PER_SEGMENT = SAMPLES_PER_SEGMENT // HOP_LENGTH # 3000

def prep_audio(waveforms: List[np.ndarray], batch_size: int, truncate=False) -> np.ndarray:
  def pad_or_trim(arr, target_len):
    curr_len = len(arr)
    if curr_len == target_len:
      return arr
    elif curr_len < target_len:
      return np.pad(arr, (0, target_len - curr_len), 'constant')
    else:
      return arr[:target_len]

  max_len = SAMPLES_PER_SEGMENT if truncate else max(len(wav) for wav in waveforms)
  if (r := max_len % SAMPLES_PER_SEGMENT) > 0: max_len += SAMPLES_PER_SEGMENT - r
  waveforms = np.array(list(map(lambda w: pad_or_trim(w, max_len), waveforms)))
  assert waveforms.shape[0] <= batch_size
  if waveforms.shape[0] < batch_size:
    # we could have a symbolic batch_size dim instead of manually padding here if conv/layernorm supported symbolic shapes
    waveforms = np.pad(waveforms, pad_width=((0, batch_size - waveforms.shape[0]), (0, 0)))

  stft = librosa.stft(waveforms, n_fft=N_FFT, hop_length=HOP_LENGTH, window='hann', dtype=np.csingle)
  magnitudes = np.absolute(stft[..., :-1]) ** 2
  mel_spec = librosa.filters.mel(sr=RATE, n_fft=N_FFT, n_mels=N_MELS) @ magnitudes

  log_spec = np.log10(np.clip(mel_spec, 1e-10, None))
  log_spec = np.maximum(log_spec, log_spec.max((1,2), keepdims=True) - 8.0)
  log_spec = (log_spec + 4.0) / 4.0

  return log_spec



LANGUAGES = {
  "en": "english", "zh": "chinese", "de": "german", "es": "spanish", "ru": "russian", "ko": "korean", "fr": "french", "ja": "japanese", "pt": "portuguese", "tr": "turkish",
  "pl": "polish", "ca": "catalan", "nl": "dutch", "ar": "arabic", "sv": "swedish", "it": "italian", "id": "indonesian", "hi": "hindi", "fi": "finnish", "vi": "vietnamese",
  "he": "hebrew", "uk": "ukrainian", "el": "greek", "ms": "malay", "cs": "czech", "ro": "romanian", "da": "danish", "hu": "hungarian", "ta": "tamil", "no": "norwegian",
  "th": "thai", "ur": "urdu", "hr": "croatian", "bg": "bulgarian", "lt": "lithuanian", "la": "latin", "mi": "maori", "ml": "malayalam", "cy": "welsh", "sk": "slovak", "te": "telugu",
  "fa": "persian", "lv": "latvian", "bn": "bengali", "sr": "serbian", "az": "azerbaijani", "sl": "slovenian", "kn": "kannada", "et": "estonian", "mk": "macedonian",
  "br": "breton", "eu": "basque", "is": "icelandic", "hy": "armenian", "ne": "nepali", "mn": "mongolian", "bs": "bosnian", "kk": "kazakh", "sq": "albanian", "sw": "swahili",
  "gl": "galician", "mr": "marathi", "pa": "punjabi", "si": "sinhala", "km": "khmer", "sn": "shona", "yo": "yoruba", "so": "somali", "af": "afrikaans", "oc": "occitan", "ka": "georgian",
  "be": "belarusian", "tg": "tajik", "sd": "sindhi", "gu": "gujarati", "am": "amharic", "yi": "yiddish", "lo": "lao", "uz": "uzbek", "fo": "faroese", "ht": "haitian creole",
  "ps": "pashto", "tk": "turkmen", "nn": "nynorsk", "mt": "maltese", "sa": "sanskrit", "lb": "luxembourgish", "my": "myanmar", "bo": "tibetan", "tl": "tagalog", "mg": "malagasy",
  "as": "assamese", "tt": "tatar", "haw": "hawaiian", "ln": "lingala", "ha": "hausa", "ba": "bashkir", "jw": "javanese", "su": "sundanese",
}

# def get_encoding(encoding_name):
#   f = requests.get("https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/{encoding_name}.tiktoken").text.splitlines()
#   ranks = {base64.b64decode(token): int(rank) for token, rank in (line.split() for line in f if line)}
#   n_vocab = len(ranks)
#   specials = [
#     "<|endoftext|>",
#     "<|startoftranscript|>",
#     *[f"<|{lang}|>" for lang in LANGUAGES.keys()],
#     "<|translate|>",
#     "<|transcribe|>",
#     "<|startoflm|>",
#     "<|startofprev|>",
#     "<|nospeech|>",
#     "<|notimestamps|>",
#     *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
#   ]
#   special_tokens = dict(zip(specials, itertools.count(n_vocab)))
#   n_vocab += len(specials)
#   import tiktoken
#   return tiktoken.Encoding(
#     name=encoding_name,
#     explicit_n_vocab=n_vocab,
#     pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
#     mergeable_ranks=ranks,
#     special_tokens=special_tokens)

def get_encoding(encoding_name):
    response = requests.get(f"https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/{encoding_name}.tiktoken")
    response.raise_for_status()  # Ensure the request was successful
    f = response.text.splitlines()

    ranks = {}
    for line in f:
        parts = line.split()
        if len(parts) == 2:  # Ensure exactly two parts
            try:
                token = base64.b64decode(parts[0])
                rank = int(parts[1])
                ranks[token] = rank
            except (ValueError, TypeError) as e:
                print(f"Skipping invalid line: {line} - Error: {e}")
        else:
            print(f"Skipping invalid line: {line}")

    n_vocab = len(ranks)
    specials = [
        "<|endoftext|>",
        "<|startoftranscript|>",
        *[f"<|{lang}|>" for lang in LANGUAGES.keys()],
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
    ]
    special_tokens = dict(zip(specials, itertools.count(n_vocab)))
    n_vocab += len(specials)

    import tiktoken
    return tiktoken.Encoding(
        name=encoding_name,
        explicit_n_vocab=n_vocab,
        pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        mergeable_ranks=ranks,
        special_tokens=special_tokens,
    )


MODEL_URLS = {
  "tiny.en": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
  "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
  "base.en": "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
  "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
  "small.en": "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
  "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
  "medium.en": "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
  "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
  "large-v1": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt",
  "large-v2": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
  "large": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
}
def init_whisper(model_name="tiny.en", batch_size=1):
  assert MODEL_URLS[model_name] is not None

  filename = requests.get(MODEL_URLS[model_name])
  state = torch.load(io.BytesIO(filename.content))
  model = Whisper(state['dims'], batch_size)
  model.load_state_dict(state, strict=False)
  enc = get_encoding("multilingual" if model.is_multilingual else "gpt2")
  return model, enc

def load_file_waveform(filename):
  waveform, _ = librosa.load(filename, sr=RATE)
  return waveform

def transcribe_file(model, enc, filename):
  return transcribe_waveform(model, enc, [load_file_waveform(filename)])

def transcribe_waveform(model: Whisper, enc, waveforms, truncate=False):

  log_spec = prep_audio(waveforms, model.batch_size, truncate)
  nsample = model.decoder.max_tokens_to_sample

  def inferloop(ctx: Union[np.ndarray, List[np.ndarray]], encoded_audio):
    pos, next_tokens = 0, ctx
    for i in range((nsample-len(start_tokens))*2):
      next_tokens = model.decoder(torch.tensor(next_tokens), pos, encoded_audio)[:, -1].argmax(axis=-1).numpy().astype(np.int32).reshape(-1, 1)
      next_tokens[ctx[:, -1] == eot] = eot
      ctx = np.concatenate((ctx, next_tokens), axis=1)
      pos = ctx.shape[-1] - 1
      if (next_tokens == eot).all(): break
    return ctx

  def gettexttoks(line): return [tok for tok in line if tok < eot or tok > enc._special_tokens["<|notimestamps|>"]][-nsample+len(start_tokens):]
  start_tokens = [enc._special_tokens["<|startoftranscript|>"]]
  if model.is_multilingual:
    # TODO detect language
    language_token = enc._special_tokens["<|startoftranscript|>"] + 1 + tuple(LANGUAGES.keys()).index("en")
    start_tokens.append(language_token)
    start_tokens.append(enc._special_tokens["<|transcribe|>"])
  start_tokens.append(enc._special_tokens["<|notimestamps|>"])

  eot = enc._special_tokens["<|endoftext|>"]

  ctx = np.tile(start_tokens, (model.batch_size,1))
  transcriptions = [[] for _ in waveforms]

  for curr_frame in range(0, log_spec.shape[-1], FRAMES_PER_SEGMENT):
    encoded_audio = model.encoder(torch.tensor(log_spec[:, :, curr_frame:curr_frame + FRAMES_PER_SEGMENT]))

    if all(len(c) == len(ctx[0]) for c in ctx): ctx = inferloop(np.array(ctx), encoded_audio)
    else: ctx = [inferloop((np.array([c]*model.batch_size)), encoded_audio)[i] for i,c in enumerate(ctx)]

    for i, (res, arr) in enumerate(zip(transcriptions, ctx)):
      if curr_frame*HOP_LENGTH <= len(waveforms[i]):res.extend(arr[np.where(arr == start_tokens[-1])[0][0]+1:eoti[0] if len (eoti:=np.where(arr == eot)[0]) else None])
    ctx = [[enc._special_tokens['<|startofprev|>']]+gettexttoks(cs)+start_tokens for cs in ctx]

  transcriptions = list(map(lambda tokens: enc.decode(tokens).strip(), transcriptions))
  return transcriptions if len(transcriptions) > 1 else transcriptions[0]

CHUNK = 1600
RECORD_SECONDS = 10

def listener(q):
  import pyaudio
  p = pyaudio.PyAudio()
  stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
  print("listening")
  for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    waveform = ((np.frombuffer(data, np.int16)/32768).astype(np.float32)*3)
    q.put(waveform)
  print("done listening")


def getenv(key:str, default=0): return type(default)(os.getenv(key, default))

if __name__ == "__main__":
  model, enc = init_whisper("small.en" if getenv("SMALL") else "tiny.en", batch_size=1)

  if len(sys.argv) > 1:
    print(transcribe_file(model, enc, sys.argv[1]))
  else:
    # online
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=listener, args=(q,))
    p.daemon = True
    p.start()

    lst = [enc._special_tokens["<|startoftranscript|>"], enc._special_tokens["<|notimestamps|>"]]
    total = None
    did_read = False
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
      while not q.empty() or total is None:
        waveform = q.get()
        if total is None: total = waveform
        else: total = np.concatenate([total, waveform])
        did_read = True
      if did_read:
        log_spec = prep_audio(total.reshape(1, -1), model.batch_size, truncate=True)
        encoded_audio = model.encoder.encode(nn.tensor(log_spec))
      out = model.decoder(nn.tensor([lst]), 0, encoded_audio, streaming=True).realize()
      idx = int(out[0,-1].argmax().numpy().item())
      lst.append(idx)
      dec = enc.decode(lst)
      print(dec)
      if dec.endswith("<|endoftext|>"):
        lst.pop()