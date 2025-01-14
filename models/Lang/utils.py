import pathlib, os, platform, gzip
import urllib.request, subprocess, functools, tempfile, hashlib
from typing import Union, Optional



OSX = platform.system() == "Darwin"
CI = os.getenv("CI", "") != ""

class tqdm:
  def __init__(self, iterable=None, desc:str='', disable:bool=False, unit:str='it', unit_scale=False, total:Optional[int]=None, rate:int=100):
    self.iterable, self.disable, self.unit, self.unit_scale, self.rate = iterable, disable, unit, unit_scale, rate
    self.st, self.i, self.n, self.skip, self.t = time.perf_counter(), -1, 0, 1, getattr(iterable, "__len__", lambda:0)() if total is None else total
    self.set_description(desc)
    self.update(0)
  def __iter__(self):
    for item in self.iterable:
      yield item
      self.update(1)
    self.update(close=True)
  def __enter__(self): return self
  def __exit__(self, *_): self.update(close=True)
  def set_description(self, desc:str): self.desc = f"{desc}: " if desc else ""
  def update(self, n:int=0, close:bool=False):
    self.n, self.i = self.n+n, self.i+1
    if self.disable or (not close and self.i % self.skip != 0): return
    prog, elapsed, ncols = self.n/self.t if self.t else 0, time.perf_counter()-self.st, shutil.get_terminal_size().columns
    if self.i/elapsed > self.rate and self.i: self.skip = max(int(self.i/elapsed)//self.rate,1)
    def HMS(t): return ':'.join(f'{x:02d}' if i else str(x) for i,x in enumerate([int(t)//3600,int(t)%3600//60,int(t)%60]) if i or x)
    def SI(x): return (f"{x/1000**int(g:=math.log(x,1000)):.{int(3-3*math.fmod(g,1))}f}"[:4].rstrip('.')+' kMGTPEZY'[int(g)].strip()) if x else '0.00'
    prog_text = f'{SI(self.n)}{f"/{SI(self.t)}" if self.t else self.unit}' if self.unit_scale else f'{self.n}{f"/{self.t}" if self.t else self.unit}'
    est_text = f'<{HMS(elapsed/prog-elapsed) if self.n else "?"}' if self.t else ''
    it_text = (SI(self.n/elapsed) if self.unit_scale else f"{self.n/elapsed:5.2f}") if self.n else "?"
    suf = f'{prog_text} [{HMS(elapsed)}{est_text}, {it_text}{self.unit}/s]'
    sz = max(ncols-len(self.desc)-3-2-2-len(suf), 1)
    bar = '\r' + self.desc + (f'{100*prog:3.0f}%|{("█"*int(num:=sz*prog)+" ▏▎▍▌▋▊▉"[int(8*num)%8].strip()).ljust(sz," ")}| ' if self.t else '') + suf
    print(bar[:ncols+1], flush=True, end='\n'*close, file=sys.stderr)
  @classmethod
  def write(cls, s:str): print(f"\r\033[K{s}", flush=True, file=sys.stderr)

@functools.lru_cache(maxsize=None)
def getenv(key:str, default=0): return type(default)(os.getenv(key, default))

def _ensure_downloads_dir() -> pathlib.Path:
  return pathlib.Path(cache_dir) / "downloads"

def fetch(url:str, name:Optional[Union[pathlib.Path, str]]=None, subdir:Optional[str]=None, gunzip:bool=False,
          allow_caching=not getenv("DISABLE_HTTP_CACHE")) -> pathlib.Path:
  if url.startswith(("/", ".")): return pathlib.Path(url)
  if name is not None and (isinstance(name, pathlib.Path) or '/' in name): fp = pathlib.Path(name)
  else: fp = _ensure_downloads_dir() / (subdir or "") / ((name or hashlib.md5(url.encode('utf-8')).hexdigest()) + (".gunzip" if gunzip else ""))
  if not fp.is_file() or not allow_caching:
    (_dir := fp.parent).mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=10) as r:
      assert r.status == 200, r.status
      length = int(r.headers.get('content-length', 0)) if not gunzip else None
      readfile = gzip.GzipFile(fileobj=r) if gunzip else r
      progress_bar = tqdm(total=length, unit='B', unit_scale=True, desc=f"{url}", disable=CI)
      with tempfile.NamedTemporaryFile(dir=_dir, delete=False) as f:
        while chunk := readfile.read(16384): progress_bar.update(f.write(chunk))
        f.close()
        pathlib.Path(f.name).rename(fp)
      progress_bar.update(close=True)
      if length and (file_size:=os.stat(fp).st_size) < length: raise RuntimeError(f"fetch size incomplete, {file_size} < {length}")
  return fp


cache_dir: str = os.path.join(getenv("XDG_CACHE_HOME", os.path.expanduser("~/Library/Caches" if OSX else "~/.cache")), "pytorch")

def colored(st, color:Optional[str], background=False): return f"\u001b[{10*background+60*(color.upper() == color)+30+['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'].index(color.lower())}m{st}\u001b[0m" if color is not None else st  # replace the termcolor library with one line  # noqa: E501