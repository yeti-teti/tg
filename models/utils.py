import pathlib, os, platform, gzip
import urllib.request, subprocess, functools, tempfile, hashlib
from typing import Union, Optional
from tqdm import tqdm


OSX = platform.system() == "Darwin"
CI = os.getenv("CI", "") != ""


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
