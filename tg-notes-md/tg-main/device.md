### device.py


***** Imports ***** 
        - __future__
            - annotations
        - dataclasses
            - dataclass
            - replace
        - collections
            - defaultdict
        - typing
            - Optional
            - Any
            - Iterator
            - Genrator
        - multiprocessing, importlib, inspect, functools, pathlib, os
        - ctypes, contextlib, sys, re, atexit, pickle, decimal, time
        - tinygrad.helpers
            - CI, OSX
            

***** Classes and function ***** 

----------------------- Profile ---------------------------------
        - class _Device
      
----------------------- Profile ---------------------------------
        - class ProfileEvent
        - class ProfileDeviceEvent(ProfileEvent)
        - class ProfileRangeEvent(ProfileEvent)
        - class ProfileGraphEntry
        - class ProfileGraphEvent
        - class ProfileResult
        - def cpu_profile(name, device, is_copy, display)

---------------------- Buffer + Allocators -----------------------
        - class BufferSpec
        - class Buffer
        - class Allocator
        - class LRUAllocator(Allocator)
        - class _MallocAllocator(LRUAllocator)
        - class CompileError(Excpetion)
        - class Compiler
        - class Compiled
        - def is_dtype_supported(dtype, device)

