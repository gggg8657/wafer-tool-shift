"""WM-811K as a tool/geometry domain-shift benchmark.

`features` is imported lazily because it needs scipy, which the GPU training
environment does not carry -- the descriptors are computed once by
`scripts/extract.py` and cached.
"""
