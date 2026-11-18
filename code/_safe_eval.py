"""Offline-safe re-eval: same pipeline as scripts/evaluate.py, but flattens
scores/labels before EER so that a stray (n,1) array cannot crash compute_eer.
Usage: python3 _safe_eval.py --config <yaml> --ckpt <pt> --split eval [--save-scores] [--limit N]
"""
import sys, runpy
sys.path.insert(0, '/root/wcl')
sys.path.insert(0, '/root/wcl/scripts')
import numpy as np
import wcl.utils as u

orig = u.compute_eer


def patched(scores, labels):
    s = np.asarray(scores)
    l = np.asarray(labels)
    if s.ndim > 1 or l.ndim > 1:
        print(f'[fix] flatten scores {s.shape} labels {l.shape}', flush=True)
    s = s.ravel()
    l = l.ravel()
    if s.shape != l.shape:
        print(f'[fix][FATAL] length mismatch {s.shape} vs {l.shape}', flush=True)
    return orig(s, l)


u.compute_eer = patched
sys.argv = ['evaluate.py'] + sys.argv[1:]
runpy.run_path('/root/wcl/scripts/evaluate.py', run_name='__main__')

