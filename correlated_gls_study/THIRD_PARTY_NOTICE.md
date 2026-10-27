# Third-party code

- `official_cov1para.py`, `official_qis.py`, `official_qis_eigh.py`: function definitions extracted from
  https://github.com/pald22/covShrinkage (commit a03fe68), Copyright (c) Olivier Ledoit, Michael Wolf
  and Patrick Ledoit; see `LICENSE_covShrinkage.txt`; the BSD 2-clause notice of each original file is retained at the top of the extracted file.
  `official_qis_eigh.py` replaces `np.linalg.eig` by `np.linalg.eigh`, which keeps the estimate positive
  definite when T < p and gives the same result when T > p.
- `nonlinshrink_port.py`: from the PyPI package nonlinshrink 0.7 (Matz Haugen, MIT licence),
  https://github.com/matzhaugen/analytic_shrinkage (licence text in `LICENSE_nonlinshrink.txt`), a port of the analytical nonlinear shrinkage
  code of Ledoit and Wolf (2020); changes: the numpy.matlib import line and the working-paper link in the docstring.
