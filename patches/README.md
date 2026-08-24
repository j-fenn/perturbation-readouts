# patches

GEARS is MIT-licensed, actively maintained, and installable. This repository
depends on it (`cell-gears==0.1.2`) and does not vendor it. **There is currently
no patch here**, and the pipeline applies none.

That is a deliberate difference from the original BMI 212 project, which
implemented the bipartite layer by editing `gears/model.py` in a vendored copy
of the GEARS tree — adding an `is_hgnn` flag that inserted a gene→perturbation
`SGConv` and an MLP merge into `GEARS_Model.forward`. That approach makes the
baseline and the experimental model the same object, so a change to one silently
changes the other, and it makes "which GEARS did you run?" unanswerable.

Here the baseline **is** the pinned package's `GEARS_Model`, untouched, and the
bipartite model is a separate module (`src/pertreadout/models/bipartite_hgnn.py`)
that consumes the same `PertData` and is trained by the same loop under the same
loss.

If a behavioural change to GEARS ever becomes necessary, it belongs in this
directory as a `.patch` file with a header explaining what it changes and why,
applied explicitly by the pipeline — never as an edit to an installed package.

## Not a patch: version pins

Two pins in `pyproject.toml` exist because `cell-gears==0.1.2` predates changes
in its own dependencies. They change no GEARS behaviour:

- `pandas<3` — GEARS uses positional `Series[0]`, removed in pandas 3.
- `scipy<1.15` — GEARS indexes a sparse matrix with a pandas boolean `Series`,
  which scipy's newer sparse index validation rejects.
