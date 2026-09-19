---
tags:
  - gotcha
---

# Gotcha: `exec()` overwrites `__doc__`

Several scripts reuse [[Script 91 Other Stages]] by `exec`-ing its source prefix.
CPython compiles a source string in `"exec"` mode, and a **leading string literal
becomes the module docstring**, assigned into the target globals.

So `exec(other_module_source)` **silently replaces the calling module's
`__doc__`**.

Effect when it bit: [[Script 98 Reorder On Norelink]] printed *91's* docstring as
its log header, so a committed artifact described the wrong experiment. The
numbers were right; the label was not.

Fix: capture `_DOC = __doc__` **before** the `exec`.

A small bug, recorded because it produces a confidently-wrong artifact rather
than an error.
