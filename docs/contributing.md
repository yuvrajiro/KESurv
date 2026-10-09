# Contributing

Contributions to implementation, tests, examples, and documentation are welcome. Keep examples executable and describe the behavior the code actually provides.

## Local checks

```console
python -m pip install -e ".[test,docs]"
python -m pytest -q
python -m sphinx -b html -W --keep-going docs docs/_build/html
```

Open `docs/_build/html/index.html` to preview the site. Sphinx treats warnings as errors in the documented build command so broken links and API references are caught early.

To regenerate the executed synthetic notebook after changing the example or prediction API:

```console
python -m pip install -e ".[docs,demo]"
python examples/build_synthetic_notebook.py
```

## Releases

Documentation is rebuilt and deployed to [kesurv.statml.in](https://kesurv.statml.in) on every push to `main`. To release a new version, bump `version` in `pyproject.toml`, then publish a GitHub release tagged `vX.Y.Z`; the `publish` workflow builds the package and uploads it to PyPI.
