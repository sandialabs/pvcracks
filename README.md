# pvcracks
Si-PV cell crack image recognition, modeling, and power loss prediction

Paper DOI: https://doi.org/10.2139/ssrn.5506469

DuraMAT DataHub DOI: https://doi.org/10.21948/2587738

## Dependencies

Install [uv](https://docs.astral.sh/uv/) and run `uv sync`. This will install the dependencies in `pyproject.toml`

If there are errors when importing `utils`, run `uv pip install -e .` in `pvcracks/` to enable intra-project imports.

An auto-generated `requirements.txt` file for use with `pip` has also been provided for your convenience, but compatibility is not guaranteed.

## Execution

Download the data from the DuraMAT DataHub. Then follow the instructions in `src/multi_channel/README.md` for instructions on how to set up the data and run the model.