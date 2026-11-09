# Vendored simple-pytree

JAXCMR bundles `simple-pytree` **0.2.2** by Cristian Garcia, under the MIT
license in [LICENSE](LICENSE).

The three Python files (`__init__.py`, `dataclass.py`, and `pytree.py`) are
copied byte-for-byte from the published wheel, without behavioral changes.
Imports within JAXCMR use `jaxcmr._vendor.simple_pytree`; this package does not
install or expose a top-level `simple_pytree` module.

## Provenance

- Project: <https://pypi.org/project/simple-pytree/0.2.2/>
- Release date: 2023-09-06
- Source artifact: `simple_pytree-0.2.2-py3-none-any.whl`
- Download: <https://files.pythonhosted.org/packages/7c/16/0272467306ef489512a843222567c9939b9aff7003f15474a1ef90168c8f/simple_pytree-0.2.2-py3-none-any.whl>
- Artifact SHA-256: `3a7a2f66883194ab14875dd01c4306c4f102f31d90e7e2d421366f12a2c49bab`

## Maintenance

The upstream distribution declares Python `<3.12`, while JAXCMR requires
Python `>=3.12`. Bundling the existing implementation removes that packaging
conflict. JAXCMR is responsible for testing this copy against its supported
Python and JAX versions.

For an update, verify the upstream artifact and license, replace the Python
files together, record the version and hash here, and document any local
changes. Run `tests/test_vendor_pytree.py` and the full test suite, then build
and install the wheel in a fresh environment without the external
`simple-pytree` distribution. Confirm this directory's license and provenance
remain included in the wheel.
