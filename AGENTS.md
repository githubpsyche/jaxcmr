# AGENTS.md — Library Docstrings & Unit Tests (Minimal)

**Applicability**

* **Library code (non‑tests):** Sections **1** and **2** apply across the entire codebase.
* **Unit tests:** Section **3** applies to all test modules.

---

## 1) Non‑test function docstrings (Google style)

**Rules**

* **All functions must be fully type‑annotated** (every argument and the return type).
* **Do not repeat types in docstrings.** Rely on annotations; docstrings explain meaning, shape, and units.
* Start with an imperative **summary**. When it improves clarity/concision, start the summary with **“Returns …”** and **omit a separate Returns section**.
* Include sections only as needed, in Google style order.
* **Args** entries must match parameter names and order in the signature.
* For arrays/structures, do not specify **shape/units** in the Args section.
* Keep examples brief and stable.

**Skeleton — returns‑first (preferred when the return is simple)**

```python
def foo(x: int, y: float) -> float:
    """Returns <what is produced>.

    Args:
      x: <meaning>.
      y: <meaning>.
    """
```

**Skeleton — classic (use when the return is complex)**

```python
def bar(q: Array, axis: int) -> tuple[Array, Array]:
    """Compute <what it does>.

    Args:
      q: <meaning>. Shape [...]; units <...>.
      axis: <meaning>.

    Returns:
      (<name1>, <name2>): <meaning of tuple and each item; shapes/units>.
    """
```

---

## 2) Class and module docstrings (Google style)

Follow the Google Python Style Guide **except as noted here**.

* Start with a one‑sentence summary.
* Optionally add any needed paragraphs of context (what/why, not how).
* Use optional sections **ONLY** when they add value that exceeds the cost of maintenance and complexity.
* **Never include an “Attributes” section in class docstrings.**
* **Never include an “Examples” section in module docstrings.**
* Do not repeat types already present in annotations.

**Class skeleton (minimal, optional sections)**

```python
class Widget:
    """One‑sentence summary.

    <Optional extended summary>

    <Optional sections>
    """
```

**Module skeleton (minimal, optional sections)**

```python
"""One‑ or two‑line summary.

<Optional extended summary>

<Optional sections>
"""
```

---

## 3) Unit tests (in tests/test_*.py files)

**Naming**

* Name tests as: `test_<behavior>_when_<condition>()`
  *Do not* prefix with `<unit>`.

**Structure & style**

* Docstrings **must** use **Given / When / Then** wording with the skeleton below.
* Test bodies use **AAA** (Arrange / Act / Assert) with **exactly one “When”** (a single action).
* **No pytest fixtures or custom marks.**
  **Exception:** `@pytest.mark.parametrize` is allowed.
* **No loops**—use parametrization for multiple cases.
* **Assert observable outcomes**; avoid internal implementation details.

**Required skeleton (copy/paste exactly)**

```python
def test_<behavior>_when_<condition>():
    """Behavior: <what the unit guarantees>.

    Given:
      - <critical preconditions>
    When:
      - <single action>
    Then:
      - <observable outcome(s)>
    Why this matters:
      - <link to requirement/invariant/regression id>
    """
    # Arrange / Given
    ...
    # Act / When
    ...
    # Assert / Then
    ...
```

## 4) Example Notebooks (in examples/*.ipynb files)

**Purpose:** example-driven, parameterizable demos or quick reports devoted to a single concept or deliverable. **Not** a source of truth for semantics (use module/function docstrings; link to the API if needed).

Each notebook must run top-to-bottom with default parameters (no manual input).

**Required structure (in order):**

1. **Markdown — Title** (`# …`)
2. **Markdown — One-line imperative summary**
3. **Markdown — Short overview** (2–5 sentences for a non-technical scientist; basic mechanics & interpretation)
4. **Code — Imports**
5. **Code — Parameters** *(tag this cell `parameters`; plain assignments compatible with papermill)*
6. **Code — Demo(s)** (clarifying demonstrations or typical usage examples; number/size **as needed**)
7. **Markdown — `## Notes` (optional)** (scope **as needed**)
