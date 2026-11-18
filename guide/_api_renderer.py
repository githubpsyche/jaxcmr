"""Render source-derived API entries for the API reference."""

from html import escape

from griffe import (
    Docstring,
    DocstringReturn,
    DocstringSectionReturns,
    DocstringSectionYields,
    Parser,
)
from plum import dispatch
from quartodoc import MdRenderer, get_object
from quartodoc.layout import Doc, DocAttribute, DocClass, DocFunction, Page


class Renderer(MdRenderer):
    """Render compact signatures and NumPy generator contracts."""

    style = "jaxcmr-reference"

    def __init__(self, module: str, **options):
        super().__init__(**options)
        self.module = module

    @dispatch
    def render(self, page: Page) -> str:
        """Place the source overview and grouped index before the complete entries."""
        module = get_object(self.module, parser="auto")
        parts = [self.render(module.docstring)] if module.docstring else []
        for section in page.contents:
            summary = self.summarize(section)
            heading, rest = summary.split("\n", 1)
            parts.append(f"{heading} {{.api-function-index}}\n{rest}")
        parts.append("## API entries")
        with self._increment_header():
            for section in page.contents:
                parts.extend(self.render(entry) for entry in section.contents)
        return "\n\n".join(parts)

    @dispatch
    def signature(self, document: DocFunction | DocClass) -> str:
        if isinstance(document, DocClass):
            for base in document.obj.bases:
                if str(base) in {"Protocol", "NamedTuple"}:
                    parameters = ", ".join(item.name for item in document.obj.type_parameters)
                    name = document.obj.path + (f"[{parameters}]" if parameters else "")
                    return f'<div class="api-signature"><code>{escape(name)}: {base}</code></div>'
        signature = self.signature(document.obj)
        plain = signature.removeprefix("```python\n").removesuffix("\n```")
        plain = plain.replace("(\n    ", "(", 1).replace(",\n)", ")")
        return f'<div class="api-signature"><code>{escape(plain)}</code></div>'

    @dispatch
    def summarize(
        self, document: Doc, path: str | None = None, shorten: bool = False
    ) -> str:
        # Entries are embedded below the index instead of published separately.
        link = f"[`{document.name}`](#{document.anchor})"
        return self._summary_row(link, self.summarize(document.obj))

    @dispatch
    def render_header(self, document: Doc) -> str:
        header = super().render_header(document)
        if self.crnt_header_level == 3:
            header = header.replace(" }", " .api-entry }")
        # Markdown otherwise treats __init__ as bold text and hides underscores.
        if document.name.startswith("__"):
            header = header.replace(document.name, f"`{document.name}`", 1)
        return header

    @dispatch
    def render(self, docstring: Docstring) -> str:  # noqa: F811 - Plum dispatch overload.
        if docstring.parser == Parser.auto:
            # A wrapped Google Returns paragraph describes one returned value.
            docstring = Docstring(
                docstring.value,
                parent=docstring.parent,
                parser=Parser.auto,
                parser_options={"per_style_options": {
                    "google": {"returns_multiple_items": False},
                    "numpy": {"allow_section_blank_line": True},
                }},
            )
        return super().render(docstring)

    @dispatch
    def render(self, document: DocAttribute) -> str:  # noqa: F811 - Plum dispatch overload.
        if "type-alias" not in document.obj.labels:
            return super().render(document)
        definition = f"type {document.obj.name} = {document.obj.value}"
        signature = (
            '<div class="api-signature"><code>'
            f'{escape(definition)}</code></div>'
        )
        return "\n\n".join([
            self.render_header(document), signature, self.render(document.obj),
        ])

    @dispatch
    def render(self, document: DocClass) -> str:  # noqa: F811 - Plum dispatch overload.
        if not any(str(base) == "TypedDict" for base in document.obj.bases):
            return super().render(document)

        # Typed dictionaries describe fields, not a constructor or methods.
        title = self.render_header(document)
        signature = (
            '<div class="api-signature"><code>'
            f'{escape(document.obj.path)}: TypedDict</code></div>'
        )
        with self._increment_header():
            description = self.render(document.obj)
        fields = ["| Name | Type | Description |", "| --- | --- | --- |"]
        for member in document.members:
            field_description = (
                " ".join(self.render(member.obj.docstring).split()).replace("|", "\\|")
                if member.obj.docstring else ""
            )
            fields.append(
                f"| `{member.name}` | {self.render_annotation(member.obj.annotation)} "
                f"| {field_description} |"
            )
        heading = (
            f"{'#' * (self.crnt_header_level + 1)} Fields "
            "{.doc-section .doc-section-attributes}"
        )
        return "\n\n".join([title, signature, description, heading, "\n".join(fields)])

    @dispatch
    def render(self, section: DocstringSectionYields) -> str:  # noqa: F811 - Plum dispatch overload.
        # Quartodoc retains the Yields heading above these value descriptions.
        return self.render(DocstringSectionReturns([
            DocstringReturn(
                name=item.name,
                annotation=item.annotation,
                description=item.description,
            )
            for item in section.value
        ]))
