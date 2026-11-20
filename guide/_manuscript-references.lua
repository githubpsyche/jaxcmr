-- Use nocite entries to display the works cited by the manuscript.
function Pandoc(page)
  local path = pandoc.path.join({quarto.project.directory, "..", "paper", "index.qmd"})
  local file = assert(io.open(path, "r"))
  local manuscript = pandoc.read(file:read("*a"), "markdown+tex_math_single_backslash")
  file:close()
  manuscript.meta.bibliography = page.meta.bibliography

  local citations = {}
  for _, reference in ipairs(pandoc.utils.references(manuscript)) do
    local id = pandoc.utils.stringify(reference.id)
    table.insert(citations, pandoc.Citation(id, "NormalCitation"))
  end
  page.meta.nocite = pandoc.Inlines({pandoc.Cite({}, citations)})
  return page
end
