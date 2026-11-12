# Setup screenshots

All screenshots show dark mode. Source pixels are unchanged. `screenshots.css`
adds red borders over the relevant controls on the page and in Quarto's lightbox.
The stylesheet is loaded only by the local and Colab setup guides.

## VS Code

Microsoft documentation screenshots, retrieved 2026-11-12. Copyright Microsoft
Corporation; licensed under [CC BY 3.0 US](https://creativecommons.org/licenses/by/3.0/us/)
per the [upstream license](https://github.com/microsoft/vscode-docs/blob/main/LICENSE.md).
The local guide identifies these as Microsoft examples and includes source links
and attribution under “Screenshot sources.” Their filenames and versions are not
the project's filenames or required Python version.

Asset URLs are relative to `https://code.visualstudio.com/assets/docs/`:

| Local file | Source asset |
| --- | --- |
| `vscode-extensions.png` | `configure/extensions/extension-marketplace/extensions-python.png` |
| `vscode-terminal.png` | `terminal/getting-started/open-terminal.png` |
| `vscode-shell.png` | `terminal/getting-started/select-shell.png` |
| `vscode-clone-palette.png`, `vscode-open-folder.png` | `editing/userinterface/commands.png` |
| `vscode-notebook-controls.png` | `datascience/jupyter/native-kernel-picker.png` |
| `vscode-run.png` | `datascience/jupyter/native-code-runs.png` |

Some Microsoft screenshots already include red outlines. Additional outlines are
CSS overlays. Update their percentage coordinates in `screenshots.css` if an image
is replaced, and check both the inline and enlarged views.

## Colab

Captured 2026-11-12 from the public
[first-simulation notebook](https://colab.research.google.com/github/githubpsyche/jaxcmr/blob/main/guide/examples/first-simulation.ipynb)
using Colab's dark theme:

- `colab-copy-dark.jpg` and `colab-save-dark.jpg`: the same File-menu capture,
  highlighting different commands in the two sections.
- `colab-runtime-dark.jpg`: the runtime settings dialog, showing Python 3 and CPU.
- `colab-run-dark.jpg`: the toolbar and the final code cell's saved output, with
  the title/menu header collapsed.

No runtime was started, notebook copied, or notebook edited for the captures above.
The run screenshot shows saved output, as its caption explains. Captures exclude
account controls and avoid the older introduction on the currently published
notebook. When updating them, preserve this distinction between displayed saved
output and evidence of an executed Colab session.

`colab-file-actions-dark.jpg` was captured later on 2026-11-12 to show the Files
panel's Copy path and Download commands. Opening the panel allocated a runtime;
only Colab's built-in sample files were viewed. No notebook cells were executed,
no personal files uploaded, and Drive was not connected. The screenshot excludes
account controls. The capture tab was closed afterward, and the original light
theme restored.
