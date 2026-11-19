# GitHub Pages handover

The repository currently serves `docs/` on `main`. The [website workflow](workflows/tests.yml)
can publish its checked build after the repository owner switches the Pages source.

1. Confirm that **Checks and website** passes on the current `main` commit and
   inspect the `website-preview` artifact. If branch protection requires checks
   by name, verify that the required names match the workflow.
2. In **Settings → Pages → Build and deployment**, choose **GitHub Actions** as
   the source. Run **Checks and website** on `main` and verify the deployment
   for that commit.
3. On the published site, check the setup pages, notebook downloads, bundled
   inputs, and Colab links. Open a changed notebook in a fresh Colab runtime.
4. Once the deployment is verified, remove the generated `docs/` directory
   from Git tracking and add `/docs/` to `.gitignore`. Until then, keep the
   branch-published copy available. Retire this note when the handover is complete.
