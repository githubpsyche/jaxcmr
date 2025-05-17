---
format:
  pdf:
    geometry: [landscape, margin=0.1in]   # page setup
    pagestyle: empty                      # no headers/footers
    include-before-body:
      text: |
        \vspace*{\fill}                   % elastic space at top
    include-after-body:
      text: |
        \vspace*{\fill}                   % elastic space at bottom
---


::: {layout-ncol="3"}
![](figures/HealeyKahana2014_BaseCMR_Fitting_pnr.png)

![](figures/HealeyKahana2014_BaseCMR_Fitting_crp.png)

![](figures/HealeyKahana2014_BaseCMR_Fitting_spc.png)
:::