# Using IDEs with R

First, read [environments.md](environments.md).

## Using VS Code

The primary advantage of using VS Code is that the Claude Code extension works better.
A `pixi global` installation is recommended.

```{json}
    "r.rterm.linux": /path/to/real/executable
    "r.rpath.linux": /path/to/real/executable
    // Yes, you need these, even if the right `R` shows up with `which R`. 
```

For code intelligence to work with your functions, you MUST open the package directory (`offp/`) as a workspace!

_If_ you are working from a parent directory, be mindful that VS Code's R extension will set your initial working directory differently than RStudio or Positron. You may want to stick a `.Rprofile` at the root of your VS Code project.
For example:

```{r}
setwd("/path/to/offp")
```

This will ensure that this is your initial working directory, and that `here` beings moving up the filesystem hierarchy from there.

## Using Positron

```{json}
    "positron.r.interpreters.pixiDiscovery": true,
    "positron.r.customBinaries": ["/path/to/your/pixi/envs/<env-name>/bin/R"],
    "positron.r.interpreters.exclude": [
        "/usr/bin/R"
    ],
```

If you do this, and use a global install of R, everything should work!
