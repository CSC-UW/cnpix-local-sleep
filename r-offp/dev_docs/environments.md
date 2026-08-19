# R environments and IDEs on `tononi-2`

Your choice of environment and how you manage it may be dictated by your choice of IDE. If support for virtual environments is poor, for example, choose a `pixi global` installation of `R` and `quarto`.

The options are:

1. Use system `R` (never recommended)
2. Use `pixi` to perform a `global install` of `R`
    1. Also manage packages using `pixi`.
    2. Manage packages using `renv`.
3. Use `pixi` to create a proper (non-global) virtual environment with `R`.
    1. Also manage packages using `pixi`.
    2. manage packages using `renv`.

Although it saddens me, I currently reccomend option (2). There are just too many rough edges with how IDEs interact with option (3).

NB: There is no case in which `conda` or `mamba` would be better than `pixi`.

## 1. Using system `R` (never recommended)

Using system `R` on `tononi-2` is terrible!
The combination of the system-wide site library (that you cannot write to without root), and a personal library, never "just works". Don't bother!
[`rig`](https://github.com/r-lib/rig) is a no-go because it requires admin (to install _and_ to use).

## 2. Using `pixi` to perform a `global install` of `R`

Because many R ecosystem tools actually _expect_ you to be using a global `R` installation, rather than a virtual environment (/sigh), this approach generally works well and avoids a lot of headaches.

```{bash}
pixi global install --environment my-r-env r-base radian quarto pandoc
# System radian or quarto would probably be fine, too. They'd find R on the PATH.
# Or, you could install radian with `uv tool install radian`
# Maybe `uv tool quarto quarto-cli` works. Never tried.
```

This creates an environment at `~/.pixi/envs/my-r-env`. By default, it exposes `R`, `Rscript`, `radian`, `quarto`, and `quarto.js` executables to your `PATH`. The _real_ executables are in `~/.pixi/envs/my-r-env`, but `pixi` also creates "trampoline binaries" (static ELF binaries) in `~/.pixi/bin`. These are NOT simply symlinks that point to the real executables. This is important, because some programs (Positron for sure) expect an `R` exectuable to be a shell script, and try to parse it looking for `R_HOME`. When they try to parse the trampoline binaries and fail, they will claim you have an invalid installation. So it is very important to always provide paths to the real executables.

Make sure to log out and back in after installing `r-base` and `radian`, or else the latter will continue to point to system R.

NB: This works for both VS Code and Positron

### 2.1. Package management using `pixi`

```{bash}
pixi global install --environment my-r-env r-arrow r-tidyverse ...
# To reproduce, use ~/.pixi/manifests/pixi-global.toml
```

If you are using VS Code and want an `R` debugger:
Unfortuantely, there is no `conda-forge` feedstock for [`vscDebugger`](https://github.com/ManuelHentschel/vscDebugger).
You can, however, simply install from within your `pixi` R environment to get it installed to a personal library, which so far seems to work:

```{r}
pak::pak("ManuelHentschel/vscDebugger")
# install.packages("vscDebugger", repos = "https://manuelhentschel.r-universe.dev")
```

You could also do this with a quick shell script if you needed to deploy to a cloud environment.

### 2.2. Package management using `renv`

This approach uses `pixi` only to install the CLI tools: base `R`, `radian`, and `quarto`.
All `R` packages are managed bu `renv`.

```{bash}
pixi global install --environment my-r-env r-base radian quarto pandoc
cd /path/to/R/package
radian
```

For the very first `renv.lock` + `.Rprofile` + `renv/activate.R` creation only:

```{r}
install.packages("renv")
# You could also install r-renv with pixi, above.
options(renv.settings.install.suggests = TRUE)
# Install development/IDE dependencies by default
renv::init(settings = list(snapshot.type = "explicit"))
# I prefer explicit, because it is single-point-of-truth. It only checks `DESCRIPTION`.
renv::settings$package.dependency.fields(c("Imports", "Depends", "LinkingTo", "Suggests"))
# To make behavior persistent. 
q()
```

> [!NOTE]
> When someones starts an `R` session, the new `.Rprofile` will `renv/activate.R`, which will
> download and install `renv` if not already installed.

> [!NOTE]
> Note that once a project is initialized as "explicit", this is saved in `renv/settings.json`.
> To re-initialize a renv from scratch, first remove `renv.lock`, `.Rprofile`, `renv/`,
> and your project library (`renv::paths$library()`), before running `renv::init()`.

#### Annoying side-quest: broken cache entries

At this point, if you have not had a totally fresh start, you may run into the following warning when you try to start an `R` session:

```{txt}
The following package(s) have broken symlinks into the cache:
...
Use `renv::repair()` to try and reinstall these packages.
```

But `renv::repair()` will report no issues. There are ugly workarounds (`renv::install()` each offending package). Or you can nuke your `R` installation and re-create it _being absolute sure not to use any caches_. Be mindful that both `pixi` and `renv` maintain caches.

```{r}
pixi global uninstall my-r-env
pixi clean cache
pixi global install --environment my-r-env r-base radian quarto
rm -rf ~/.cache/R/renv # You may face permission errors and need to chmod
...
```

## 3. Using `pixi` to install a proper R virtual environent

If using Positron, set:

```{json}
"positron.r.interpreters.pixiDiscovery": true,
```

Unfortunately, package development with virtual environments in Positron has a little hiccup at the moment,
because the default build task tries to use misquoted relative paths:
([workaround](https://github.com/posit-dev/positron/issues/11623)).
But you can just define your own, or use `devtools::build()`.

Use the provided `pixi.toml`:

```{bash}
cd path/to/package
pixi install -e dev # Also available: `-e vscode`
```

> [!CAUTION]
> Even if you install a `pixi` "feature" environment (e.g. `pixi install -e dev`),
> `pixi run` uses the `default` environment by default. Of course you can
> override that with `pixi run -e dev`, but because this is the `R` hell ecosystem,
> many tools will just not!
> So it may make sense to just shove all possible dependencies into the `default`
> environment... ='C. Sorta defeats the purpose...
> Or maybe you can ensure that only the `vscode` env is ever created?

Known problems:

- Building with `devtools::build()` is currently running into a bug in the GCC conda-forge feedstock packaging.
 . n environment variable is getting treated as a filename (`TOOLS=addr2line`), and the symlink in
 `.pixi/` needs to be deleted.
- `.Rbuildignore` seems to be getting ignored. The gigantic `.pixi` directory is being traversed!
  This causes build times to balloon from ~5s to ~2m.

### 3.1 Package management using `pixi`

Use the provided `pixi.toml`. If using VS Code:

```{bash}
cd path/to/package
pixi install -e vscode # Also available: `-e dev`
```

In your `.vscode/settings.json`:

```{json}
{
    "r.rterm.linux": "${workspaceFolder}/.pixi/envs/vscode/bin/radian",
    "r.rpath.linux": "${workspaceFolder}/.pixi/envs/vscode/bin/R",
}
```

Also note that you need to make sure `.pixi/` is in your `.Rbuildignore`:

```{txt}
^\.pixi$
^pixi\.lock$
^pixi\.toml$
```

This is a bit finnicky, though. Some quirks:

- `pixi run which R` changes unexpectedly from `vscode` to `default`. Can we find a way to make sure `default` is never created?
- The VS Code R Extension suddently decides to ignore its `settings.json`...
- `pak` installs are complaining that pandoc isn't installed, even though it clearly is.
