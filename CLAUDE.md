## OpenMC Codebase Tools

Read the FULL `AGENTS.md` in this directory before starting work. It contains
project context, coding conventions, and documentation of the RAG search tools
registered in `.mcp.json`.

### Claude Code-specific: first-call behavior

The first `openmc_rag_search` call of each session returns an index status
message instead of search results. When this happens, you MUST use the
`AskUserQuestion` tool to present the rebuild/use-existing choice to the user.
Do not ask conversationally — always use the widget. Do not skip this step even
if the index looks current — the user may have uncommitted changes that warrant
a rebuild.

## Design decisions

- **No C++ PENDF parser.** GENDF has one (`src/gendf_parser.cpp` +
  `src/gendf.cpp`, exposed as `openmc.lib.gendf.GENDFLibrary`) because its
  ASCII tapes are parsed at runtime. PENDF data is preprocessed once into
  HDF5 (`PendfLibrary`/`GroupedPendfLibrary`); the ENDF-6 text -> HDF5 step
  (`PendfLibrary.from_endf_directory` / `tools/pendf_to_hdf5.py`) is a
  seldom-run, one-time build step, not something in the collapse hot path,
  so a compiled parser isn't worth building for it.
