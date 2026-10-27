# Cross-Polarity Network Enricher (CPNE) — v0.6 demo profile

A Streamlit app for matching positive- and negative-mode LC–MS features using adduct-aware neutral masses and retention time, then enriching a target molecular network with cross-polarity and optional PostMN MassQL information.

## Install and run

Install Python 3.10+ if needed, open a terminal in this folder, and run:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser, usually at `http://localhost:8501`.

## Try the configured example

Click **Load example files** at the top of the app. It preloads this example profile:

- Positive GNPS2 task ID: `4a83b08f0c9c41888b7db29c267f3cf1`
- Negative GNPS2 task ID: `0bd4d95478e7499dad2ffead34ddec13`
- Target: positive-mode network enriched with negative-mode information
- Positive adducts: `[M+H]+`, `[M+Na]+`, `[M+NH4]+`
- Negative adducts: `[M-H]-`, `[M-H2O-H]-`, `[M+FA-H]-`, `[M+Na-2H]-`
- Neutral-mass tolerance: 10 ppm
- RT tolerance: 0.20 min, with RT compatibility required
- Optional edge tagging: target precursor Δm = 42.047 Da, allowed error = ±0.001 Da
- Task-specific PostMN MassQL exports from the two uploaded files are bundled with the app

The app first attempts to fetch those public GNPS2 task outputs. After a successful fetch, it attempts to save the selected GraphML and node/feature table files in `examples/gnps2_demo_task_cache.zip`. Later, if GNPS2 is unavailable, the app can read that snapshot if it is still present. On Streamlit Community Cloud, files written at runtime are not guaranteed to persist after a restart.

**Note about the first offline use:** The current development environment could not access the GNPS2 file server, so a real-task snapshot could not be pre-populated in this distributed ZIP. If GNPS2 is unavailable before the app has ever fetched these tasks successfully, CPNE falls back to a clearly labelled synthetic example network and feature tables. Synthetic MassQL tables are used with that fallback so you can click through the full workflow end to end. The exact task-specific MassQL TSV exports you supplied are also bundled as `examples/provided_postmn_positive.tsv` and `examples/provided_postmn_negative.tsv` and can be downloaded from within the app.

After one successful live fetch in a local installation, keep `examples/gnps2_demo_task_cache.zip` with the app folder to retain the actual GNPS2 example for future offline demos. To distribute that real snapshot to another computer, include the cache ZIP with the project only if you are permitted to redistribute those results. For an online deployment, see `DEPLOY_TO_STREAMLIT_CLOUD.md`.

## Logo

`assets/cpne_logo.png` is the current placeholder icon: a circular blue/orange molecular-network emblem with a green central integration point. It is displayed in the app header and used as the browser tab icon.

## Upload files mode

Provide:

- Positive-mode feature/quantification table (`.csv`, `.tsv`, `.txt`)
- Negative-mode feature/quantification table (`.csv`, `.tsv`, `.txt`)
- Target network GraphML for the selected target polarity
- Optional PostMN MassQL Full Table exports for either or both polarities

MZmine/GNPS2 columns such as `row ID`, `row m/z`, `row retention time`, `cluster index`, `parent mass`, and `RTMean` are supported. PostMN MassQL TSV exports with the `# PostMN MassQL export` metadata preamble are supported directly.

## Matching and neutral masses

For each selected adduct, neutral mass is calculated as:

`neutral mass = observed m/z × |charge| − total ion mass shift`

Mass tolerance and RT-window fields accept a period as the decimal separator, e.g. `10.0` and `0.20`. Multiple adduct hypotheses can be selected simultaneously, and the best adduct pair is reported for each matching feature pair. Cross-polarity matches are candidate assignments, not proof that two ions are the same compound.

## Optional edge Δm tagging

The edge option compares the absolute precursor m/z difference between the two nodes connected by each existing network edge. For a target of `42.047` Da and allowed error `0.001` Da, edges within 42.046–42.048 Da are tagged. It does not search fragment-ion differences or neutral losses. Results are stored as GraphML edge attributes and included in the edge-table CSV.

## Outputs

- `cpne_cross_polarity_matches.csv`
- `cpne_enriched_<polarity>_network.graphml`
- `cpne_cytoscape_style.xml` (companion style, if needed)
- `cpne_network_edge_table.csv`
- `cpne_run_summary.json`
- A ZIP bundle containing the output files

The GraphML includes a human-friendly node attribute **`Matched cross polarities`** (`Yes`/`No`) and machine-readable `CPNE_` attributes. Cytoscape may retain its current visual style, so map that attribute to node border color in the Cytoscape Style panel if desired.


## User guide

The short guide is available as `HOW_TO_USE.md` and can also be opened inside the app using the **How to use CPNE** button.

## Deploy online

See [`DEPLOY_TO_STREAMLIT_CLOUD.md`](DEPLOY_TO_STREAMLIT_CLOUD.md) for steps to publish a shareable app using GitHub and Streamlit Community Cloud. Before making the app public, remember that uploads are processed on the hosting server and that public repository contents are visible to everyone.
