# How to use CPNE

**Cross-Polarity Network Enricher (CPNE)** links candidate features observed in positive and negative LC–MS ionization modes, then adds cross-polarity and optional PostMN MassQL information to a molecular network.

## Quick start

1. Click **Load example files** to preload the demo task IDs and settings. CPNE tries GNPS2 first. If that is unavailable, it uses a saved task snapshot if one exists, otherwise a clearly labelled synthetic offline example.
2. Choose the input route: **Upload files** or **Enter GNPS2 task IDs**.
3. Choose the target network. For example, **Positive network ← Negative data** keeps the positive network and adds information from matched negative-mode features.
4. Select the adduct hypotheses for both polarities. Custom adducts can be entered using an ion mass shift and charge magnitude.
5. Set the neutral-mass tolerance (default **10 ppm**) and retention-time window (default **0.20 min**). Retention-time compatibility is always required: candidate matches need RT values in both modes and must fall within the selected window. Features without an RT value cannot be assigned.
6. Optionally enable **edge tagging** to mark existing network edges whose absolute precursor m/z difference is within your target Δm and tolerance.
7. Optionally upload PostMN MassQL Full Table `.tsv` exports. Select/confirm the scan or node ID column and the query-validation column for each polarity.
8. Click **Run matching and enrich network**. Inspect the candidate match table and run summary before interpreting matches.
9. Download the enriched GraphML and related CSV/JSON/ZIP outputs. In Cytoscape, use the node attribute **Matched cross polarities** to map matched nodes to a border color. The GraphML data columns are more portable than imported visual styling.

## Input files

### Upload files

Provide positive-mode and negative-mode feature/quantification tables in CSV, TSV or TXT format, plus the GraphML network you want to retain and enrich. The app recognizes common columns such as `row ID`, `row m/z`, `row retention time`, `cluster index`, `parent mass` and `RTMean`.

### GNPS2 task IDs

Enter the positive- and negative-mode task IDs and click **Fetch GNPS2 task results**. This route works with supported public task outputs. Private tasks, unfinished jobs or changed GNPS2 output structures may require the file-upload route instead.

### PostMN MassQL

Export the **Full Table** from PostMN MassQL and upload the TSV for the corresponding polarity. CPNE understands the `# PostMN MassQL export` metadata preamble. Check the selected ID and validation columns in the preview; scan IDs and network node/cluster IDs are not interchangeable in every workflow, so confirm the mapping for your data.

Create PostMN MassQL results at: <https://apps.gnps2.org/postmn_massql>

## Understanding the matching

CPNE calculates candidate neutral masses from each observed m/z and each selected adduct hypothesis, taking the ion charge into account. It compares neutral-mass candidates within the ppm tolerance and always filters by retention-time difference. Both features must have RT values, and their RT difference must be within the selected window. A match is a candidate assignment—not proof that two features are the same compound. Isomers, coelution, in-source fragments and incorrect adduct hypotheses can produce ambiguous matches.

Edge tagging uses the **absolute difference between precursor m/z values of nodes connected by an existing edge**. It does not search MS/MS fragment differences or neutral losses.

## Important notes

- Review ambiguous or one-to-many matches before using them for chemical conclusions.
- For RT matching to be meaningful, both polarities should have comparable chromatographic conditions and RT scales.
- If the app is hosted online, uploaded files are sent to and processed by the hosting server. Do not upload confidential or restricted data to a public demo.
- The app's runtime-generated GNPS2 cache is a convenience, not durable cloud storage. Keep a separate copy of any task snapshot you need to preserve.
