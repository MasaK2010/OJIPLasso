# Simple OJIP Pipeline (Supplementary Release)

This folder contains a compact Python script that reproduces the minimum
data-processing steps requested for the Supplementary Materials. It is
written to be easy to read rather than feature complete.

## Files

- `ojip_lasso_regression.py` – single-file workflow that:
 1. Loads metadata describing sample combinations and AquaPen file names.
 2. Reads AquaPen raw fluorescence text files, applies spline smoothing,
     and exports the smoothed curves.
 3. Calculates a focused set of OJIP variables and lightweight quality
     flags, saving them as CSV (CAP-style table with `Sample_ID` rows).
 4. Splits data into teaching and prediction groups, runs nested
    cross-validation (Lasso regression using a 1-SE rule), and exports a teaching vs.
     prediction scatter figure mimicking Figure 6.
 5. Writes all processed CSV files to `Output/`.

## Cross-platform and runtime notes

- The script is cross-platform (Windows/macOS/Linux). It uses script-relative paths, so you can run it from any current working directory; running from the repository root is recommended.
- A non-interactive Matplotlib backend (Agg) is used, so figures render correctly in headless or CI environments.
- All inputs are read from `CSV/` and all outputs are written to `Output/` under the repository folder.

## Preparation: AquaPen OJIP data and metadata

This project expects AquaPen OJIP data in a strict "per-sheet wide" CSV format and a
metadata CSV that maps numeric MeasurementIDs to sample records.

### Per-sheet wide OJIP CSV (strict)

- The first column header (first cell) must be `index` for OJIP data.
- Subsequent column headers are numeric MeasurementIDs (for example `3`, `8`, `10`, ...),
  which typically corresponds to the 'index' value assigned by the AquaPen instrument.
- The file uses three header rows:
  1. `index` (first cell)
  2. `time` — per-column run timestamps for each measurement
  3. `id` — should contain the literal `OJIP` for each measurement column
- Data rows start on row 4 and contain numeric fluorescence values indexed by microseconds
  (typically starting near 11 µs and ending near 2,001,621 µs).
- If a single experiment contains multiple species or timepoints, split them into separate
  per-sheet CSV files (for example `Cv1.csv`, `Ao1.csv`) and place them under `CSV/`.

Example file (for reference): `CSV/Ao1.csv`.

### Metadata mapping (`CSV/metadata.csv`)

Provide a metadata CSV that maps numeric `MeasurementID` values to sample properties.
Required columns: `Species`, `Condition`, `Replicate`, `TimeLabel`, `Time`, `NH3_mM`, `MeasurementID`, `OJIPFile`.
Optional columns: `Notes`.

`MeasurementID` must be numeric and match one of the measurement column headers in the per-sheet CSV.
Typically, this is the index assigned by the AquaPen instrument. The included `CSV/metadata.csv` is
prefilled with example Cv/Ao samples — update `MeasurementID`, `NH3_mM`, and `OJIPFile` as needed.
`NH3_mM` is the target for regression in this release.

## Usage

1. Ensure your AquaPen per-sheet OJIP CSVs and `CSV/metadata.csv` follow the preparation section above.
2. Adjust `DATA_TEACH` and `DATA_PRED` near the top of the script so they
   describe the exact combinations you want in the teaching and prediction
   groups. Each entry is a dictionary specifying column values to match.
3. (First time only) Install dependencies:

    - Windows (PowerShell):

       ```powershell
       pip install -r requirements.txt
       ```

    - Linux/macOS (bash/zsh):

       ```bash
       python3 -m pip install -r requirements.txt
       ```

    Optional: use a virtual environment

    - Windows (PowerShell):

       ```powershell
       python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
       ```

    - Linux/macOS (bash/zsh):

       ```bash
       python3 -m venv .venv
       source .venv/bin/activate
       python3 -m pip install -r requirements.txt
       ```

4. Run the script from the project root:

    - Windows (PowerShell):

       ```powershell
       python ojip_lasso_regression.py
       ```

    - Linux/macOS (bash/zsh):

       ```bash
       python3 ojip_lasso_regression.py
       ```

Note: You can execute the script from any directory. Because the script resolves paths relative to its own location, running it from the repository root is recommended for clarity.

5. Collect the outputs from `Output/`:
   - `smoothed_curves.csv`
   - `ojip_variables.csv`
   - `ojip_variables_with_metadata.csv`
   - `nh3_predicted_vs_observed_teach_pred.png`

The script intentionally avoids optional arguments or extensive error
handling to keep the logic transparent for readers who are new to Python.
