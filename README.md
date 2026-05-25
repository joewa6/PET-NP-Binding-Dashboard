# PET Contaminant Screening Dashboard

Streamlit dashboard for exploring MD simulation results of environmental
contaminants binding to PET nanoplastics.

> Binding metrics, occupancy values, and system labels in this repository are
> scrambled/anonymized. Molecular structures are retained so the 2D/3D viewers
> work as a realistic dashboard showcase.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Regenerate Anonymized Data

From the repository root:

```bash
REAL_DATA_DIR=/path/to/real/data/_50_ python scramble_data.py
```
