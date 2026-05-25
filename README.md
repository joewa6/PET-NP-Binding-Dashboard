# PET Contaminant Screening Dashboard

Streamlit dashboard for exploring MD simulation results of environmental
contaminants binding to PET nanoplastics.

> Data in this repository is scrambled/anonymized. It is included only as a
> working dashboard showcase.

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
