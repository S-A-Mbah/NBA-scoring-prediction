# NBA Next-Season Scoring Prediction Dashboard

Predicts each NBA player's points per game next season from this season's stats.

## Run it (Docker)

1. Start Docker Desktop.
2. In the project folder:

   ```bash
   docker compose up --build
   ```

3. Open http://localhost:8501. Stop with `Ctrl+C`.

## Run it (without Docker)

Needs Python 3.12+.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app/dashboard.py
```

On macOS/Linux, activate with `source .venv/bin/activate`.

## Rebuild the data (optional)

The cleaned data is already included. Only needed if you change the cleaning code:

```bash
python -m src.collect_data
python -m src.data_processor
```

## Tests

```bash
pytest -q
```
