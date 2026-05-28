# Name Corpus Attribution

The `firstnames_raw.csv` file in this directory is a verbatim copy of the
[firstname-database](https://github.com/MatthiasWinkelmann/firstname-database)
dataset.

## Credits

- **Original creators (2007–2008):** Jörg Michael
- **Updated version (2016+):** Matthias Winkelmann

## License

This dataset is distributed under the **GNU Free Documentation License (GFDL),
Version 1.2 or later**. The full license text is reproduced in
`LICENSE-firstname-database.txt`.

The derived file `training_corpus.csv` — produced by `build_corpus.py` from
the raw file plus the curated FAIMR seed lists — is a modification of the
original work and therefore inherits the same GFDL terms. Re-distribution
of `training_corpus.csv` must preserve this attribution and the GFDL notice.

The trained classifier (`fairness/names/model.pkl`) consists of numerical
weights learned from this data. Under prevailing case law and common
interpretation, model parameters are not considered derivative works of
training data, so the model artifact and inference code (`classifier.py`)
remain under the FAIMR project's main licence (see top-level `LICENSE`).

## Coverage

The upstream dataset covers ~46k unique first names across 55+ countries with
per-country frequencies and a gender label (M, F, ?M, ?F, ?, =). FAIMR uses
this to provide calibrated `P(gender | name)` probabilities for the bias
detector — see `fairness/names/model_card.json` for per-culture accuracy
breakdowns of the trained classifier.

## How to refresh

To re-download the upstream raw file:

```
curl -sL -A "Mozilla/5.0" -o data/names/firstnames_raw.csv \
    https://raw.githubusercontent.com/MatthiasWinkelmann/firstname-database/master/firstnames.csv
python data/names/build_corpus.py
python fairness/names/train_classifier.py
```
