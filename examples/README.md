# Example predictions

`demo_overconfident.csv` is synthetic: a 4-class model whose scores were sharpened on purpose (temperature 0.55),
so it is overconfident. `demo_binary_underconfident.csv` is a synthetic binary model whose scores were flattened.

```bash
.venv/bin/python -m calibrationcard grade examples/demo_overconfident.csv --html report.html
```
