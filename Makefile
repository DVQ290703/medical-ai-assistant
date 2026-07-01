.PHONY: data train eval serve test lint clean

data:   ; bash scripts/build_data.sh
train:  ; bash scripts/train.sh
eval:   ; bash scripts/evaluate.sh
serve:  ; bash scripts/serve.sh
test:   ; pytest -q
lint:   ; ruff check src tests
clean:  ; find . -type d -name __pycache__ -exec rm -rf {} +
