cat > pyproject.toml << 'EOF'
[project]
name = "gbm-longitudinal-toolkit"
version = "0.1.0"
description = "Longitudinal radiomics framework for GBM treatment response prediction"
requires-python = ">=3.12"

[tool.ruff]
line-length = 88

[tool.mypy]
python_version = "3.12"
strict = false
EOF