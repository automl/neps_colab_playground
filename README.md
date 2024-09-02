# NePS tutorial DL2024
NePS wrapper code specific to DL2024 tutorial


# Setup
```bash
pip install -r requirements.txt
```

# Simple run
```python
# assuming current working directoy is `neps_tutorial_DL2024/`

from train import run_pipeline_demo
from utils import set_seeds


set_seeds(42)
run_pipeline_demo()
```