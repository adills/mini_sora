import sys
from pathlib import Path
import os

# Add project root to sys.path so `import mini_sora` works no matter where pytest is run
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MINI_SORA_TEST_MODE", "1")  # Enable test mode by default for all tests