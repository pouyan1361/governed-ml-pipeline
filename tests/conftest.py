import sys
from pathlib import Path

# Make "from src import ..." work no matter where pytest is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
