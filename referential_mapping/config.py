"""
config.py — Tous les seuils et paramètres configurables du pipeline.
"""
from pathlib import Path

# Racines
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = DATA_DIR / "outputs"

# --- Étape 2 : Similarité sémantique (titres) ---
EMBEDDING_MODEL     = "all-MiniLM-L6-v2"   # modèle local sentence-transformers
SEMANTIC_THRESHOLD  = 0.40   # score cosinus min pour conserver une paire (large pour ne rien perdre)
TOP_K               = 5      # nb candidats Ref B max par exigence Ref A

# --- Étape 4 : LLM ---
LLM_MODEL               = "gpt-4o-mini"
LLM_MAX_RETRIES         = 2
LLM_CONFIRM_THRESHOLD   = 0.75   # score LLM min pour déclencher la double vérification
LLM_CONCURRENCY         = 5      # nb appels LLM en parallèle (asyncio semaphore)

# --- Chemins de sortie ---
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
