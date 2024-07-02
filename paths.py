import os
ROOT = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(ROOT, 'output')

os.makedirs(OUTPUT_DIR, exist_ok=True)