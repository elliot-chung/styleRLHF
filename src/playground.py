from .data import Catalog, build_synthetic_outfit_dataset
from .config import CORPUS_CSV
import random

catalog = Catalog()
catalog.load_csv(CORPUS_CSV)
outfits = build_synthetic_outfit_dataset(catalog, 10)

for outfit in outfits:
  print(outfit)