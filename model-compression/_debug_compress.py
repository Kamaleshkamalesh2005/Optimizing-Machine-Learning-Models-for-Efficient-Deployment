from pathlib import Path
from src.compress import CompressionEngine

engine = CompressionEngine(
    upload_path=Path('uploads/original_model.pt'),
    compressed_dir=Path('compressed'),
    results_path=Path('config/results.json'),
)
print('load')
engine.load_original_model()
print('quant')
print(engine.run_dynamic_quantization())
print('prune')
print(engine.run_pruning())
print('distill')
print(engine.run_distillation())
print('select')
print(engine.select_and_save_best().keys())
print('done')
