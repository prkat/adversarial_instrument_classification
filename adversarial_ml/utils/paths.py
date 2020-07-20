from pathlib import Path

d_path = 'insert/your/path/here'
misc_path = str(Path(__file__).parent.parent.parent / 'misc')
csv_path = 'insert/your/path/here'

adversary_path = str(Path(misc_path) / 'adversaries')
train_path = str(Path(misc_path) / 'pre_trained')
model_path = str(Path(train_path) / 'models')
log_path = str(Path(train_path) / 'logs')
