import yaml
from model.autoformer import Autoformer
from utils.visualizations import visual_results


with open('configuration.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

autoformer_model = Autoformer(**config['model_structure'])
autoformer_model.train_model(**config['train_settings'])

autoformer_model = Autoformer(**config['model_structure'])
previous, actuals, results = autoformer_model.test_model(**config['test_settings'])

visual_results(previous, actuals, results, 1000, 1300, save_path='/autoformer/results/')
