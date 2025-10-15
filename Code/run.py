import copy 
import subprocess
import json 
import os 
from pathlib import Path
import sys 
from datetime import datetime

_, dataset, modelname = sys.argv

print(f"MODENAME: {modelname}")

# Define parameters
root_path =  f'../Dataset/{dataset}/'
write_type = ['train', 'val', 'test'] 
path = [root_path + i + '/' for i in write_type ]
save_path =  f'../Result/{dataset}/pred/'

graph_settings = {'path': path, 
                  'save_path':save_path,
                  'random_state': 42,  # for subgraph seed! 
                  'k_hops': 3, 
                  'sample_neighbors_nr': 5, 
                  'batch_size':1024 * 16,  
                  'repeat_pos': 5, 
                  'group_size': 3
                  }

early_stop_params  = { 'patience': 30, 
                        'save_best_path': f'../Result/{dataset}/val/'}

optuna_params = {'tuning_epoches': 100,
                 'tuning_trails': 50}

training_params = {
                   'device': 'cuda', 
                   'epoches': 3000,
                   'seed' : 42 # for training seed! 
                   }


def gnn(hyperparameters, graph_settings, early_stop_params, optuna_params, training_params):
    # run loss function
    for k, v in hyperparameters.items():
        params = {}
        params['graph_settings'] = copy.deepcopy(graph_settings)
        params['parameter_k'] = k
        params['parameter_v'] = v
        params['early_stop_params'] = early_stop_params
        params['optuna_params'] = optuna_params
        params['training_params'] = training_params

        print(f"\n[INFO] Running experiment {k}")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent)
        cmd = ['python',  'exp/experiment.py', json.dumps(params)]
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(f"[ERROR] Experiment {k} failed!")
        else:
            print(f"[DONE] Experiment {k} finished.")

class SimpleLogger:
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.terminal = sys.stdout
        # Open file in append mode
        self.log = open(self.log_file, "a", encoding="utf-8")
        self._write(f"\n===== New Run {dataset} Started at {datetime.now()} =====\n")

    def _write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # write immediately

    def write(self, message):
        self._write(message)

    def flush(self):
        # Needed for compatibility with sys.stdout
        pass

    def close(self):
        self._write(f"===== Run Finished at {datetime.now()} =====\n\n")
        self.log.close()

if __name__ == '__main__':
    log_file = "terminal.log"
    logger = SimpleLogger(log_file)
    # Redirect all prints to logger
    sys.stdout = logger
    sys.stderr = logger  # optional: capture errors too
    layer_type = {
        'SAGEConv' : 'nan',
        'GATConv' : 'attr',
        'GCNConv' : 'weight' }
    
    configerations = {}
    for layer, edge_type in layer_type.items():
        hyperparameters = {
                        # whole graph : nodes 
                        'ngnn_ce_' + layer[:3].lower() : {
                               'hyperparameters' : ['hidden_channels','dropout'],
                               'loss_fn_name' : 'ce',
                               'layer': layer,
                               'edge_type' : edge_type, 
                               'output_channels': 2,
                               'graph_type' : 'graph',
                               'warmup' : None
                               },
                        'ngnn_focal_' + layer[:3].lower() : {
                                'hyperparameters' : ['hidden_channels','dropout'],
                                'loss_fn_name' : 'focal',
                                'layer': layer,
                                'edge_type' : edge_type, 
                                'output_channels': 1, 
                                'graph_type' : 'graph',
                                'warmup' : None
                                },
                        'ngnn_me_' + layer[:3].lower() : {
                                'hyperparameters' : ['hidden_channels','dropout'],
                                'loss_fn_name' : 'me',
                                'layer': layer,
                                'edge_type' : edge_type, 
                                'output_channels': 2,
                                'graph_type' : 'graph',
                                'warmup' : None
                                },
                        'ngnn_balancedsoftmax_' + layer[:3].lower() : {
                                'hyperparameters' : ['hidden_channels','dropout'],
                                'loss_fn_name' : 'balancedsoftmax',
                                'layer': layer,
                                'edge_type' : edge_type, 
                                'output_channels': 2,
                                'graph_type' : 'graph',
                                'warmup' : None
                                },
                        'ngnn_tam_balancedsoftmax_' + layer[:3].lower() : {
                                'hyperparameters' : ['hidden_channels','dropout', 'temp_phi', 'temp_gamma', 'tam_alpha', 'tam_beta'],
                                'loss_fn_name' : 'balancedsoftmax',
                                'layer': layer,
                                'edge_type' : edge_type, 
                                'output_channels': 2,
                                'graph_type' : 'graph_tam',
                                'warmup' : 15
                                },
                        'ngnn_tam_ce_' + layer[:3].lower() : {
                                'hyperparameters' : ['hidden_channels','dropout', 'temp_phi', 'temp_gamma', 'tam_alpha', 'tam_beta'],
                                'loss_fn_name' : 'ce',
                                'layer': layer,
                                'edge_type' : edge_type, 
                                'output_channels': 2,
                                'graph_type' : 'graph_tam',
                                'warmup' : 15
                                },
                        'ngnn_bat_ce_' + layer[:3].lower() : {
                                'hyperparameters' : ['hidden_channels','dropout'],
                                'loss_fn_name' : 'ce',
                                'layer': layer,
                                'edge_type' : edge_type, 
                                'output_channels': 2,
                                'graph_type' : 'graph_bat',
                                'warmup' : 15
                                },
                        # whole graph : weighted nodes 
                        'ngnn_weighted_ce_' + layer[:3].lower() : {
                                'hyperparameters' : ['hidden_channels','dropout'],
                                'loss_fn_name' : 'ce',
                                'layer': layer,
                                'edge_type' : edge_type, 
                                'output_channels': 2,
                                'graph_type' : 'graph_weight',
                                'warmup' : None
                                },
                        # subgraph method 
                        'ggnn_ce_' + layer[:3].lower() : {
                                        'hyperparameters'  : ['internal_hidden_channels','internal_dropout', 'output_hidden_channels', 'output_dropout'],
                                        'loss_fn_name' : 'ce',
                                        'layer': layer,
                                        'edge_type' : edge_type, 
                                        'output_channels': 2,
                                        'graph_type' : 'subgraph',
                                        'warmup' : None
                                        },

                        'ggnn_focal_' + layer[:3].lower() : {
                                        'hyperparameters'  : ['internal_hidden_channels','internal_dropout', 'output_hidden_channels', 'output_dropout'],
                                        'loss_fn_name' : 'focal',
                                        'layer': layer,
                                        'edge_type' : edge_type, 
                                        'output_channels': 1,
                                        'graph_type' : 'subgraph',
                                        'warmup' : None
                                        },
                        'ggnn_me_' + layer[:3].lower() : {
                                        'hyperparameters'  : ['internal_hidden_channels','internal_dropout', 'output_hidden_channels', 'output_dropout'],
                                        'loss_fn_name' : 'me',
                                        'layer': layer,
                                        'edge_type' : edge_type, 
                                        'output_channels': 2,
                                        'graph_type' : 'subgraph',
                                        'warmup' : None
                                        },
                       
                        'ggnn_lambda_' + layer[:3].lower() : {
                                        'hyperparameters'  : ['internal_hidden_channels','internal_dropout', 'output_hidden_channels', 'output_dropout'],
                                        'loss_fn_name' : 'lambda',
                                        'layer': layer,
                                        'edge_type' : edge_type, 
                                        'output_channels': 1,
                                        'graph_type' : 'subgraph_rank',
                                        'warmup' : None
                                        },
                          'ggnn_balancedsoftmax_' + layer[:3].lower() : {
                                       'hyperparameters'  : ['internal_hidden_channels','internal_dropout', 'output_hidden_channels', 'output_dropout'],
                                       'loss_fn_name' : 'balancedsoftmax',
                                       'layer': layer,
                                       'edge_type' : edge_type, 
                                       'output_channels': 2,
                                       'graph_type' : 'subgraph',
                                       'warmup' : None
                                       },
                        # subgraph: weighted samples 
                        'ggnn_weighted_ce_' + layer[:3].lower() : {
                                        'hyperparameters'  : ['internal_hidden_channels','internal_dropout', 'output_hidden_channels', 'output_dropout'],
                                        'loss_fn_name' : 'ce',
                                        'layer': layer,
                                        'edge_type' : edge_type, 
                                        'output_channels': 2,
                                        'graph_type' : 'subgraph_weight',
                                        'warmup' : None
                                        }
                        }
        configerations.update(hyperparameters)
    new_hyperparameters = {modelname: configerations[modelname]}

    print(new_hyperparameters)
    gnn(new_hyperparameters, graph_settings, early_stop_params, optuna_params, training_params)
    logger.close()