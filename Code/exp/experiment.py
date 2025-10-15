
import schedulefree
from loss.me import ME
from loss.balanced import focal_loss, BalancedSoftmaxLoss
from loss.lambdarank import LambdaNDCGLoss2
from models.standardmodel import NodeGNN, GraphGNN, TAM
from loss.standard import neg_pr_auc_pytorch, cross_entropy_with_logits
from train.metrics import pr_auc_score
import optuna
import torch 
from train.trainer import Trainer, evaluate, predict, EarlyStopper
from graph.loader import CachedSubgraphLoader, CachedGraphLoader, BatLoader
import copy 
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
import sys, json 
import random
import numpy as np 


def set_deterministic_mode(enable=True, seed=42):
    """
    Configure PyTorch for reproducible or performance-oriented runs.

    Args:
        enable (bool): 
            True → deterministic & reproducible results.
            False → allow randomness for faster performance.
        seed (int): random seed (only used if enable=True)
    """
    if enable:
        print(f"[Seed fixed at {seed}] Running in DETERMINISTIC mode.")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("[Random mode] Running in PERFORMANCE mode.")
        torch.seed()
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    
    
def get_loss_function(loss_name, trial=None, best_params=None):
    # cross entropy
    if loss_name == "ce":
        return lambda inputs, targets : cross_entropy_with_logits(inputs, targets)
    # focal loss 
    elif loss_name == "focal":
        if trial is not None:
            alpha = trial.suggest_float("alpha", 1, 30) # the weights of class 1 (minority)
            gamma = trial.suggest_float("gamma", 1.0, 5.0)
        elif best_params is not None:
            alpha = best_params["alpha"]
            gamma = best_params["gamma"]
        else:
            raise ValueError("Provide either trial or best_params")
        return  lambda inputs, targets : focal_loss(inputs, targets, alpha, gamma)
    # lambda rank loss
    elif loss_name == "lambda":
        if trial is not None:
            sigma = trial.suggest_float("sigma", 0.5, 2.0, log=True)
        elif best_params is not None:
            sigma = best_params["sigma"]
        else:
            raise ValueError("Provide either trial or best_params")
        return LambdaNDCGLoss2(sigma=sigma)
    # me 
    elif loss_name == "me":
        if trial is not None:
            balancing_factor = trial.suggest_float("balancing_factor", 0.1, 0.9)
        elif best_params is not None:
            balancing_factor = best_params["balancing_factor"]
        else:
            raise ValueError("Provide either trial or best_params")
        return ME(balancing_factor=balancing_factor)
    # balanced softmax 
    elif loss_name == 'balancedsoftmax':
        return BalancedSoftmaxLoss()
    else:
        raise ValueError(f"Unsupported loss: {loss_name}")
    

def objective(trial, data_train, data_val, model_inst, parameters, layer, edge_type, output_channels, train_loss_f_name,
            tuning_epochs, tuning_standard_loss_fn, device, warmup, graph_type):
    params_dict = {}
    params_dict['output_channels'] = output_channels
    if layer is not None:
        params_dict['layer'] = layer
    if edge_type is not None:
        params_dict['edge_type'] = edge_type
    for p in parameters:
        if 'channels' in p:
            params_dict[p] = trial.suggest_int(p,  8, 64, 4)
        elif 'dropout' in p:
            params_dict[p] = trial.suggest_float(p, 0, 0.5)
        # special tuning params for tam
        elif 'temp' in p:
            params_dict[p] = trial.suggest_float(p, 0.1, 3.0)
        elif 'tam' in p:
            params_dict[p] = trial.suggest_float(p, 0.1, 3.0)
    if model_inst == TAM:
        params_dict['warmup'] =  warmup
        params_dict['include_mask'] =  False
    model = model_inst(**params_dict)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr)
    # tune on loss functions 
    loss_fn = get_loss_function(train_loss_f_name, trial)
    #  Training of the model.
    trainer = Trainer(data_train, loss_fn, device)
    if_tam = isinstance(model, TAM)
    for epoch in range(tuning_epochs): 
        if if_tam:
            model.reset_stage(epoch)
        trainer.train(model, optimizer, epoch)
        # Validation of the model.
        if if_tam:
            val_loss = evaluate(model.base_model, optimizer, data_val, device, loss_fn = tuning_standard_loss_fn, graph_type = graph_type)
        else:
            val_loss = evaluate(model, optimizer, data_val, device, loss_fn = tuning_standard_loss_fn, graph_type = graph_type)
        trial.report(val_loss, epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    del model, optimizer, loss_fn
    torch.cuda.empty_cache()
    return val_loss


class Experiment:
    def __init__(self, graph_setting, early_stop_params, optuna_params, training_params, tuning_standard_loss_fn) -> None:
        self.path = graph_setting['path']
        self.save_path = graph_setting['save_path'] 
        self.random_state = graph_setting['random_state'] 
        self.k_hops = graph_setting['k_hops']  
        self.sample_neighbors_nr = graph_setting['sample_neighbors_nr'] 
        self.batch_size = graph_setting['batch_size']
        self.repeat_pos = graph_setting['repeat_pos'] 
        self.group_size = graph_setting['group_size'] 
        self.rst = {}
        self.graph_data = [ torch.load(self.path[i] +'graph.pt', weights_only = False) for i in range(3)]
        
        self.device = training_params['device']
        self.seed = training_params['seed']
        self.epoches = training_params['epoches']
        self.early_stop_params = early_stop_params
        self.optuna_params = optuna_params
        self.tuning_standard_loss_fn = tuning_standard_loss_fn

    def load_graph(self, graph_type, warmup = None):
        for g in self.graph_data:
            g.y = (g.y == 0).long() # change the label to binary 
        if graph_type == 'graph':
            data_train = copy.deepcopy(self.graph_data[0])
            data_val = copy.deepcopy(self.graph_data[1])
            data_test = copy.deepcopy(self.graph_data[2])
        elif graph_type == 'graph_weight':
            data_train = CachedGraphLoader(self.graph_data[0], self.batch_size, self.k_hops, self.sample_neighbors_nr, input_nodes=None, by='weights', random_state = self.random_state)
            data_val = copy.deepcopy(self.graph_data[1])
            data_test = copy.deepcopy(self.graph_data[2])

        elif graph_type == 'subgraph':
            data_train =  CachedSubgraphLoader(self.graph_data[0], self.k_hops, self.batch_size, None, self.sample_neighbors_nr, self.repeat_pos, self.group_size, self.random_state)
            data_val =  CachedSubgraphLoader(self.graph_data[1], self.k_hops, self.batch_size, None, self.sample_neighbors_nr, self.repeat_pos, self.group_size, self.random_state)
            data_test=  CachedSubgraphLoader(self.graph_data[2], self.k_hops, self.batch_size, None, self.sample_neighbors_nr, self.repeat_pos, self.group_size, self.random_state)
        elif graph_type == 'subgraph_rank':
            data_train =  CachedSubgraphLoader(self.graph_data[0], self.k_hops, self.batch_size, 'qids', self.sample_neighbors_nr, self.repeat_pos, self.group_size, self.random_state)
            data_val =  CachedSubgraphLoader(self.graph_data[1], self.k_hops, self.batch_size, None, self.sample_neighbors_nr, self.repeat_pos, self.group_size, self.random_state)
            data_test=  CachedSubgraphLoader(self.graph_data[2], self.k_hops, self.batch_size, None, self.sample_neighbors_nr, self.repeat_pos, self.group_size, self.random_state)
        
        elif graph_type == 'subgraph_weight':
            data_train =  CachedSubgraphLoader(self.graph_data[0], self.k_hops, self.batch_size, 'weights', self.sample_neighbors_nr, self.random_state)
            data_val =  CachedSubgraphLoader(self.graph_data[1], self.k_hops, self.batch_size, None, self.sample_neighbors_nr, self.random_state)
            data_test=  CachedSubgraphLoader(self.graph_data[2], self.k_hops, self.batch_size, None, self.sample_neighbors_nr, self.random_state)

        elif graph_type == 'graph_bat':
            data_train =  BatLoader(self.graph_data[0].to(self.device), include_mask = False, warmup=warmup)
            data_val =  copy.deepcopy(self.graph_data[1])
            data_test = copy.deepcopy(self.graph_data[2])
            
        elif graph_type == 'graph_tam':
            data_train = copy.deepcopy(self.graph_data[0])
            data_val = copy.deepcopy(self.graph_data[1])
            data_test = copy.deepcopy(self.graph_data[2])
        return data_train, data_val, data_test
    
    def run(self, k, v, save =True):
        hyperparameters, loss_fn_name, layer, edge_type, output_channels, graph_type, warmup =  v["hyperparameters"], v["loss_fn_name"],\
                                                                                         v["layer"],v["edge_type"], v["output_channels"], \
                                                                                            v['graph_type'],  v['warmup']
        data_train, data_val, data_test = self.load_graph(graph_type, warmup = warmup)
        
        study = optuna.create_study(direction="minimize")
        if graph_type in ['graph', 'graph_weight', 'graph_bat']:
            model_inst = NodeGNN
        elif graph_type in ['subgraph_rank', 'subgraph', 'subgraph_weight']:
            model_inst = GraphGNN 
        elif graph_type == 'graph_tam':
            model_inst = TAM
        set_deterministic_mode(enable=False, seed=None) # make sure this is random 
        study.optimize(lambda trial: objective(trial, data_train, data_val, model_inst,
                                                hyperparameters, 
                                                layer, edge_type,
                                                output_channels, 
                                                loss_fn_name,
                                                self.optuna_params['tuning_epoches'], 
                                                self.tuning_standard_loss_fn,
                                                self.device,
                                                warmup,
                                                graph_type), 
                    n_trials=self.optuna_params['tuning_trails'], timeout=None) 
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        best_param = trial.params
        if model_inst == NodeGNN: 
            model = model_inst( layer, edge_type,
                                best_param["hidden_channels"], 
                                best_param["dropout"],
                                output_channels
                                ).to(self.device)
        elif model_inst == GraphGNN: 
            model = model_inst( layer, edge_type,
                                best_param["internal_hidden_channels"], 
                                best_param["internal_dropout"],
                                best_param["output_hidden_channels"], 
                                best_param["output_dropout"],
                                output_channels
                                ).to(self.device)
        elif model_inst == TAM: 
            model = model_inst( layer, edge_type,
                                best_param["hidden_channels"], 
                                best_param["dropout"],
                                output_channels, 
                                best_param["temp_phi"],  
                                best_param["temp_gamma"],  
                                best_param["tam_alpha"], 
                                best_param["tam_beta"],
                                include_mask=False,
                                warmup=warmup
                                ).to(self.device)
        set_deterministic_mode(enable=True, seed=self.seed) # make sure this is not random 
        optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=best_param['lr'])  
        # early stop 
        earlystopper = None
        if self.early_stop_params is not None:
            self.early_stop_params.update({'model_name': k})
            earlystopper = EarlyStopper(**self.early_stop_params)
        loss_fn = get_loss_function(loss_fn_name, None,
                                    best_param) 
        trainer = Trainer(data_train, loss_fn, self.device)
        if_tam = isinstance(model, TAM)
        for e in range(self.epoches):
            if if_tam:
                model.reset_stage(e)
            train_loss = trainer.train(model, optimizer, e)
            if earlystopper is not None:
                if if_tam:
                    val_loss = evaluate(model.base_model, optimizer, data_val, self.device, self.tuning_standard_loss_fn, graph_type = graph_type)
                else:
                    val_loss = evaluate(model, optimizer, data_val, self.device, self.tuning_standard_loss_fn,graph_type = graph_type)
                if_stop = earlystopper.early_stop( val_loss, best_model =  model)
                if if_stop:
                    print('early stop at epoches: ', e, ', stop val loss: ', val_loss, ' min val loss: ', earlystopper.min_validation_loss)
                    break 
        if if_tam:
            y_pred, y_true = predict(model.base_model, optimizer, data_test, self.device, graph_type)
        else:
            y_pred, y_true = predict(model, optimizer, data_test, self.device, graph_type)
        self.rst[k] = (y_pred, y_true, best_param, model)
        if save:
            torch.save(self.rst[k], self.save_path + k + '.pt')
        return  self.rst


if __name__ == '__main__':
    tuning_standard_loss_fn = neg_pr_auc_pytorch
    layer_dict = {'GCNConv': GCNConv, 
                  'GATConv': GATConv, 
                  'SAGEConv': SAGEConv
                  }
    # Read JSON-formatted arguments
    input_args = json.loads(sys.argv[1])

    # Extract mlp_settings, early_stop_params, optuna_params, training_params
    graph_settings  =  input_args['graph_settings']
    early_stop_params  =  input_args['early_stop_params']
    optuna_params  =  input_args['optuna_params']
    training_params  =  input_args['training_params']
    parameter_k = input_args['parameter_k']
    parameter_v = input_args['parameter_v']
    parameter_v['layer'] = layer_dict[parameter_v['layer']]

    # Run experiment
    runner = Experiment(graph_settings,
                     early_stop_params,
                     optuna_params,
                     training_params, 
                     tuning_standard_loss_fn
                    )
    runner.run(parameter_k, parameter_v)



    