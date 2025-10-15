from graph.loader import BatLoader, CachedGraphLoader, CachedSubgraphLoader
from torch_geometric.data import Data
from torch_geometric.loader import ImbalancedSampler
import torch 
from loss.lambdarank import LambdaNDCGLoss2
from models.standardmodel import TAM, NodeGNN

class Trainer:
    def __init__(self, data, loss_fn, device):
        self.loss_fn = loss_fn
        self.device = device 

        if isinstance(data, Data):
            self.training_type = 'graph'
            self.data = data 
            
        elif isinstance(data, CachedGraphLoader) or isinstance(data, CachedSubgraphLoader) or isinstance(data, BatLoader):
            self.loader = data
            self.training_type = self.loader.graph_property
        else:
            print(f'select data type {type(data)} is not supported')
            raise KeyError 

    
    def graph_train(self, model, optimizer, epoch):
        self.data.to(self.device) 
        model.to(self.device) 
        model.train()
        optimizer.train()
        optimizer.zero_grad()
        if isinstance(model, TAM):
            y_pred = model(self.data)
        else:
            y_pred = model(self.data.x, self.data.edge_index, self.data.edge_attr)
        if hasattr(self.data, 'mask') or hasattr(self.data, 'train_mask'):
            loss =  self.loss_fn(y_pred[self.data.mask], self.data.y[self.data.mask].float())
        else:
            loss =  self.loss_fn(y_pred, self.data.y.float())
        loss.backward()
        optimizer.step()
        if epoch % 50 ==0:
            print(f"Epoch {epoch} training loss {loss.detach()}")
        return loss.detach()
    
    def batch_train(self, model, optimizer, epoch):
        loss = None
        loss_total = 0
        instance_nr = 0
        model.to(self.device) 
        model.train()
        optimizer.train()
        optimizer.zero_grad()
        for batch in self.loader:
            if isinstance(batch, (list, tuple)):
                batch = [t.to(self.device,  non_blocking=True) for t in batch]
            else:
                batch.to(self.device, non_blocking=True)
            if isinstance(model, NodeGNN):
                y_pred = model(batch.x, batch.edge_index, batch.edge_attr)
            else:
                y_pred = model(batch)
            if isinstance(self.loader.sampler, ImbalancedSampler): 
                mask = (batch.mask & torch.isin(batch.n_id, batch.input_id)) # weight sampling and only counts the balanced input 
            else:
                mask  = batch.mask
            if isinstance(self.loss_fn, LambdaNDCGLoss2):
                score = y_pred[mask].reshape(-1).unsqueeze(0).to(self.device).view(-1, batch.group_size)
                relevance = batch.y[mask].unsqueeze(0).to(self.device).view(-1, batch.group_size)
                n = torch.ones(batch.batch.max()+1).view(-1, batch.group_size).sum(dim=1).view(-1).to(self.device)
                loss = self.loss_fn(score, relevance, n=n).mean()
                loss_total += loss.detach() * score.shape[0]
            else:
                loss = self.loss_fn(y_pred[mask], batch.y[mask].float())
                loss_total += loss.detach()*len(y_pred[mask])
            instance_nr += len(y_pred[mask])
            loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if epoch % 50 ==0:
            print(f"Epoch {epoch} training loss {(loss_total/instance_nr)}")
        return loss_total/instance_nr 

    def augment_train(self, model, optimizer, epoch):
        data = self.loader.augment(model, epoch)
        data.to(self.device) 
        model.to(self.device) 
        model.train()
        optimizer.train()
        optimizer.zero_grad()
        y_pred = model(data.x, data.edge_index, data.edge_attr)
        if hasattr(data, 'mask') or hasattr(data, 'train_mask'):
            loss =  self.loss_fn(y_pred[data.mask], data.y[data.mask].float())
        else:
            loss =  self.loss_fn(y_pred, data.y.float())
        loss.backward()
        optimizer.step()
        if epoch % 50 ==0:
            print(f"Epoch {epoch} training loss {loss.detach()}")
        return loss.detach()
    
    def train(self, model, optimizer, epoch):
        if self.training_type == 'graph':
            self.graph_train(model, optimizer, epoch)
        elif self.training_type == 'batch':
            self.batch_train(model, optimizer, epoch)
        elif self.training_type == 'augment':
            self.augment_train(model, optimizer, epoch)
        else:
            raise  ValueError(f"Unsupported traing type: {self.training_type}")

@torch.no_grad()
def evaluate( model, optimizer, data, device, loss_fn, graph_type):
    model.to(device)
    model.eval()
    optimizer.eval()
    if graph_type not in ['subgraph', 'subgraph_rank', 'subgraph_weight']:
        data.to(device)
        if isinstance(model, TAM):
            y_pred = model(data)
        else:
            y_pred = model(data.x, data.edge_index, data.edge_attr)
        loss =  loss_fn(y_pred[data.mask], data.y[data.mask].float())
        return loss.detach()
    else:
        loss_total = 0
        instance_nr = 0
        for batch in data:
            if isinstance(batch, (list, tuple)):
                batch = [t.to(device) for t in batch]
            else:
                batch.to(device)
            y_pred = model(batch)
            loss = loss_fn(y_pred[batch.mask], batch.y[batch.mask].float())
            loss_total += loss.detach()*len(y_pred[batch.mask])
            instance_nr += len(y_pred[batch.mask])
        return loss_total/instance_nr 
            

@torch.no_grad()
def predict( model, optimizer, data, device, graph_type):
    model.to(device)
    model.eval()
    optimizer.eval()
    if graph_type not in ['subgraph', 'subgraph_rank', 'subgraph_weight']:
        data.to(device)
        if isinstance(model, TAM):
            y_pred_ls = model(data)[data.mask].cpu().tolist()
        else:
            y_pred_ls = model(data.x, data.edge_index, data.edge_attr)[data.mask].cpu().tolist()
        y_true_ls = data.y[data.mask].cpu().tolist()
    else:
        y_pred_ls = []
        y_true_ls = []
        for batch in data:
            if isinstance(batch, (list, tuple)):
                batch = [t.to(device) for t in batch]
            else:
                batch.to(device)
            y_pred = model(batch)
            y_pred_ls = y_pred_ls + y_pred[batch.mask].cpu().tolist()
            y_true_ls = y_true_ls + batch.y[batch.mask].cpu().tolist()
    return y_pred_ls, y_true_ls 


class EarlyStopper:
    def __init__(self, model_name, patience=1, min_delta=0,  save_best_path = None):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.save_best_path = save_best_path
        self.model_name = model_name

    def early_stop(self, validation_loss, best_model = None):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            if best_model is not None:
                assert self.save_best_path is not None and self.save_best_path != ''
                torch.save(best_model, self.save_best_path + f'/{self.model_name}_best_model_val.pt')
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    