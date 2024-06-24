#!python3

# may need to run `./build.sh` from `../..`
# or `source ~/.iris/setup.zsh`

from termcolor import colored
import os
import numpy as np
import pandas as pd
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
import torch_geometric.nn as geom_nn
import torch_geometric.loader as geom_loader
import pytorch_lightning as pl
import pickle
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# Initialize PyTorch environment
DATASET_PATH = "./data"
CHECKPOINT_PATH = "./saved_models"
pl.seed_everything(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
use_cpu = True
if use_cpu:
    num_devices = 1
    device = torch.device("cpu")
    #num_workers = os.cpu_count()
    num_workers = 0
else:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #num_devices = torch.cuda.device_count()
    #num_workers = torch.cuda.device_count()
    num_devices = 1
    num_workers = 10
    #num_workers = 127


# Global variables:
_dataset_directory = 'graphs'
_model_name = "iris_scheduler_gnn_graph_level_classifier"
_batch_size = 1
_root_dir = os.path.join(CHECKPOINT_PATH, "GraphLevel", _model_name)
_model_filename = os.path.join(CHECKPOINT_PATH, f"{_model_name}.pth")
_pretrained_filename = os.path.join(CHECKPOINT_PATH, f"GraphLevel{_model_name}.ckpt")
_pretrained_shape_file = os.path.join(CHECKPOINT_PATH, f"GraphLevel{_model_name}-shape.pkl")
_train_dataset = None
_test_dataset = None
_model = None
_graph_train_loader = None
_graph_val_loader = None
_graph_test_loader = None
_tu_dataset = None

def target_to_tensor(target_str):
    if target_str == "locality":    return [1,0,0]
    if target_str == "concurrency": return [0,1,0]
    if target_str == "mixed":       return [0,0,1]
    assert False, "Unknown target string"

#TODO: used tensor to target to actually spit out the predicted graph-level classification
def tensor_to_target(target_id):
    if target_id == [1,0,0]: return "locality"
    if target_id == [0,1,0]: return "concurrency"
    if target_id == [0,0,1]: return "mixed"
    assert False, "Unknown target id"

class GNNModel(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, num_layers=2, layer_name="GCN", dp_rate=0.1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        gnn_layer_by_name = {
                "GCN": geom_nn.GCNConv,
                "GAT": geom_nn.GATConv,
                "GraphConv": geom_nn.GraphConv
                }
        gnn_layer = gnn_layer_by_name[layer_name]
        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers-1):
            layers += [
                gnn_layer(in_channels=in_channels,
                          out_channels=out_channels,
                          **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = c_hidden
        layers += [gnn_layer(in_channels=in_channels,
                             out_channels=c_out,
                             **kwargs)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        for l in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(l, geom_nn.MessagePassing):
                try:
                    x = l(x, edge_index)
                except:
                    import ipdb
                    ipdb.set_trace()
                    print("not a big enough matrix")
            else:
                x = l(x)
        return x

class GraphGNNModel(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, dp_rate_linear=0.5, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of output features (usually number of classes)
            dp_rate_linear - Dropout rate before the linear layer (usually much higher than inside the GNN)
            kwargs - Additional arguments for the GNNModel object
        """
        super().__init__()
        self.GNN = GNNModel(c_in=c_in,
                            c_hidden=c_hidden,
                            c_out=c_hidden, # Not our prediction output yet!
                            **kwargs)
        self.head = nn.Sequential(
            nn.Dropout(dp_rate_linear),
            nn.Linear(c_hidden, c_out)
        )

    def forward(self, x, edge_index, batch_idx):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
            batch_idx - Index of batch element for each node
        """
        x = self.GNN(x, edge_index)
        x = geom_nn.global_mean_pool(x, batch_idx) # Average pooling
        x = self.head(x)
        return x

class GraphLevelGNN(pl.LightningModule):

    def __init__(self, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()
        print(model_kwargs)
        self.model = GraphGNNModel(**model_kwargs)
        self.model.to(device)

        #TODO use Softmax for multi-class classification rather than the Binary Cross Entropy loss function.
        #self.loss_module = nn.BCEWithLogitsLoss() if self.hparams.c_out == 1 else nn.CrossEntropyLoss()
        #self.loss_module = nn.Softmax()
        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, data, mode="train"):
        x, edge_index, batch_idx = data['x'], data['edge_index'], data['batch']
        if mode != "predict":
            #using a data loader adds a constraint on unwrapping the data
            edge_index = edge_index[0]
        nx = self.model(x, edge_index, batch_idx)
        nx = nx.squeeze(dim=-1)
        if self.hparams.c_out == 1:
            preds = (nx > 0).float()
            data['y'] = data['y'].float()
        else:
            preds = nx.argmax(dim=1)#was dim=-1
        if mode == "predict":
            return preds
        #TODO check main difference in x, edge_index and batch_idx --- are some values in a different range?
        if nx.shape[1] == 1:
            import ipdb
            ipdb.set_trace()
        loss = self.loss_module(nx, data['y'])
        acc = (preds == data['y']).sum().float() / preds.shape[0]
        return loss, acc

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-2, weight_decay=0.0) # High lr because of small dataset and small model
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="train")
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="val")
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="test")
        self.log('test_acc', acc)

class GraphDataset(torch.utils.data.Dataset):
    # In general:
    #
    # Nodes = Items, People, Locations, Cars etc
    # Edges = Connections, Interactions, Similarity etc
    # Node Features = Attributes
    # Labels = Node-level, Edge-level, Graph-level
    #Optional:
    #   Edge Weights = Strength of the Connection, Number of Interactions ...
    #   Edge Features = Additional (multi-dimensional) properties describing the edge

    # In this instance:
    # Nodes = tasks
    # Edges = dependencies
    # Node Features = # of tasks x memory buffers used per task
    # Labels = Graph-level (in this instance target_for
    def __init__(self, data_list, shape=None):
        self.data_list = data_list

        # the shape and batches need to be processed differently if we are using the model directly for predictions, the data loaders do more fancy things (like wrapping batches and edge_index's in lists)
        for_prediction_mode = True if shape is not None else False

        # get max number of node features
        # feature_matrix is of the shape [ # nodes x # feature per node]
        # unfortunately this varies per graph and so needs to be padded
        # thus we need to collect the largest shape, which means we need
        # to call these functions twice (once per initialization and once per indexing)
        shapes = []
        if shape is None: #this is the first training and so we need to determine the largest size matrix to pad
            for data in data_list:
                nodes = self.__construct_nodes(data)
                feature_matrix = self.__construct_node_feature_matrix_for_graph(nodes)
                shapes.append(feature_matrix.shape)
        else: shapes.append(shape)

        self.max_feature_matrix_shape = max(shapes)
        self.num_node_features = self.max_feature_matrix_shape[1]

        #now do all the file IO and process it in one shot
        self.gnn_form_of_json = []
        for i,val in enumerate(self.data_list):
            x = self.__convert_from_json_to_gnn(i,wrap_batch=for_prediction_mode) #if the edge_index isn't empty add it!
            if x['edge_index'].numel() != 0: self.gnn_form_of_json.append(x)

    def to(self,device):
        for item in self.gnn_form_of_json:
            item['x'].to(device)
            item['edge_index'].to(device)
            item['batch'].to(device)
            item['y'].to(device)

    def get_item_at_index(self,idx):
        #TODO: what features do we need?
        return self.gnn_form_of_json[idx]

    def __len__(self):
        return len(self.gnn_form_of_json)

    def shuffle(self):
        import random
        random.shuffle(self.gnn_form_of_json)

    def shape(self):
        return self.max_feature_matrix_shape

    def num_node_features(self):
        return self.max_feature_matrix_shape[1]

    def num_classes(self):
        return 3 #the range of labels in the target_to_tensor(target_str) function

    def __construct_nodes(self,data):
        return data['content']['iris-graph']['graph']['tasks']

    def __construct_edges(self,data):
        nodes = self.__construct_nodes(data)
        number_of_nodes = len(nodes)
        edges = []
        for n in nodes:
            #TODO: determine how to add the d2h and h2d memory buffers to node_feature_matrix
            if 'final' in n['name']: continue
            if 'dmemflush' in n['name']: continue
            ref_node_name = n['name']
            for d in n['depends']:
                # remove the leading task from the string, get its numerical value
                # and increment at the corresponding id
                edges.append([int(ref_node_name.replace('task','')),int(d.replace('task',''))])
        return np.array(edges,dtype=int).transpose()

    def __construct_node_feature_matrix_for_graph(self,nodes):
        # generate the full list of node attributes (memory buffer names used)
        node_uses_memory = {}
        for node in nodes:
            param_names = []
            #ignore commands (currently only 1 command per task)
            parameters = None
            if "kernel" in node['commands'][0].keys():
                parameters = node['commands'][0]['kernel']['parameters']
                for param in parameters:
                    param_names.append(param['value'])
            elif "d2h" in node['commands'][0].keys():
                #TODO: determine how to add the d2h and h2d memory buffers to node_feature_matrix
                #parameters = node['commands'][0]['d2h']['device_memory']
                #param_names.append(parameters)
                continue
            elif "h2d" in node['commands'][0].keys():
                #parameters = node['commands'][0]['h2d']['device_memory']
                #param_names.append(parameters)
                continue
            assert parameters is not None
            node_uses_memory[node['name'].replace('task','')] = param_names
        # apply 1-hot encodings for non-numerical features (memory buffer names)
        one_hot = pd.DataFrame(node_uses_memory).transpose()
        node_feature_matrix = pd.get_dummies(one_hot).astype(np.float64)
        return node_feature_matrix.to_numpy()
    
    def __getitem__(self,idx):
        return self.gnn_form_of_json[idx]

    def __convert_from_json_to_gnn(self, idx, wrap_batch=False):
        # Nodes = Task #
        # Edges = List of Dependencies
        # Node Attributes = Memory objects
        # Labels = Scheduled for

        # Homogeneous Graph: No difference between tasks, and no different edge-types are needed (they are all dependencies)
        data = self.data_list[idx]
        nodes = self.__construct_nodes(data)
        edges = self.__construct_edges(data)
        label = data['schedule-for']
        node_feature_matrix = self.__construct_node_feature_matrix_for_graph(nodes)

        # pad node_feature_matrix for the maximum sized matrix (based on the num_node_features)
        padding_to_add = tuple(map(lambda i, j: i - j, self.max_feature_matrix_shape, node_feature_matrix.shape))
        node_feature_matrix = np.pad(node_feature_matrix,[(0,padding_to_add[0]), (0,padding_to_add[1])],mode="constant")
        assert node_feature_matrix.shape == self.max_feature_matrix_shape
        #transform and return
        node_feature_matrix = torch.Tensor(node_feature_matrix)
        #batch_id = torch.LongTensor([1,4097,256]) #can be 1 since this is a per graph classification
        if wrap_batch: batch_id = np.array([idx])
        else: batch_id = np.array(idx)#TODO: stopgap measure---remove idx+1 when I figure our why resuming the checkpoint fails to index appropriately
        #TODO: expect y to return a list of probabilities but is this correct?
        return {'x':node_feature_matrix,
                'edge_index':torch.from_numpy(edges),
                'y':torch.LongTensor(target_to_tensor(label)),
                'batch':torch.from_numpy(batch_id)}


def train_graph_classifier(**model_kwargs):
    global _graph_train_loader, _graph_val_loader, _graph_test_loader


    # Create a PyTorch Lightning trainer with the generation callback
    os.makedirs(_root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=_root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=num_devices,
                         max_epochs=1,
                         #TODO: bring back good training size of epochs
                         #max_epochs=30,
                         log_every_n_steps=1,
                         enable_progress_bar=True)
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    if os.path.isfile(_pretrained_filename):
        print("Found pretrained model, loading...")
        model = GraphLevelGNN.load_from_checkpoint(_pretrained_filename)
    else:
        pl.seed_everything(42)
        model = GraphLevelGNN(c_in=_tu_dataset.num_node_features,
                              c_out=3,
                              **model_kwargs)
        trainer.fit(model, _graph_train_loader, _graph_val_loader)
        model = GraphLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on validation and test set
    train_result = trainer.test(model, _graph_train_loader, verbose=False)
    test_result = trainer.test(model, _graph_test_loader, verbose=False)
    result = {"test": test_result[0]['test_acc'], "train": train_result[0]['test_acc']}

    # save the model
    trainer.save_checkpoint(_pretrained_filename)
    # and write out the training size
    with open(_pretrained_shape_file,"wb") as shape_file:
        pickle.dump(_tu_dataset.shape(),shape_file)
    return model, result

def train():
    global _model

    _model, result = train_graph_classifier(c_hidden=256,
                                           layer_name="GraphConv",
                                           num_layers=3,
                                           dp_rate_linear=0.5,
                                           dp_rate=0.0)
    print(f"Train performance: {100.0*result['train']:4.2f}%")
    print(f"Test performance:  {100.0*result['test']:4.2f}%")
    print(result)
    torch.save(_model,_model_filename)
    print(colored("Saved trained model to {}".format(_model_filename),"green"))

def generate_demonstration_dataset():
    tu_dataset = torch_geometric.datasets.TUDataset(root=DATASET_PATH, name="MUTAG")
    print("Data object:", tu_dataset.data)
    print("Length:", len(tu_dataset))
    print(f"Average label: {tu_dataset.data.y.int().mean().item():4.2f}")
    return tu_dataset

def generate_dataset():
    raw_dataset = []
    #build the list of graph names, the payloads used to generate them, and the proper scheduling target
    #locality data
    for i in range(0,13):
        num_tasks = 2**i
        filename = '{}/linear-{}.json'.format(_dataset_directory,num_tasks)
        raw_dataset.append({
                'args':'--graph={} --kernels="ijk" --duplicates="0" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --kernel-split="100" --depth={} --num-tasks={} --min-width=1 --max-width=1 --use-data-memory'.format(filename,num_tasks,num_tasks),
                'file' : filename,
                'schedule-for' : 'locality'
                })
    #concurrency data
    for i in range(0,13):
        num_tasks = 2**i
        filename = '{}/diamond-{}.json'.format(_dataset_directory,num_tasks)
        raw_dataset.append({
                'args':'--graph={} --concurrent-kernels="ijk:{}" --kernels="ijk" --duplicates="0" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split="100" --depth=1 --num-tasks={} --min-width={} --max-width={} --sandwich --use-data-memory'.format(filename,num_tasks,num_tasks,num_tasks,num_tasks),
                'file' : filename,
                'schedule-for' : 'concurrency'
                })
    #mixed data
    for i in range(3,13):
        num_tasks = 2**i
        filename = '{}/mashload-{}.json'.format(_dataset_directory,num_tasks)
        raw_dataset.append({
                'args':'--graph={} --kernels="ijk" --kernel-split="100" --depth=10 --num-tasks={} --min-width=7 --max-width=7 --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --concurrent-kernels="ijk:14" --skips=3 --cdf-mean=2 --cdf-std-dev=0 --use-data-memory --handover-in-memory-shuffle --num-memory-shuffles=32'.format(filename,num_tasks),
                'file' : filename,
                'schedule-for' : 'mixed'
                })

    # generate the datasets if it doesn't exists --- delete the graphs directory
    from pathlib import Path
    if (Path(_dataset_directory).is_dir() == False):
        Path(_dataset_directory).mkdir(parents=True, exist_ok=True)

        from dagger.dagger_generator import parse_args as gen_dagger_parse_args
        from dagger.dagger_generator import run as generate_graph_as_json
        for d in raw_dataset:
            print("generating graph:{}".format(d['file']))
            rargs = gen_dagger_parse_args(d['args'])
            generate_graph_as_json()

    # fill in content of each payload from the json
    from dagger.dagger_generator import get_task_graph_from_json
    for d in raw_dataset:
        d['content'] = get_task_graph_from_json(d['file'])

    # Create a dataset and data loader
    tu_dataset = GraphDataset(raw_dataset)
    return tu_dataset

def predict():
    new_dataset = []
    num_tasks = 32
    #new linear test
    filename = '{}/linear-{}.json'.format(_dataset_directory,num_tasks)
    new_dataset.append({
            'args':'--graph={} --kernels="ijk" --duplicates="0" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --kernel-split="100" --depth={} --num-tasks={} --min-width=1 --max-width=1 --use-data-memory'.format(filename,num_tasks,num_tasks),
            'file' : filename,
            'schedule-for' : 'locality'
            })
    #new concurrency test
    filename = '{}/diamond-{}.json'.format(_dataset_directory,num_tasks)
    new_dataset.append({
            'args':'--graph={} --concurrent-kernels="ijk:{}" --kernels="ijk" --duplicates="0" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split="100" --depth=1 --num-tasks={} --min-width={} --max-width={} --sandwich --use-data-memory'.format(filename,num_tasks,num_tasks,num_tasks,num_tasks),
            'file' : filename,
            'schedule-for' : 'concurrency'
            })
    #new mixed test
    filename = '{}/mashload-{}.json'.format(_dataset_directory,num_tasks)
    new_dataset.append({
            'args':'--graph={} --kernels="ijk" --kernel-split="100" --depth=10 --num-tasks={} --min-width=7 --max-width=7 --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --concurrent-kernels="ijk:14" --skips=3 --cdf-mean=2 --cdf-std-dev=0 --use-data-memory --handover-in-memory-shuffle --num-memory-shuffles=32'.format(filename,num_tasks),
            'file' : filename,
            'schedule-for' : 'mixed'
            })

    from pathlib import Path
    if (Path(_dataset_directory).is_dir() == False):
        Path(_dataset_directory).mkdir(parents=True, exist_ok=True)

        from dagger.dagger_generator import parse_args as gen_dagger_parse_args
        from dagger.dagger_generator import run as generate_graph_as_json
        for d in new_dataset:
            print("generating graph:{}".format(d['file']))
            rargs = gen_dagger_parse_args(d['args'])
            generate_graph_as_json()

    # fill in content of each payload from the json
    from dagger.dagger_generator import get_task_graph_from_json
    for d in new_dataset:
        d['content'] = get_task_graph_from_json(d['file'])

    # load the max shape that the model was trained with
    with open(_pretrained_shape_file ,"rb") as shape_file:
        model_trained_on_shape = pickle.load(shape_file)

    # Create a dataset and data loader with this new max shape
    tu_dataset = GraphDataset(new_dataset,shape=model_trained_on_shape)

    # Convert to feature matrix and predict
    #_graph_test_loader = geom_loader.DataLoader(tu_dataset, num_workers=num_workers, batch_size=_batch_size)
    #batch = next(iter(_graph_test_loader))
    #from termcolor import colored
    #print(colored("Begin bad output","red"))
    #print("Batch:", batch)
    #print("Labels:", batch['y'][:10])
    #print("Batch indices:", batch['edge_index'][:40])
    #print(colored("End bad output","light_red"))

    ##trainer = pl.Trainer(resume_from_checkpoint=_pretrained_filename)
    #trainer = pl.Trainer(default_root_dir=_root_dir,
    #                    callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
    #                    accelerator="gpu" if str(device).startswith("cuda") else "cpu",
    #                    devices=num_devices,
    #                    max_epochs=30,
    #                    log_every_n_steps=1,
    #                 enable_progress_bar=True)
    #trainer.logger._default_hp_metric = None # Optional logging argument that we don't need
    #test_result = trainer.test(_model, _graph_test_loader, verbose=True, ckpt_path=_pretrained_filename )

    #TODO: interpret predicted to result to a scheduler decision
    model = torch.load(_model_filename)
    model.to(device)
    model.eval()
    tu_dataset.to(device)
    #for each item to predict
    for i, val in enumerate(tu_dataset):
        inputs = tu_dataset.get_item_at_index(i)
        x, edge_index = inputs['x'], inputs['edge_index']
        x.to(device)
        edge_index.to(device)
        prediction = model(inputs,mode="predict")
        prediction = prediction.detach().cpu().numpy()
        print(colored("Prediction completed! Set # {} predicted: {}".format(i, prediction),"cyan"))
        import ipdb
        ipdb.set_trace()
    #prediction = model(x, edge_index)
    # take the pre-trained model and new DAG to make the prediction
    return



def load_dataset():
    global _graph_train_loader, _graph_val_loader, _graph_test_loader, _tu_dataset

    #_tu_dataset = generate_demonstration_dataset()
    _tu_dataset = generate_dataset()
    _tu_dataset.shuffle()
    #use 70% of the dataset for training
    train_size = int(0.8 * len(_tu_dataset))
    test_size = len(_tu_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(_tu_dataset, [train_size, test_size])
    _graph_train_loader = geom_loader.DataLoader(train_dataset, shuffle=True, num_workers=num_workers, batch_size=_batch_size)
    _graph_val_loader = geom_loader.DataLoader(test_dataset, num_workers=num_workers,batch_size=_batch_size) # Additional loader if you want to change to a larger dataset
    _graph_test_loader = geom_loader.DataLoader(test_dataset, num_workers=num_workers,batch_size=_batch_size)

    batch = next(iter(_graph_test_loader))
    print("Batch:", batch)
    print("Labels:", batch['y'][:10])
    print("Batch indices:", batch['edge_index'][:40])

if __name__ == '__main__':
    if os.path.isfile(_pretrained_filename):
        print("Found pretrained model, loading...")
        _model = GraphLevelGNN.load_from_checkpoint(_pretrained_filename)
    else:
        print("Building a new model...")
        load_dataset()
        #load_demonstration_dataset()
        train()

    predict()

