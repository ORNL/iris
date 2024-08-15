#!python3

# all dependencies for this demo should be installed by running the following:
# conda env create --name gnn-prediction --file gnn-prediction.yaml
# conda activate env gnn-prediction

# to train the model:
# ./iris-gnn.py
# and to use it on actual IRIS json dags:
# ./iris-gnn.py <file.json>
# or
# ./iris-gnn.py <directory of json files>

# may need to run `./build.sh` from `../..`
# or `source ~/.iris/setup.zsh`

import os
import sys
import numpy as np
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
import torch_geometric.nn as geom_nn
import torch_geometric.loader as geom_loader
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pickle
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from termcolor import colored
from functools import reduce
from operator import itemgetter

# Variable training parameters
_max_epochs = 60

# NOTE: we don't currently support batching which leads to longer training times
#   TODO    * support batching
#           * add support for multi-gpu
_batch_size = 1

# Initialize PyTorch environment
if "WORKING_DIRECTORY" in os.environ.keys(): os.chdir(os.environ.get('WORKING_DIRECTORY'))
DATASET_PATH = "./data"
CHECKPOINT_PATH = "./saved_models"
pl.seed_everything(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
use_gpu = torch.cuda.is_available()
if use_gpu:
    device = torch.device("cuda")
    num_devices = 1#torch.cuda.device_count()
    num_workers = 0
else:
    device = torch.device("cpu")
    num_devices = 1
    num_workers = 0#os.cpu_count()/4

# Other fixed global variables:
_num_classes = 3
_dataset_directory = 'graphs'
_model_name = "iris_scheduler_gnn_graph_level_classifier"
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
_training_log = { "accuracy" : [], "loss" : []}
_validation_log = { "accuracy" : [], "loss" : []}

def summarize(prediction,actual):
    print(colored("predicted:{} actual:{}".format(value_to_target(prediction),tensor_to_target(actual)),"cyan"))
    if actual == "?": print(colored("\'?\' indicates we performed prediction on unknown data.","green"))

def target_to_tensor(target_str):
    if target_str == "locality":    return [1, 0, 0]
    if target_str == "concurrency": return [0, 1, 0]
    if target_str == "mixed":       return [0, 0, 1]
    if target_str == "?":           return [0, 0, 0]
    assert False, "Unknown target string"

def target_to_value(target_str):
    if target_str == "locality":    return[ 0]
    if target_str == "concurrency": return[ 1]
    if target_str == "mixed":       return[ 2]
    if target_str == "?":           return[-1]
    assert False, "Unknown target string"

def tensor_to_target(target_id):
    target_id = target_id.tolist()
    if target_id == [1, 0, 0]: return "locality"
    if target_id == [0, 1, 0]: return "concurrency"
    if target_id == [0, 0, 1]: return "mixed"
    if target_id == [0, 0, 0]: return "?"
    assert False, "Unknown target id"

def value_to_target(target_id):
    if target_id ==  0:  return "locality"
    if target_id ==  1:  return "concurrency"
    if target_id ==  2:  return "mixed"
    if target_id == -1: return "?"
    assert False, "Unknown target id"

def tensor_to_value(target_id):
    target_id = target_id.tolist()
    if target_id == [1, 0, 0]: return [ 0]
    if target_id == [0, 1, 0]: return [ 1]
    if target_id == [0, 0, 1]: return [ 2]
    if target_id == [0, 0, 0]: return [-1]
    assert False, "Unknown target id"

def value_to_tensor(target):
    target = target.tolist()
    if target ==  0: return torch.LongTensor([1, 0, 0])
    if target ==  1: return torch.LongTensor([0, 1, 0])
    if target ==  2: return torch.LongTensor([0, 0, 1])
    if target == -1: return torch.LongTensor([0, 0, 0])
    assert False, "Unknown target id"

class GNN(torch.nn.Module):
    def __init__(self, c_in, c_hidden, c_out, **kwargs):
        super(GNN, self).__init__()
        self.conv1 = geom_nn.GraphConv(c_in,c_hidden)
        self.conv2 = geom_nn.GraphConv(c_hidden,c_hidden)
        self.conv3 = geom_nn.GraphConv(c_hidden,c_hidden)
        self.lin = geom_nn.Linear(c_hidden, c_out)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = geom_nn.global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

class GraphLevelGNN(pl.LightningModule):
    global _training_log, _validation_log
    def __init__(self, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()
        print(model_kwargs)
        self.model = GNN(**model_kwargs)
        print(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.to(device)
        self.model.to(device)
        self.model.train()
        
        self.validation_step_outputs = []
        self.training_step_outputs = []
        self.train_metrics = {"epoch": [], "mean_accuracy": [], "mean_loss": []}
        self.valid_metrics = {"epoch": [], "mean_accuracy": [], "mean_loss": []}

    def forward(self, data, mode="train"):
        if mode == "predict":
            x, edge_index, batch_idx, y = data['x'], data['edge_index'][0], data['batch'], data['y']
        else:
            x, edge_index, batch_idx, y = data['x'][0], data['edge_index'][0], data['batch'], data['y'][0]
        #TODO: This unpacking of each value is to do with the data-loader, and they are associated with automatic batching (from the pytorch lightning module). This would need to be unpicked to support automatic batching
        out = self.model(x, edge_index, batch_idx)
        out = out[-1]#only the last row holds the result
        if mode == "predict":
            return out.argmax()
        pred = value_to_tensor(out.argmax()).float().to(device)
        loss = self.criterion(out, y.float()).sum()
        acc  = (pred == y).sum().float() / _num_classes
        return loss, acc

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0.01)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="train")
        _training_log['accuracy'].append(acc.tolist())
        _training_log['loss'].append(loss.tolist())
        
        self.training_step_outputs.append({"accuracy": acc, "loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="val")
        _validation_log['accuracy'].append(acc.tolist())
        _validation_log['loss'].append(loss.tolist())

        self.validation_step_outputs.append({"accuracy": acc, "loss": loss})
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="test")
        self.log('test_acc', acc)

    def on_validation_epoch_end(self):
        all_acc = [x['accuracy'].item() for x in self.validation_step_outputs]
        all_loss = [x['loss'].item() for x in self.validation_step_outputs]
        
        self.valid_metrics["epoch"].append(self.current_epoch)
        self.valid_metrics["mean_accuracy"].append(np.mean(all_acc))
        self.valid_metrics["mean_loss"].append(np.mean(all_loss))

        self.validation_step_outputs.clear()  # free memory

    def on_train_epoch_end(self):
        all_acc = [x['accuracy'].item() for x in self.training_step_outputs]
        all_loss = [x['loss'].item() for x in self.training_step_outputs]

        self.train_metrics["epoch"].append(self.current_epoch)
        self.train_metrics["mean_accuracy"].append(np.mean(all_acc))
        self.train_metrics["mean_loss"].append(np.mean(all_loss))

        self.training_step_outputs.clear()  # free memory


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
        shapes = None
        if shape is None: #this is the first training and so we need to determine the largest size matrix to pad
            shapes = self.__compute_max_dimensions(data_list)
        else: shapes = shape

        #initialize the values to pad up to
        self._num_nodes = shapes[0]
        self._num_node_features = shapes[1]

        #now do all the file IO and process it in one shot
        self.gnn_form_of_json = []
        for i,val in enumerate(self.data_list):
            x = self.__convert_from_json_to_gnn(i,wrap_batch=for_prediction_mode) #if the edge_index isn't empty add it!
            if x['edge_index'].numel() != 0: self.gnn_form_of_json.append(x)

    def to(self, item, device):
        item['x']           = item['x'].to(device)
        item['edge_index']  = item['edge_index'].to(device)
        item['batch']       = item['batch'].to(device)
        item['y']           = item['y'].to(device)
        return item

    def get_item_at_index(self,idx):
        x = self.gnn_form_of_json[idx]
        x = self.to(x,device)
        return x

    def __compute_max_dimensions(self,data_list):
        feat_dimensions = []
        for data in data_list:
            nodes = self.__construct_nodes(data)
            memory_set = self.__get_set_of_unique_memory(nodes)
            feat_dimensions.append((len(nodes),len(memory_set)))
        #looks kind of hacky, but max(dimensions) on returns a tuple based on the first dimension
        #this solution actually gets the max in each
        return (max(feat_dimensions, key=itemgetter(0))[0],max(feat_dimensions, key=itemgetter(1))[1])

    def __len__(self):
        return len(self.gnn_form_of_json)

    def shuffle(self):
        import random
        random.shuffle(self.gnn_form_of_json)

    def shape(self):
        return (self._num_nodes,self._num_node_features)

    def num_nodes(self):
        return self._num_nodes

    def num_node_features(self):
        return self._num_node_features

    def num_classes(self):
        return _num_classes #the range of labels in the target_to_tensor(target_str) function

    def __construct_nodes(self,data):
        return data['content']['iris-graph']['graph']['tasks']

    def __get_memory_objects(self,node):
        if "kernel" in node.keys():
            node_memory_object_names = [p['value'] for p in node['kernel']['parameters']]
        elif "d2h" in node.keys():
            node_memory_object_names = [node['d2h']['device_memory']]
        elif "h2d" in node.keys():
            node_memory_object_names = [node['h2d']['device_memory']]
        else:
            assert False, "unknown command in task {}".format(node[i]['name'])
        return node_memory_object_names

    def __get_set_of_unique_memory(self,nodes):
        memory_set = {}
        for i, node in enumerate(nodes):
            for command in node['commands']:
                node_memory_object_names = self.__get_memory_objects(command)
                memory_set = set(node_memory_object_names + list(memory_set))
        return memory_set

    def __construct_node_feature_matrix_for_graph(self,nodes):
        #[num_nodes, num_node_features]
        feature_matrix = np.zeros(shape=(self.num_nodes(),self.num_node_features()))#create a full size feature matrix
        memory_set = self.__get_set_of_unique_memory(nodes)
        for n, node in enumerate(nodes):
            for command in node['commands']:
                memory_this_node = self.__get_memory_objects(command)
                feature_vector = [int(f in memory_this_node) for f in memory_set]
            try:
                for i,f in enumerate(feature_vector):
                    feature_matrix[n][i] = f
            except:
                import ipdb
                ipdb.set_trace()
                print('debug')
        return feature_matrix

    def __construct_edges(self,data,nodes):
        number_of_nodes = len(nodes)
        edges = []
        for i,n in enumerate(nodes):
            #TODO: determine how to add the d2h and h2d memory buffers to node_feature_matrix
            if 'initial' in n['name']: n['name'] = 'task0'
            if 'final' in n['name']: n['name'] = 'task{}'.format(i)
            if 'dmemflush' in n['name']: n['name'] = "task{}".format(i)
            if 'transferto' in n['name']: n['name'] = "task{}".format(i)
            if 'transferfrom' in n['name']: n['name'] = "task{}".format(i)

            ref_node_name = n['name']
            for d in n['depends']:
                # remove the leading task from the string, get its numerical value
                # and increment at the corresponding id
                edges.append([int(ref_node_name.replace('task','')),int(d.replace('task',''))])
        return np.array(edges,dtype=int).transpose()

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
        edges = self.__construct_edges(data,nodes)
        label = data['schedule-for']
        node_feature_matrix = self.__construct_node_feature_matrix_for_graph(nodes)

        #transform and return
        node_feature_matrix = torch.Tensor(node_feature_matrix)
        if wrap_batch:
            batch_id = np.array([idx])
            edges = np.array([edges])
        else: batch_id = np.array(idx)
        edges = torch.from_numpy(edges)
        label = torch.LongTensor(target_to_tensor(label))
        batch_id = torch.from_numpy(batch_id)

        return {'x': node_feature_matrix,
                'edge_index':edges,
                'y':label,
                'batch':batch_id}

def train_graph_classifier(**model_kwargs):
    global _graph_train_loader, _graph_val_loader, _graph_test_loader, _max_epochs


    # Create a PyTorch Lightning trainer with the generation callback
    os.makedirs(_root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=_root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=num_devices,
                         max_epochs=_max_epochs,
                         #log_every_n_steps=5,
                         num_sanity_val_steps=0,
                         enable_progress_bar=True)

    # Check whether pretrained model exists. If yes, load it and skip training
    if os.path.isfile(_pretrained_filename):
        print("Found pretrained model, loading...")
        model = GraphLevelGNN.load_from_checkpoint(_pretrained_filename)
    else:
        pl.seed_everything(42)
        model = GraphLevelGNN(c_in=_tu_dataset.num_node_features(),
                              c_out=_num_classes,
                              **model_kwargs)
        trainer.fit(model, _graph_train_loader, _graph_val_loader)

        import matplotlib.pylab as plt
        plt.clf()
        plt.close("all")
        plt.plot(model.valid_metrics['epoch'], model.valid_metrics['mean_accuracy'], label='Validation', color="C0")
        plt.plot(model.train_metrics['epoch'], model.train_metrics['mean_accuracy'], label='Training', color="C1")
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.savefig('accuracy_metrics.pdf')
        plt.clf()
        plt.close("all")

        plt.plot(model.valid_metrics['epoch'], model.valid_metrics['mean_loss'], label='Validation', color="C0")
        plt.plot(model.train_metrics['epoch'], model.train_metrics['mean_loss'], label='Training', color="C1")
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('loss_metrics.pdf')
        plt.clf()
        plt.close("all")


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

    _model, result = train_graph_classifier(c_hidden=64,
                                           layer_name="GraphConv",
                                           num_layers=3,
                                           dp_rate_linear=0.01,
                                           dp_rate=0.0)
    print(f"Train performance: {100.0*result['train']:4.2f}%")
    print(f"Test performance:  {100.0*result['test']:4.2f}%")
    print(result)
    import __main__
    setattr(__main__, "GraphLevelGNN", GraphLevelGNN)
    setattr(__main__, "GNN", GNN)
    #setattr(__main__, "GraphGNNModel", GraphGNNModel)
    #setattr(__main__, "GNNModel", GNNModel)

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
    for i in range(3,13):
        num_tasks = 2**i
        filename = '{}/linear-{}.json'.format(_dataset_directory,num_tasks)
        raw_dataset.append({
                'args':'--graph={} --kernels="ijk" --duplicates="0" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --kernel-split="100" --depth={} --num-tasks={} --min-width=1 --max-width=1 --use-data-memory'.format(filename,num_tasks,num_tasks),
                'file' : filename,
                'schedule-for' : 'locality'
                })
    #locality data (with duplicates)
    for i in range(3,13):
        num_tasks = 2**i
        filename = '{}/lineartwo-{}.json'.format(_dataset_directory,num_tasks)
        raw_dataset.append({
                'args':'--graph={} --kernels="ijk" --duplicates="2" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --kernel-split="100" --depth={} --num-tasks={} --min-width=1 --max-width=1 --use-data-memory'.format(filename,num_tasks,num_tasks),
                'file' : filename,
                'schedule-for' : 'locality'
                })
    for i in range(3,13):
        num_tasks = 2**i
        filename = '{}/linearthree-{}.json'.format(_dataset_directory,num_tasks)
        raw_dataset.append({
            'args':'--graph={} --kernels="ijk" --duplicates="3" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --kernel-split="100" --depth={} --num-tasks={} --min-width=1 --max-width=1 --use-data-memory --concurrent-kernels="ijk:{}"'.format(filename,num_tasks,num_tasks,i),
                'file' : filename,
                'schedule-for' : 'locality'
                })
    #concurrency data
    for i in range(3,13):
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
        filename = '{}/chainlink-{}.json'.format(_dataset_directory,num_tasks)
        raw_dataset.append({
                'args':'--graph={} --concurrent-kernels="ijk:2" --kernels="ijk" --duplicates="0" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split="100" --depth={} --num-tasks={} --min-width=1 --max-width=2 --cdf-mean=1.5 --cdf-std-dev=0 --sandwich --use-data-memory'.format(filename,round(num_tasks/2),num_tasks),
                'file' : filename,
                'schedule-for' : 'mixed'
                })
    for i in range(3,13):
        num_tasks = 2**i
        filename = '{}/tangled-{}.json'.format(_dataset_directory,num_tasks)
        raw_dataset.append({
                'args':'--graph={} --concurrent-kernels="ijk:12" --kernels="ijk" --duplicates="0" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split="100" --depth={} --num-tasks={} --min-width=1 --max-width=12 --cdf-mean=2 --cdf-std-dev=0 --skips=3 --sandwich --use-data-memory'.format(filename,num_tasks,num_tasks),
                'file' : filename,
                'schedule-for' : 'mixed'
                })
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
    #tu_dataset.to(device)
    return tu_dataset

def generate_fresh_prediction_payload_for_testing():
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
    filename = '{}/chainlink-{}.json'.format(_dataset_directory,num_tasks)
    new_dataset.append({
            'args':'--graph={} --concurrent-kernels="ijk:2" --kernels="ijk" --duplicates="0" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split="100" --depth={} --num-tasks={} --min-width=1 --max-width=2 --cdf-mean=1.5 --cdf-std-dev=0 --sandwich --use-data-memory'.format(filename,round(num_tasks/2),num_tasks),
            'file' : filename,
            'schedule-for' : 'mixed'
            })
    filename = '{}/tangled-{}.json'.format(_dataset_directory,num_tasks)
    new_dataset.append({
            'args':'--graph={} --concurrent-kernels="ijk:12" --kernels="ijk" --duplicates="0" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split="100" --depth={} --num-tasks={} --min-width=1 --max-width=12 --cdf-mean=2 --cdf-std-dev=0 --skips=3 --sandwich --use-data-memory'.format(filename,num_tasks,num_tasks),
            'file' : filename,
            'schedule-for' : 'mixed'
            })
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

    return new_dataset

def load_content_from_files(new_dataset):
    from dagger.dagger_generator import get_task_graph_from_json
    for d in new_dataset:
        d['content'] = get_task_graph_from_json(d['file'])
        if 'schedule-for' not in d.keys(): d['schedule-for']='?'
    return new_dataset

def predict(new_dataset):
    assert new_dataset is not None, "The given list of json graphs to predict is empty!"
    # load the max shape that the model was trained with
    with open(_pretrained_shape_file ,"rb") as shape_file:
        model_trained_on_shape = pickle.load(shape_file)

    # Create a dataset and data loader with this new max shape
    tu_dataset = GraphDataset(new_dataset,shape=model_trained_on_shape)
    tu_dataset.shuffle()

    #the following environments are needed to load the pickled model from __main__
    import __main__
    setattr(__main__, "GraphLevelGNN", GraphLevelGNN)
    setattr(__main__, "GNN", GNN)
    #setattr(__main__, "GraphGNNModel", GraphGNNModel)
    #setattr(__main__, "GNNModel", GNNModel)

    model = torch.load(_model_filename)
    model.to(device)
    model.eval()

    prediction = None
    #for each item to predict
    for i, val in enumerate(tu_dataset):
        inputs = tu_dataset.get_item_at_index(i)
        prediction = model(inputs,mode="predict")
        prediction = prediction.tolist()
        #prediction = prediction.detach().cpu().numpy()
        print(colored("Prediction completed! Set # {}".format(i),"cyan"))
        summarize(prediction,actual=inputs['y'])

    assert prediction is not None,"We don't have a prediction!"
    # TODO take the pre-trained model and new DAG to make the prediction
    return prediction

#TODO: determine if we need to resize/pad edge_index to the maximum size 
#(as we do for the feature matrix) but for batching
def my_collate_fun(batch_dataset):
    # all edge_index vectors need to be the same shape per batch
    # so in order to be more efficient using batching we have to add extra padding operations
    edge_index_dims = []
    for data in batch_dataset:
        edge_index_dims.append(data['edge_index'].shape)
    max_shape = (max(edge_index_dims, key=itemgetter(0))[0],max(edge_index_dims, key=itemgetter(1))[1])
    max_shape = torch.Size(max_shape)
    for data in batch_dataset:
        if data['edge_index'].shape < max_shape:
            new_edge_index = np.zeros(shape=max_shape,dtype=np.int64)
            for d, dimension in enumerate(data['edge_index'].shape):
                for i, value in enumerate(data['edge_index'][d]):
                    new_edge_index[d][i] = value
            data['edge_index'] = torch.from_numpy(new_edge_index)
    #TODO: debug from here!
    #import ipdb
    #ipdb.set_trace()
    return torch.utils.data.dataloader.default_collate(batch_dataset)

def load_dataset():
    global _graph_train_loader, _graph_val_loader, _graph_test_loader, _tu_dataset

    #_tu_dataset = generate_demonstration_dataset()
    _tu_dataset = generate_dataset()
    _tu_dataset.shuffle()
    #use 70% of the dataset for training
    train_size = int(0.8 * len(_tu_dataset))
    test_size = len(_tu_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(_tu_dataset, [train_size, test_size])

    _graph_train_loader = DataLoader(train_dataset, shuffle=True, num_workers=num_workers, batch_size=_batch_size, collate_fn=my_collate_fun)
    _graph_val_loader = DataLoader(test_dataset, num_workers=num_workers,batch_size=_batch_size, collate_fn=my_collate_fun) # Additional loader if you want to change to a larger dataset
    _graph_test_loader = DataLoader(test_dataset, num_workers=num_workers, batch_size=_batch_size, collate_fn=my_collate_fun)

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
        #load_demonstration_dataset()
        load_dataset()
        train()

    #if no arguments are given do the standard inference
    if len(sys.argv) == 1:
        new_dataset = generate_fresh_prediction_payload_for_testing()
        # fill in content of each payload from the json
        new_dataset = load_content_from_files(new_dataset)
        predict(new_dataset)
        sys.exit()

    #else load in each IRIS json file and update it with the inference predictions
    #this was test was generated from the dynamic evaluation paper and copied in here with:
    #   cp -r ../../../iris/apps/dagger/dagger-payloads/ ./prototype-payloads

    # 1) generate the file listing
    json_files = np.array([])
    for argument in sys.argv[1:]:
        if os.path.isdir(argument):
            for each_file in os.listdir(argument):
                json_files = np.append(json_files,{'file':os.path.join(argument,each_file)})
        elif os.path.isfile(argument):
            json_files = np.append(json_files,{'file':argument})
    json_files = json_files.flatten()

    # 2) convert each into a compliant data-frame
    new_dataset = load_content_from_files(json_files)
    # 3) run the same prediction function on it.
    prediction = predict(new_dataset)
    if len(json_files) == 1: #if only 1 file is provided echo the result as the last output for the IRIS policy to load it.
        print(value_to_target(prediction))

