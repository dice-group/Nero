from ontolearn import KnowledgeBase
from typing import List, Tuple, Set
from owlapy.model import OWLEquivalentClassesAxiom, OWLClass, IRI, OWLObjectIntersectionOf, OWLObjectUnionOf, \
    OWLObjectSomeValuesFrom, OWLObjectInverseOf, OWLObjectProperty, OWLThing
from argparse import ArgumentParser

from owlapy.render import DLSyntaxObjectRenderer

import random
from collections import deque
from .model import NERO

import torch
from torch import nn
import numpy as np
from .static_funcs import *
from .util_classes import *
from .neural_arch import *
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class Trainer:
    """ Trainer of Neural Class Expression Learner"""

    def __init__(self, learning_problems, args, storage_path: str, num_instances: int):
        self.loss_func = None
        self.model = None
        self.optimizer = None
        self.learning_problems = learning_problems
        # List of URIs representing instances / individuals
        self.instances = None
        self.num_instances = num_instances
        # Input arguments
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.storage_path = storage_path

        if self.args.quality_function_training == 'accuracy':
            self.quality_function = accuracy
        elif self.args.quality_function_training == 'fmeasure':
            self.quality_function = f_measure
        else:
            raise KeyError

    def __str__(self):
        return f'Trainer:\t|I|={self.num_instances}, ' \
               f'|D|={len(self.learning_problems)}, ' \
               f'|T|={len(self.learning_problems.target_class_expressions)}, ' \
               f'd:{self.args.num_embedding_dim}, ' \
               f'Quality func for training:{self.args.quality_function_training}, ' \
               f'NumEpoch={self.args.num_epochs}, ' \
               f'LR={self.args.learning_rate}, ' \
               f'BatchSize={self.args.batch_size}, ' \
               f'Device:{self.device}'

    def neural_architecture_selection(self):
        param = {'num_embedding_dim': self.args.num_embedding_dim,
                 'num_instances': self.num_instances,
                 'num_outputs': len(self.learning_problems.target_class_expressions)}

        arc = self.args.neural_architecture
        if arc == 'DeepSet':
            model = DeepSet(param)
        elif arc == 'DeepSetBase':
            model = DeepSetBase(param)
        elif arc == 'ST':
            model = ST(param)
        else:
            raise NotImplementedError(f'There is no {arc} model implemented')

        return NERO(model=model,
                    quality_func=self.quality_function,
                    target_class_expressions=self.learning_problems.target_class_expressions,
                    instance_idx_mapping=self.learning_problems.str_individuals_to_idx)

    def construct_dataloader(self):
        return torch.utils.data.DataLoader(
            Dataset(self.learning_problems,
                    num_workers_for_labelling=self.args.num_workers),
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers, shuffle=True)

    def batch_step(self, xpos, xneg, y):
        # (6.2) Zero the parameter gradients.
        self.optimizer.zero_grad()
        # (6.3) Forward.
        predictions = self.model.forward(xpos=xpos, xneg=xneg)
        # (6.4) Compute Loss.
        batch_loss = self.loss_func(input=predictions, target=y)
        # (6.5) Backward loss.
        batch_loss.backward()
        # (6.6) Update parameters according.
        self.optimizer.step()
        return batch_loss.detach().item()

    def epoch_step(self, data_loader):
        epoch_loss = 0
        for pos, neg, y in data_loader:
            epoch_loss += self.batch_step(pos.to(self.device), neg.to(self.device), y.to(self.device))
        return epoch_loss

    @timeit
    def training_loop(self) -> None:
        self.model.train()
        self.model.to(self.device)
        losses = []
        data_loader = self.construct_dataloader()
        print(f'{self.model},d:{self.args.num_embedding_dim}, '
              f'|Theta|={sum([p.numel() for p in self.model.parameters()])}, '
              f'Loss={self.loss_func}')
        # (5) Iterate training data.
        for it in range(self.args.num_epochs):
            # Compute Epoch loss.
            epoch_loss = self.epoch_step(data_loader)
            print(f'{it}.th epoch loss: {epoch_loss}')
            losses.append(epoch_loss)
            # Not working at the moment
            #self.validation(it)
        np.savetxt(fname=self.storage_path + "/loss_per_epoch.csv", X=np.array(losses), delimiter=",")

    def validation(self, ith_epoch: int):
        if ith_epoch % self.args.val_at_every_epochs == 0:
            self.model.eval()
            validate(self.model, random.choices(self.learning_problems, k=10), args={'topk': 3},
                     info='Validation on Training Data Starts')
            self.model.train()

    def start(self):
        """
         (1) Initialize the model
         (2) Select the loss function and the optimizer
         (3)

        Parameter: None
        ---------

        Returns: NERO
        ---------

        """
        # (1) Initialize the model.
        self.model = self.neural_architecture_selection()
        # (2) Select the loss function and Optimizer.
        self.loss_func, self.optimizer = select_loss_and_optim(loss_func_name=self.args.loss_func,
                                                               learning_rate=self.args.learning_rate,
                                                               model_params=self.model.parameters())
        # (4) Start training loop
        self.training_loop()

        self.model.eval()
        self.save_model(self.model)
        return self.model

    # @timeit
    def save_model(self, model):
        model.to('cpu')
        torch.save(model.state_dict(), self.storage_path + f'/final_model.pt')

        # (2) Serialize model and weights
        embeddings = model.embeddings_to_numpy()
        df = pd.DataFrame(embeddings, index=self.instances)
        df.to_csv(self.storage_path + '/instance_embeddings.csv')
        if self.args.plot_embeddings > 0:
            low_emb = PCA(n_components=2).fit_transform(embeddings)
            plt.scatter(low_emb[:, 0], low_emb[:, 1])
            plt.show()
