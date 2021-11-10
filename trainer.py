from ontolearn import KnowledgeBase
from typing import List, Tuple, Set
from owlapy.model import OWLEquivalentClassesAxiom, OWLClass, IRI, OWLObjectIntersectionOf, OWLObjectUnionOf, \
    OWLObjectSomeValuesFrom, OWLObjectInverseOf, OWLObjectProperty, OWLThing
from argparse import ArgumentParser

from owlapy.render import DLSyntaxObjectRenderer

import random
from collections import deque
from model import DT

import torch
from torch import nn
import numpy as np
from static_funcs import *
from util_classes import *
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class Trainer:
    def __init__(self, knowledge_base: KnowledgeBase, learning_problems, args):
        self.knowledge_base = knowledge_base
        self.learning_problems = learning_problems
        # List of URIs representing instances / individuals
        self.instances = None
        # Input arguments
        self.args = args
        # Create an experiment folder
        self.storage_path, _ = create_experiment_folder(folder_name='Experiments')
        self.logger = create_logger(name='Trainer', p=self.storage_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.renderer = DLSyntaxObjectRenderer()

        # Describe and store the setting of the trainer
        self.__describe_and_store_setting()

    def __describe_and_store_setting(self):
        # cuda device
        self.logger.info('Device:{0}'.format(self.device))
        if torch.cuda.is_available():
            self.logger.info('Name of selected Device:{0}'.format(torch.cuda.get_device_name(self.device)))

        pd.DataFrame(self.learning_problems.data_points).to_csv(self.storage_path + '/GeneratedDataPoints.csv')
        self.save_as_json(self.learning_problems.instance_idx_mapping, name='instance_idx_mapping')
        self.save_as_json({i: {'DL-Syntax': self.renderer.render(cl.concept),
                               'ExpressionChain': [self.renderer.render(_.concept) for _ in retrieve_concept_chain(cl)]}
                           for i, cl in
                           enumerate(self.learning_problems.target_class_expressions)}, name='target_class_expressions')
        self.logger.info('Learning Problem object serialized')
        self.save_as_json(self.args, name='settings')

    def save_as_json(self, obj, name=None):
        with open(self.storage_path + f'/{name}.json', 'w') as file_descriptor:
            json.dump(obj, file_descriptor, indent=3)

    def training_loop(self):

        self.logger.info('Expensive Labelling Starts')
        data_loader = torch.utils.data.DataLoader(Dataset(self.learning_problems),
                                                  batch_size=self.args['batch_size'],
                                                  num_workers=0, shuffle=True)
        self.logger.info('Expensive Labelling Ends')

        # (2) Initialize DT
        model = DT(param={'num_embedding_dim': self.args['num_embedding_dim'],
                          'num_instances': self.knowledge_base.individuals_count(),
                          'num_outputs': len(self.learning_problems.target_idx_individuals)})

        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args['learning_rate'])

        self.logger.info('Training Loop starts')
        # (1) Set model in training mode.
        model.train()
        # (2) Send model to selected device.
        model.to(self.device)
        # (3) Store average loss per epoch
        losses = []
        # (4) Start training loop
        printout_constant = (self.args['num_epochs'] // 20) + 1
        start_time = time.time()
        for it in range(1, self.args['num_epochs'] + 1):
            epoch_loss = 0
            # (5) Mini-batch.
            for x, y in data_loader:
                # (5.1) Send the batch into device.
                x, y = x.to(self.device), y.to(self.device)

                # (5.2) Zero the parameter gradients.
                optimizer.zero_grad()
                # (5.3) Forward.
                predictions = model(x)
                # (5.4) Compute Loss.
                batch_loss = loss_func(y, predictions)
                epoch_loss += batch_loss.item()
                # (5.5) Backward loss.
                batch_loss.backward()
                # (5.6) Update parameters according.
                optimizer.step()
            # (6) Store epoch loss
            losses.append(epoch_loss)

            # (7) Print-out
            if it % printout_constant == 0:
                self.logger.info(f'{it}.th epoch loss: {epoch_loss}')
        training_time = time.time() - start_time
        # Save
        self.logger.info(f'TrainingRunTime {training_time / 60:.3f} minutes')
        self.logger.info('Save the loss epoch trajectory')
        np.savetxt(fname=self.storage_path + "/loss_per_epoch.csv", X=np.array(losses), delimiter=",")
        self.logger.info('Save Weights')
        save_weights(model, self.storage_path)

        model.eval()
        self.logger.info('Training Loop ends')
        return model

    def start(self):
        """
        Training starts
        :return:
        """
        self.logger.info('Start')
        model = self.training_loop()

        embeddings = model.embeddings.weight.data.detach().numpy()
        df = pd.DataFrame(embeddings, index=self.instances)
        df.to_csv(self.storage_path + '/instance_embeddings.csv')

        if self.args['plot_embeddings'] > 0:
            # import umap
            # reducer = umap.UMAP()
            # low_emb = reducer.fit_transform(embeddings)
            low_emb = PCA(n_components=2).fit_transform(embeddings)
            plt.scatter(low_emb[:, 0], low_emb[:, 1])
            plt.show()
