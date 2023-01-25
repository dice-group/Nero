import json
import random

from ontolearn import KnowledgeBase

from typing import List, Tuple, Set, Dict
from owlapy.model import OWLEquivalentClassesAxiom, OWLClass, IRI, OWLObjectIntersectionOf, OWLObjectUnionOf, \
    OWLObjectSomeValuesFrom, OWLObjectInverseOf, OWLObjectProperty, OWLThing

from ontolearn.learning_problem_generator import LearningProblemGenerator
from owlapy.render import DLSyntaxObjectRenderer
from ontolearn.binders import DLLearnerBinder

from .static_funcs import create_experiment_folder, create_logger, select_target_expressions, \
    generate_learning_problems_from_targets, save_as_json, timeit
from .util_classes import LP
from .trainer import Trainer
# from .expression import TargetClassExpression
import numpy as np
import pandas as pd
from collections import deque
import os
import time
import gc
import typing
import torch
import random
import itertools

random.seed(0)
torch.manual_seed(0)


class Execute:

    def __init__(self, args):
        self.args = args
        # (1) Create Logging & Experiment folder for serialization
        self.storage_path, _ = create_experiment_folder(main_folder_name=args.storage_path)
        self.logger = create_logger(name='Experimenter', p=self.storage_path)
        self.report = dict()
        self.report['storage_path'] = self.storage_path
        save_as_json(storage_path=self.storage_path, obj=vars(self.args), name='settings')

    @timeit
    def initialize_knowledge_base(self, backend: str) -> KnowledgeBase:
        """
        Initialize knowledge base

        Parameter:
        ---------

        backend:str

        Returns: KnowledgeBase
        ---------

        """

        print(f"Knowledge Base being Initialized {self.args.path_knowledge_base}")
        kb, num_instances, num_named_classes, num_properties = read_rdf_knowledge_base(
            path=self.args.path_knowledge_base, backend=backend)
        self.report.update(
            {'num_instances': num_instances, 'num_named_classes': num_named_classes, 'num_properties': num_properties})
        print(f'Number of individuals: {self.report["num_instances"]}')
        print(f'Number of named classes / expressions: {self.report["num_named_classes"]}')
        print(f'Number of properties / roles : {self.report["num_properties"]}')
        return kb

    def construct_targets_and_problems(self, kb: KnowledgeBase) -> typing.Tuple:
        """
        Construct target class expressions, i.e., choose good labels for a multi-label classification problem

        Parameter: KnowledgeBase
        ---------

        Returns: A tuple containing an LP object with a refinement operator object.
        ---------

        """
        # (1) Select target expressions according to input strategy
        rho = None
        target_class_expressions, instance_idx_mapping, rho = select_target_expressions(kb, self.args)
        # e_pos, e_neg = generate_random_learning_problems(instance_idx_mapping, self.args)
        # (2) Generate training data points via sampling from targets.
        e_pos, e_neg = generate_learning_problems_from_targets(target_class_expressions, instance_idx_mapping,
                                                               self.args)
        # (3) Initialize a LP object to store training data and target expressions compactly.
        lp = LP(e_pos=e_pos, e_neg=e_neg, instance_idx_mapping=instance_idx_mapping,
                target_class_expressions=target_class_expressions)
        return lp, rho

    @timeit
    def training(self, kb):
        learning_problems, _ = self.construct_targets_and_problems(kb)
        save_as_json(storage_path=self.storage_path,
                     obj=learning_problems.str_individuals_to_idx, name='instance_idx_mapping')

        pd.DataFrame([t.__dict__ for t in learning_problems.target_class_expressions]).to_csv(path_or_buf=self.storage_path + '/target_class_expressions.csv')

        self.report['num_outputs'] = len(learning_problems.target_class_expressions)
        trainer = Trainer(learning_problems=learning_problems, args=self.args, storage_path=self.storage_path,
                          num_instances=self.report['num_instances'])
        print(trainer)
        return trainer

    def start(self) -> None:
        """
         Train and Eval NERO

        (1) Train Nero on generated learning problems
        (2) Eval Nero if learning problems are given.

        Parameter: None
        ---------

        Returns: None
        ---------

        """
        start_time = time.time()
        # Initialize Trainer
        trainer = self.training(self.initialize_knowledge_base(backend=self.args.backend))
        # (1) Train NERO.
        trained_nero = trainer.start()

        # (2) Evaluate it on some learning problems.
        if self.args.path_lp is not None:
            # (2) Load learning problems
            with open(self.args.path_lp) as json_file:
                settings = json.load(json_file)
            lp = [(k, list(v['positive_examples']), list(v['negative_examples'])) for k, v in
                  settings['problems'].items()]

            # (2) Evaluate model
            self.evaluate(trained_nero, lp, self.args)

        print(f'Total Runtime of the experiment:{time.time() - start_time}')

    def evaluate(self, nero, lp, args) -> None:
        """
         Evalueate a pre-trained model

        Parameter: None
        ---------
        ncel:

        lp:

        args:
        Returns: None
        ---------

        """
        print(f'Evaluation Starts on {len(lp)} number of learning problems')
        nero.target_class_expressions = pd.read_csv(self.storage_path + '/target_class_expressions.csv', index_col=0)

        ncel_results = dict()
        # (1) Iterate over input learning problems.
        for _, (goal_exp, p, n) in enumerate(lp):
            ncel_report = nero.fit(str_pos=p, str_neg=n,
                                   topk=args.topK,
                                   use_search=args.use_search)
            ncel_report.update({'Target': goal_exp})

            ncel_results[_] = ncel_report

        # Overall Preport
        avg_f1_ncel = np.array([i['F-measure'] for i in ncel_results.values()]).mean()
        avg_runtime_ncel = np.array([i['Runtime'] for i in ncel_results.values()]).mean()
        avg_expression_ncel = np.array([i['NumClassTested'] for i in ncel_results.values()]).mean()
        print(
            f'Average F-measure NERO:{avg_f1_ncel}\t Avg. Runtime:{avg_runtime_ncel}\t Avg. Expression Tested:{avg_expression_ncel} in {len(lp)} ')


def read_rdf_knowledge_base(path, backend):
    num_instances, num_named_classes, num_properties = None, None, None
    if backend == 'ontolearn':
        from .static_funcs import ClosedWorld_ReasonerFactory
        kb = KnowledgeBase(path=path, reasoner_factory=ClosedWorld_ReasonerFactory)
        num_instances = kb.individuals_count()
        num_named_classes = len([i for i in kb.ontology().classes_in_signature()])
        num_properties = len([i for i in itertools.chain(kb.ontology().data_properties_in_signature(),
                                                         kb.ontology().object_properties_in_signature())])
    elif backend=='pandas':
        raise NotImplementedError(f'backend pandas has not yet been implemented.')
    elif backend == 'polars':
        raise NotImplementedError(f'backend pandas has not yet been implemented.')
    else:
        raise KeyError(f'Incorrect backend parameter {backend}')

    assert num_instances > 0 and num_named_classes > 0 and num_properties > 0

    return kb, num_instances, num_named_classes, num_properties
