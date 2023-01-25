import random
from logging import Logger

from ontolearn import KnowledgeBase

from typing import List, Tuple, Set, Dict
from owlapy.model import OWLEquivalentClassesAxiom, OWLClass, IRI, OWLObjectIntersectionOf, OWLObjectUnionOf, \
    OWLObjectSomeValuesFrom, OWLObjectInverseOf, OWLObjectProperty, OWLThing

from ontolearn.learning_problem_generator import LearningProblemGenerator
from owlapy.render import DLSyntaxObjectRenderer
from ontolearn.binders import DLLearnerBinder

from .static_funcs import *
from .util_classes import *
from .trainer import Trainer
from .expression import TargetClassExpression
import numpy as np
import pandas as pd
from collections import deque
import os
from random import randint
import time
import gc
import typing
random.seed(0)
torch.manual_seed(0)

class Experiment:
    """ Main class for conducting experiments """

    def __init__(self, args):
        self.args = args
        # (1) Create Logging & Experiment folder for serialization
        self.storage_path, _ = create_experiment_folder(main_folder_name='Experiments')
        self.logger = create_logger(name='Experimenter', p=self.storage_path)
        self.args['storage_path'] = self.storage_path
        # (2) Initialize KB.
        kb = self.initialize_knowledge_base()
        # (3) Initialize Training Data (D: {(E^+,E^-)})_i ^N .
        self.lp, self.rho = self.construct_targets_and_problems(kb)
        # (4) Init Trainer.
        self.trainer = Trainer(learning_problems=self.lp, args=self.args, logger=self.logger)

        self.list_str_individuals = list(self.lp.str_individuals_to_idx.keys())
        self.describe_and_store()
        del kb
        gc.collect()

    def initialize_knowledge_base(self) -> KnowledgeBase:
        """
        Initialize knowledge base

        Parameter: None
        ---------

        Returns: KnowledgeBase
        ---------

        """

        self.logger.info(f"Knowledge Base being Initialized {self.args['path_knowledge_base']}")
        # (1) Create ontolearn.KnowledgeBase instance
        # @TODO: Create KB class that uses polars or pandas to read and process an RDF KB.
        kb = KnowledgeBase(path=self.args['path_knowledge_base'],
                           reasoner_factory=ClosedWorld_ReasonerFactory)
        # (2) Store and Log some info about KB
        self.args['num_instances'] = kb.individuals_count()
        self.args['num_named_classes'] = len([i for i in kb.ontology().classes_in_signature()])
        self.args['num_properties'] = len([i for i in itertools.chain(kb.ontology().data_properties_in_signature(),
                                                                      kb.ontology().object_properties_in_signature())])
        self.logger.info(f'Number of individuals: {self.args["num_instances"]}')
        self.logger.info(f'Number of named classes / expressions: {self.args["num_named_classes"]}')
        self.logger.info(f'Number of properties / roles : {self.args["num_properties"]}')
        try:
            assert self.args['num_instances'] > 0
        except AssertionError:
            print(f'Number of entities can not be 0, *** {self.args["num_instances"]}')
            print('Background knowledge should be OWL 2.')
            exit(1)
        return kb

    def construct_targets_and_problems(self, kb: KnowledgeBase) -> typing.Tuple[LP, object]:
        """
        Construct target class expressions, i.e., choose good labels for a multi-label classification problem

        Parameter: KnowledgeBase
        ---------

        Returns: A tuple containing an LP object with a refinement operator object.
        ---------

        """
        # (1) Select target expressions according to input strategy
        rho = None
        target_class_expressions, instance_idx_mapping, rho = select_target_expressions(kb, self.args,
                                                                                        logger=self.logger)
        # e_pos, e_neg = generate_random_learning_problems(instance_idx_mapping, self.args)
        # (2) Generate training data points via sampling from targets.
        e_pos, e_neg = generate_learning_problems_from_targets(target_class_expressions, instance_idx_mapping,
                                                               self.args, logger=self.logger)
        # (3) Initialize a LP object to store training data and target expressions compactly.
        lp = LP(e_pos=e_pos, e_neg=e_neg, instance_idx_mapping=instance_idx_mapping,
                target_class_expressions=target_class_expressions)
        self.logger.info(lp)
        return lp, rho

    def describe_and_store(self) -> None:
        """
        Describe and save data

        Parameter:
        ---------

        Returns: None
        ---------

        """

        assert self.args['num_instances'] > 0
        self.logger.info('Experimental Setting is being serialized.')
        if torch.cuda.is_available():
            self.logger.info('Name of selected Device:{0}'.format(torch.cuda.get_device_name(self.trainer.device)))

        self.logger.info('Serialize Index of Instances.')
        # (2) Store Integer mapping of instance: index of individuals
        save_as_json(storage_path=self.storage_path,
                     obj=self.lp.str_individuals_to_idx, name='instance_idx_mapping')
        # (3) Store Target Class Expressions with respective expression chain from T -> ... -> TargetExp
        # Instead of storing as list of objects, we can store targets as pandas dataframe
        self.logger.info('Serialize Pandas Dataframe containing target expressions')
        df = pd.DataFrame([t.__dict__ for t in self.lp.target_class_expressions])
        df.to_csv(path_or_buf=self.storage_path + '/target_class_expressions.csv')
        del df
        gc.collect()
        self.args['num_outputs'] = len(self.lp.target_class_expressions)
        # (4) Store input settings
        save_as_json(storage_path=self.storage_path, obj=self.args, name='settings')
        # (5) Log details about input KB.

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

        # (1) Train model & Validate
        self.logger.info('Experiment starts')
        start_time = time.time()
        # (1) Train NERO.
        nero = self.trainer.start()
        # (2) Evaluate it on some learning problems.
        if self.args['path_lp'] is not None:
            # (2) Load learning problems
            with open(self.args['path_lp']) as json_file:
                settings = json.load(json_file)
            lp = [(k, list(v['positive_examples']), list(v['negative_examples'])) for k, v in
                  settings['problems'].items()]

            # (2) Evaluate model
            self.evaluate(nero, lp, self.args)

        self.logger.info(f'Total Runtime of the experiment:{time.time() - start_time}')

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
        self.logger.info(f'Evaluation Starts on {len(lp)} number of learning problems')
        nero.target_class_expressions = pd.read_csv(self.storage_path + '/target_class_expressions.csv', index_col=0)

        ncel_results = dict()
        # (1) Iterate over input learning problems.
        for _, (goal_exp, p, n) in enumerate(lp):
            ncel_report = nero.fit(str_pos=p, str_neg=n,
                                   topk=args['topK'],
                                   use_search=args['use_search'])
            ncel_report.update({'Target': goal_exp})

            ncel_results[_] = ncel_report

        # Overall Preport
        avg_f1_ncel = np.array([i['F-measure'] for i in ncel_results.values()]).mean()
        avg_runtime_ncel = np.array([i['Runtime'] for i in ncel_results.values()]).mean()
        avg_expression_ncel = np.array([i['NumClassTested'] for i in ncel_results.values()]).mean()
        self.logger.info(
            f'Average F-measure NCEL:{avg_f1_ncel}\t Avg. Runtime:{avg_runtime_ncel}\t Avg. Expression Tested:{avg_expression_ncel} in {len(lp)} ')
        self.logger.info('Evaluation Ends')
