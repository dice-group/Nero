"""
====================================================================
Deep Tunnelling
====================================================================
Drill with training.
Authors: Caglar Demir

(1) Parse input knowledge base.
(2) Generate N number of random learning problems.
(3) Sample T number of class expressions as labels/targets.
(4) Generate F-measure distributed of (2) over (3).
(5) Train a permutation invariant neural net on (2) to approximate (4).
"""
from argparse import ArgumentParser

from core.experiment import Experiment
import torch
from torch import nn
import random

random.seed(0)
torch.manual_seed(0)


def main(args):
    Experiment(args).start()

# OWL version 1 and OWL 2, our library only works with OWL 2
if __name__ == '__main__':
    parser = ArgumentParser()
    # General
    parser.add_argument("--path_knowledge_base", type=str,
                        default='/home/demir/Desktop/Softwares/DeepTunnellingForRefinementOperators/KGs/Family/family-benchmark_rich_background.owl',
                        help='The absolute path of a knowledge base required.')
    parser.add_argument("--path_lp", type=str,
                        default="/home/demir/Desktop/Softwares/DeepTunnellingForRefinementOperators/LPs/Family/lp_dl_learner.json",
                        help='If None, examples are randomly generated')
    parser.add_argument("--dl_learner_binary_path", type=str, default='dllearner-1.4.0/')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of cpus used during batching')

    # Class Expression Learning
    parser.add_argument("--num_individual_per_example", type=int, default=10,
                        help='Input set size |E^+| and |E^-|.')
    parser.add_argument("--num_of_learning_problems_training", type=int, default=10,
                        help='Total number of LP => this val x |target exp|.')
    parser.add_argument("--num_of_learning_problems_testing", type=int, default=None,
                        help='Total number of randomly generated learning problems for testing.')

    # Neural related
    parser.add_argument("--neural_architecture", type=str,
                        default='DeepSet',
                        help='[ST(Set Transformer),DeepSet]')

    parser.add_argument("--number_of_target_expressions", type=int,
                        default=1_000,
                        help='Randomly select target class expressions as labels.')

    parser.add_argument("--target_expression_selection", type=str,
                        default='diverse_target_expression_selection',
                        help='[diverse_target_expression_selection, random_target_expression_selection]')
    parser.add_argument("--tolerance_for_search_unique_target_exp", type=int,
                        default=100,
                        help='Break refinement if its first expressions are not informed')

    # Hyperparameters of Neural Class Expression
    parser.add_argument("--num_embedding_dim", type=int, default=25, help='Number of embedding dimensions.')
    # Training Related
    parser.add_argument("--learning_rate", type=int, default=.001, help='Learning Rate')
    parser.add_argument("--num_epochs", type=int, default=100, help='Number of iterations over the entire dataset.')
    parser.add_argument("--val_at_every_epochs", type=int, default=2500, help='How often eval.')

    parser.add_argument("--batch_size", type=int, default=512)
    # Inference Related
    parser.add_argument("--topK", type=int, default=1,
                        help='Test the highest topK target expressions')

    # Analysis Related
    parser.add_argument("--plot_embeddings", type=int, default=0, help='1 => Yes, 0 => No')
    parser.add_argument("--eval_dl_learner", type=int, default=0, help='1 => Yes, 0 => No')
    main(vars(parser.parse_args()))