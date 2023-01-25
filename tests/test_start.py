from main import argparse_default, Experiment
class TestDefaultParams:
    def test(self):
        Experiment(vars(argparse_default([]))).start()