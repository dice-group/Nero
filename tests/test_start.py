from main import argparse_default, Execute
class TestDefaultParams:
    def test(self):
        Execute(argparse_default([])).start()