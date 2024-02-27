from train import Trainer
from test import Tester
from config import ConfigInit

def train():
    Trainer.train()


def test():
    Tester.test()


def main():
    train()
    # test()
    #Tester.test_one_OL()
    # Tester.test_one_COCO()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    config_obj = ConfigInit()
    main()
