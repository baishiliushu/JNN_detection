from trainCls import Trainer
from train import Trainer as T_no_cls
from test import Tester
from config import ConfigInit, Config

def train():
    t = Trainer()
    if 'cls' not in Config.network_type:
        t = T_no_cls()
    t.train()


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
