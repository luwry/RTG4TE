# -*-coding:utf-8-*-
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file',type=str, default='data/ree_train.json')
    parser.add_argument('--val-file', type=str, default='data/ree_dev.json')
    parser.add_argument('--test-file', type=str, default='data/ree_test.json')
    parser.add_argument('--tmp_dir', default='./tmp')