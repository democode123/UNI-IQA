import argparse
import train as TrainModel
import scipy.io as sio
import os


def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument("--CAM", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=19901116)

    parser.add_argument('--scnn_root', type=str, default='saved_weights/scnn.pkl')

    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--trainset", type=str, default="./databases/")
    parser.add_argument("--live_set", type=str, default="./databases/databaserelease2/")
    parser.add_argument("--QACS_set", type=str, default="./databases/QACS/")

    parser.add_argument("--eval_live", type=bool, default=True)
    parser.add_argument("--eval_QACS", type=bool, default=True)



    parser.add_argument('--ckpt_path', default='./checkpoint', type=str,
                        metavar='PATH', help='path to checkpoints')
    # parser.add_argument('--ckpt', default=None, type=str, help='name of the checkpoint to load')

    parser.add_argument("--train_txt", type=str, default='train.txt') # train.txt | train_synthetic.txt | train_authentic.txt | train_sub2.txt | train_score.txt
    parser.add_argument("--ranking", type=bool, default=True)  # True for learning-to-rank False for regular regression
    parser.add_argument("--batch_size", type=int, default=64)
    # parser.add_argument("--batch_size2", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224, help='None means random resolution')
    parser.add_argument("--max_epochs", type=int, default=100)
    # parser.add_argument("--max_epochs2", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay_interval", type=int, default=3)
    parser.add_argument("--decay_ratio", type=float, default=0.1)
    parser.add_argument("--epochs_per_eval", type=int, default=5)
    parser.add_argument("--epochs_per_save", type=int, default=5)

    return parser.parse_args()


def main(cfg):
    t = TrainModel.Trainer(cfg)
    if cfg.train:
        t.fit()
    else:
        test_results_srcc, test_results_plcc = t.eval()
        out_str = 'Testing: LIVE SRCC: {:.4f} QACS SRCC: {:.4f}'.format(test_results_srcc['live'],test_results_srcc['QACS'])
        out_str2 = 'Testing: LIVE PLCC: {:.4f} QACS PLCC: {:.4f}'.format(test_results_plcc['live'],test_results_plcc['QACS'])
        print(out_str)
        print(out_str2)

if __name__ == "__main__":
    config = parse_config()
    for i in range(0, 100):
        config = parse_config()
        split = i + 1
        config.split = split
        config.ckpt_path = os.path.join(config.ckpt_path, '1')
        if not os.path.exists(config.ckpt_path):
            os.makedirs(config.ckpt_path)

        main(config)







