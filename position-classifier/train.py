import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from classifier.trainer import Trainer

from classifier.utils import load_data
from classifier.utils import split_data

from classifier.models.cnn1d import PositionClassifier


def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--gpu_id", type=int, default=0 if torch.cuda.is_available() else -1)
    p.add_argument("--train_ratio", type=float, default=.8)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--dropout_p", type=float, default=.3)
    p.add_argument("--verbose", type=int, default=1)

    config = p.parse_args()

    return config


def main(config):
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    # 학습 데이터 불러오기
    x, y = load_data(pattern=5)
    x, y = split_data(x.to(device), y.to(device), train_ratio=config.train_ratio)

    # 입출력 데이터 크기 설정
    input_size = int(x[0].shape[-1])
    output_size = int(max(y[0])) + 1

    print("Train:", x[0].shape, y[0].shape)
    print("Valid:", x[1].shape, y[1].shape)
    
    model = PositionClassifier(input_size, output_size).to(device)
    optimizer = optim.Adam(model.parameters())
    crit = nn.NLLLoss()

    if config.verbose >= 1:
        print(model)
        print(optimizer)
        print(crit)

    trainer = Trainer(model, optimizer, crit)

    trainer.train(
        train_data=(x[0], y[0]),
        valid_data=(x[1], y[1]),
        config=config
    )

    torch.save({
        "model": trainer.model.state_dict(),
        "opt": optimizer.state_dict(),
        "config": config,
    }, config.weights)
    


if __name__ == '__main__':
    config = define_argparser()
    main(config)