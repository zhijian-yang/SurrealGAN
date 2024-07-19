import os

import pandas as pd

from SurrealGAN.SurrealGAN.Surreal_GAN_representation_learning import (
    repetitive_representation_learning,
)

if __name__ == "__main__":
    output_dir = "./surrealgan_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_data = pd.read_csv("sample_data.csv")

    npattern = 3
    final_saving_epoch = 20000
    fold_number = 10

    repetitive_representation_learning(
        train_data,
        npattern,
        fold_number,
        1,
        final_saving_epoch,
        output_dir,
        lr=0.0008,
        batchsize=300,
        verbose=False,
        lipschitz_k=0.5,
        covariate=None,
        lam=0.4,
        gamma=0.8,
    )
