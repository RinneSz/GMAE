from model import GMAE_node
from data import GraphDataModule

from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from regression import generate_split, evaluate
import os


def cli_main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = GMAE_node.add_model_specific_args(parser)
    parser = GraphDataModule.add_argparse_args(parser)
    args = parser.parse_args()
    args.max_steps = args.tot_updates + 1
    if not args.test and not args.validate:
        print(args)
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    dm = GraphDataModule.from_argparse_args(args)
    n_node_features = dm.dataset.num_node_features

    # generate split
    split = generate_split(dm.dataset.data.num_nodes, train_ratio=0.1, val_ratio=0.1)

    # ------------
    # model
    # ------------
    if args.checkpoint_path != '':
        model = GMAE_node.load_from_checkpoint(
            args.checkpoint_path,
            strict=False,
            n_node_features=n_node_features,
            n_encoder_layers=args.n_encoder_layers,
            n_decoder_layers=args.n_decoder_layers,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            attention_dropout_rate=args.attention_dropout_rate,
            dropout_rate=args.dropout_rate,
            intput_dropout_rate=args.intput_dropout_rate,
            weight_decay=args.weight_decay,
            ffn_dim=args.ffn_dim,
            dataset_name=dm.dataset_name,
            warmup_updates=args.warmup_updates,
            tot_updates=args.tot_updates,
            peak_lr=args.peak_lr,
            end_lr=args.end_lr,
            mask_ratio=args.mask_ratio,
            n_val_sampler=dm.n_val_sampler,
        )
    else:
        model = GMAE_node(
            n_node_features=n_node_features,
            n_encoder_layers=args.n_encoder_layers,
            n_decoder_layers=args.n_decoder_layers,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            attention_dropout_rate=args.attention_dropout_rate,
            dropout_rate=args.dropout_rate,
            intput_dropout_rate=args.intput_dropout_rate,
            weight_decay=args.weight_decay,
            ffn_dim=args.ffn_dim,
            dataset_name=dm.dataset_name,
            warmup_updates=args.warmup_updates,
            tot_updates=args.tot_updates,
            peak_lr=args.peak_lr,
            end_lr=args.end_lr,
            mask_ratio=args.mask_ratio,
            n_val_sampler=dm.n_val_sampler,
        )
    if not args.test and not args.validate:
        print(model)
    print('total params:', sum(p.numel() for p in model.parameters()))

    # ------------
    # training
    # ------------
    metric = 'train_loss'
    dirpath = args.default_root_dir + f'/lightning_logs/checkpoints'
    checkpoint_callback = ModelCheckpoint(
        monitor=metric,
        dirpath=dirpath,
        filename=dm.dataset_name + '-{epoch:03d}-{' + metric + ':.4f}',
        save_top_k=2,
        mode='min',
        save_last=True,
    )
    if not args.test and not args.validate and os.path.exists(dirpath + '/last.ckpt'):
        args.resume_from_checkpoint = dirpath + '/last.ckpt'
        print('args.resume_from_checkpoint', args.resume_from_checkpoint)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.callbacks.append(checkpoint_callback)
    trainer.callbacks.append(LearningRateMonitor(logging_interval='step'))

    if not args.test and not args.validate:
        print('Evaluating.....')
        trainer.fit(model, datamodule=dm)
        acc_last = evaluate(model, dm, split)

        model = model.load_from_checkpoint(checkpoint_callback.best_model_path)
        acc_best = evaluate(model, dm, split)
        if acc_last > acc_best:
            acc_best = acc_last
        print('Acc:', acc_best)

    if args.test or args.validate:
        print('Evaluating.....')
        acc = evaluate(model, dm, split)
        print('Acc:', acc)


if __name__ == '__main__':
    cli_main()
