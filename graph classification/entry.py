from model import GMAE_graph
from data import GraphDataModule
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import os
from regression import svc_classify


def cli_main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = GMAE_graph.add_model_specific_args(parser)
    parser = GraphDataModule.add_argparse_args(parser)
    args = parser.parse_args()

    if not args.test and not args.validate:
        print(args)
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    dm = GraphDataModule.from_argparse_args(args)
    if dm.dataset_name == 'ZINC':
        n_node_features = dm.dataset['train_dataset'].data.x.size(1)
        n_edge_features = None
        if dm.dataset['train_dataset'].data.edge_attr is not None:
            n_edge_features = 1
    else:
        n_node_features = dm.dataset['dataset'].data.x.size(1)
        n_edge_features = None
        if dm.dataset['dataset'].data.edge_attr is not None:
            n_edge_features = dm.dataset['dataset'].data.edge_attr.size(1)
    # ------------
    # model
    # ------------
    if args.checkpoint_path != '':
        model = GMAE_graph.load_from_checkpoint(
            args.checkpoint_path,
            strict=False,
            n_encoder_layers=args.n_encoder_layers,
            n_decoder_layers=args.n_decoder_layers,
            num_heads=args.num_heads,
            n_node_features=n_node_features,
            n_edge_features=n_edge_features,
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
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            mask_ratio=args.mask_ratio,
        )
    else:
        model = GMAE_graph(
            n_encoder_layers=args.n_encoder_layers,
            n_decoder_layers=args.n_decoder_layers,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            n_node_features=n_node_features,
            n_edge_features=n_edge_features,
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
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            mask_ratio=args.mask_ratio,
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

    # early stop
    dm.setup()
    len_trainloader = len(dm.train_dataloader())
    patience = args.early_stop_epoch * len_trainloader
    print(f'Early stop patience is {patience} updates.')
    earlystop_callback = EarlyStopping(
        monitor=metric,
        patience=patience,
        mode='min', # trainer will stop when the value stopped decreasing
        check_on_train_epoch_end=True,
    )
    trainer.callbacks.append(earlystop_callback)

    if not args.test and not args.validate:
        trainer.fit(model, datamodule=dm)
        if dm.dataset_name == 'ZINC':
            print('Pre-training finished!')
            exit()
        dm.setup()
        dataloader = dm.train_dataloader()
        acc_val, acc_test = svc_classify(model, dataloader)
        print(f'------Last Val Acc={acc_val}, Test Acc={acc_test}')
        model = model.load_from_checkpoint(checkpoint_callback.best_model_path)

    if args.test or args.validate:
        dm.setup()

    dataloader = dm.train_dataloader()
    acc_val, acc_test = svc_classify(model, dataloader)
    print(f'------Best Val Acc={acc_val}, Test Acc={acc_test}')

if __name__ == '__main__':
    cli_main()
