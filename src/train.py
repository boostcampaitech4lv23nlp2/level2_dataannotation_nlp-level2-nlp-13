import datetime
import logging

import pytorch_lightning as pl
import pytz
import torch
import wandb
from data_loader.data_loaders import KfoldDataloader
from model import model as module_arch
from pytorch_lightning.loggers import WandbLogger
from utils import logger, utils


def train(config):
    now_time = datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d-%H:%M:%S")
    wandb.init(
        entity=config.wandb.team_account_name,
        project=config.wandb.project_repo,
        name=f"{config.wandb.name}_{config.wandb.info}_{now_time}",
    )

    dataloader, model = utils.new_instance(config)
    assert config.k_fold.use_k_fold == isinstance(
        dataloader, KfoldDataloader
    ), "Check your config again: Make sure `k_fold.use_k_fold` is compatible with `dataloader.architecture`"

    wandb_logger = WandbLogger()
    save_path = f"{config.path.save_path}{config.model.name}_maxEpoch{config.train.max_epoch}_batchSize{config.train.batch_size}_{wandb_logger.experiment.name}/"
    wandb_logger.experiment.config.update({"save_dir": save_path})
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=config.train.max_epoch,
        log_every_n_steps=1,
        logger=wandb_logger,
        deterministic=True,
        precision=config.utils.precision,
        num_sanity_val_steps=int(config.k_fold.use_k_fold is not True),
        callbacks=[
            utils.early_stop(
                monitor=utils.monitor_config(key=config.utils.monitor, on_step=config.utils.on_step)["monitor"],
                mode=utils.monitor_config(key=config.utils.monitor, on_step=config.utils.on_step)["mode"],
                patience=config.utils.patience,
            ),
            utils.best_save(
                save_path=save_path,
                top_k=config.utils.top_k,
                monitor=utils.monitor_config(key=config.utils.monitor, on_step=config.utils.on_step)["monitor"],
                mode=utils.monitor_config(key=config.utils.monitor, on_step=config.utils.on_step)["mode"],
                filename="{epoch}-{step}-{val_loss}-{val_f1}",
            ),
        ]
        if not config.k_fold.use_k_fold
        else [
            utils.early_stop(
                monitor=utils.monitor_config(key=config.utils.monitor, on_step=config.utils.on_step)["monitor"],
                mode=utils.monitor_config(key=config.utils.monitor, on_step=config.utils.on_step)["mode"],
                patience=config.utils.patience,
            )
        ],
    )

    if config.k_fold.use_k_fold:
        if config.utils.on_step is False:
            assert config.utils.patience >= config.k_fold.num_folds, "The given 'config.utils.patience' is way too low."
        internal_fit_loop = trainer.fit_loop
        trainer.fit_loop = getattr(module_arch, "KFoldLoop")(config.k_fold.num_folds, export_path=save_path)
        trainer.fit_loop.connect(internal_fit_loop)
        trainer.fit(model=model, datamodule=dataloader, ckpt_path=config.path.resume_path)
    else:
        trainer.fit(model=model, datamodule=dataloader, ckpt_path=config.path.resume_path)
        trainer.test(model=model, datamodule=dataloader)  # K-fold CV runs test_step internally as part of fitting step

    wandb.finish()
    config["path"]["best_model_path"] = trainer.checkpoint_callback.best_model_path
    logger.log_config_yaml(config, save_path)

    # trainer.save_checkpoint(save_path + "model.ckpt")
    # model.plm.save_pretrained(save_path)
    # torch.save(model, save_path + "model.pt")


# def continue_train(args, config):
#     now_time = datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
#     wandb.init(
#         entity=config.wandb.team_account_name,
#         project=config.wandb.project_repo,
#         name=f"{config.wandb.name}_{config.wandb.info}",
#     )
#     dataloader, model = utils.new_instance(config)
#     model, args, config = utils.load_model(args, config, dataloader, model)
#     wandb_logger = WandbLogger(project=config.wandb.project)

#     save_path = f"{config.path.save_path}{config.model.name}_maxEpoch{config.train.max_epoch}_batchSize{config.train.batch_size}_{wandb_logger.experiment.name}_{now_time}/"
#     trainer = pl.Trainer(
#         accelerator="gpu",
#         devices=1,
#         max_epochs=config.train.max_epoch,
#         log_every_n_steps=1,
#         logger=wandb_logger,
#         callbacks=[
#             utils.early_stop(
#                 monitor=utils.monitor_config[config.utils.monitor]["monitor"],
#                 patience=config.utils.patience,
#                 mode=utils.monitor_config[config.utils.monitor]["mode"],
#             ),
#             utils.best_save(
#                 save_path=save_path,
#                 top_k=config.utils.top_k,
#                 monitor=utils.monitor_config[config.utils.monitor]["monitor"],
#                 mode=utils.monitor_config[config.utils.monitor]["mode"],
#                 filename="{epoch}-{step}-{val_loss}-{val_f1}",
#             ),
#         ],
#     )

#     trainer.fit(model=model, datamodule=dataloader)
#     trainer.test(model=model, datamodule=dataloader)
#     wandb.finish()

#     trainer.save_checkpoint(save_path + "model.ckpt")
#     model.plm.save_pretrained(save_path)
#     # torch.save(model, save_path + "model.pt")


# def k_train(config):
#     project_name = config.wandb.project

#     results = []
#     num_folds = config.k_fold.num_folds

#     exp_name = WandbLogger(project=project_name).experiment.name
#     for k in range(num_folds):
#         k_datamodule = KfoldDataloader(k, config)

#         Kmodel = module_arch.Model(
#             config.model.name,
#             config.train.learning_rate,
#             config.train.loss,
#             k_datamodule.new_vocab_size,
#             config.train.use_frozen,
#         )

#         if k + 1 == 1:
#             name_ = f"{k+1}st_fold"
#         elif k + 1 == 2:
#             name_ = f"{k+1}nd_fold"
#         elif k + 1 == 3:
#             name_ = f"{k+1}rd_fold"
#         else:
#             name_ = f"{k+1}th_fold"
#         wandb_logger = WandbLogger(project=project_name, name=exp_name + f"_{name_}")
#         save_path = f"{config.path.save_path}{config.model.name}_maxEpoch{config.train.max_epoch}_batchSize{config.train.batch_size}_{wandb_logger.experiment.name}_{name_}/"
#         trainer = pl.Trainer(
#             accelerator="gpu",
#             devices=1,
#             max_epochs=config.train.max_epoch,
#             log_every_n_steps=1,
#             logger=wandb_logger,
#             deterministic=True,
#             precision=config.utils.precision,
#             callbacks=[
#                 utils.early_stop(
#                     monitor=utils.monitor_config[config.utils.monitor]["monitor"],
#                     patience=config.utils.patience,
#                     mode=utils.monitor_config[config.utils.monitor]["mode"],
#                 ),
#                 utils.best_save(
#                     save_path=save_path,
#                     top_k=config.utils.top_k,
#                     monitor=utils.monitor_config[config.utils.monitor]["monitor"],
#                     mode=utils.monitor_config[config.utils.monitor]["mode"],
#                     filename="{epoch}-{step}-{val_loss}-{val_f1}",
#                 ),
#             ],
#         )

#         trainer.fit(model=Kmodel, datamodule=k_datamodule)
#         score = trainer.test(model=Kmodel, datamodule=k_datamodule)
#         wandb.finish()

#         results.extend(score)
#         # torch.save(Kmodel, save_path + f"{name_} model.pt")
#         trainer.save_checkpoint(save_path + f"{name_} model.ckpt")

#     result = [x["test_pearson"] for x in results]
#     score = sum(result) / num_folds
#     print(f"{num_folds}-fold pearson 평균 점수: {score}")


def sweep(config, exp_count):
    project_name = config.wandb.project

    sweep_config = {
        "method": "bayes",
        "parameters": {
            "lr": {
                "distribution": "uniform",
                "min": 1e-5,
                "max": 3e-5,
            },
        },
        "early_terminate": {
            "type": "hyperband",
            "max_iter": 30,
            "s": 2,
        },
    }

    sweep_config["metric"] = {"name": "test_pearson", "goal": "maximize"}

    def sweep_train(config=None):
        wandb.init(config=config)
        config = wandb.config

        dataloader, model = utils.new_instance(config, config=None)

        wandb_logger = WandbLogger(project=project_name)
        save_path = f"{config.path.save_path}{config.model.name}_sweep_id_{wandb.run.name}/"
        trainer = pl.Trainer(
            gpus=1,
            max_epochs=config.train.max_epoch,
            logger=wandb_logger,
            log_every_n_steps=1,
            deterministic=True,
            precision=config.utils.precision,
            callbacks=[
                utils.early_stop(
                    monitor=utils.monitor_config[config.utils.monitor]["monitor"],
                    patience=config.utils.patience,
                    mode=utils.monitor_config[config.utils.monitor]["mode"],
                ),
                utils.best_save(
                    save_path=save_path,
                    top_k=config.utils.top_k,
                    monitor=utils.monitor_config[config.utils.monitor]["monitor"],
                    mode=utils.monitor_config[config.utils.monitor]["mode"],
                    filename="{epoch}-{step}-{val_loss}-{val_f1}",
                ),
            ],
        )
        trainer.fit(model=model, datamodule=dataloader)
        trainer.test(model=model, datamodule=dataloader)
        trainer.save_checkpoint(save_path + "model.ckpt")
        # torch.save(model, save_path + "model.pt")

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=project_name,
    )

    wandb.agent(sweep_id=sweep_id, function=sweep_train, count=exp_count)
