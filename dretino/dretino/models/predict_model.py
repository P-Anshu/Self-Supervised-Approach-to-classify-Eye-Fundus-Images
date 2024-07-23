def test(
    Model, dm, file_name, trainer, wab=False, fast_dev_run=False, overfit_batches=False
):
    """Testing on the test dataset

    Parametres
    ----------
    Model : (LightningModule), Model
    dm : (LightningDataModule), DataModule
    file_name : (str), Filename created using datetime
    trainer : (LightningTrainer), Trainer used in training
    wab : (bool, optional), Wandb integration. Defaults to False.
    fast_dev_run : (bool, optional), Fast dev run for unit test of the model code. Defaults to False.
    overfit_batches : (bool, optional), Used to overfit batches for sanity check. Defaults to False

    Returns
    -------
    None
    """
    if fast_dev_run:
        return
    elif overfit_batches:
        return
    else:
        model = Model.load_from_checkpoint(file_name + ".ckpt")
        trainer.test(model, dm)
