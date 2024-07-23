import os
import dotenv
import wandb
import albumentations as A
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from albumentations.pytorch import ToTensorV2
from dretino.dataloader.build_features import DRDataModule
from dretino.models.predict_model import test
from dretino.models.train_model import Model, train
from dretino.models.get_preds import create_preds_data_loader, get_perds
from dretino.visualization.visualize import show_images, cal_mean, plot_metrics
from sklearn.model_selection import train_test_split


def helper(
    wab,
    fast_dev_run,
    overfit_batches,
    num_classes=5,
    IMG_PATH="../data/processed/",
    **args
):
    file_name, trainer = train(
        Model,
        dm,
        wab=wab,
        fast_dev_run=fast_dev_run,
        overfit_batches=overfit_batches,
        **args
    )

    print(file_name)

    if not fast_dev_run:
        plot_metrics(file_name)

        test(
            Model,
            dm,
            file_name,
            trainer,
            wab=wab,
            fast_dev_run=fast_dev_run,
            overfit_batches=overfit_batches,
        )

        train_dataloader, val_dataloader, test_dataloader = create_preds_data_loader(
            df_train,
            df_valid,
            df_test,
            train_path=IMG_PATH + "images_resized",
            valid_path=IMG_PATH + "images_resized",
            test_path=IMG_PATH + "test_images_resized",
            transforms=test_transforms,
        )

        logits_test, y_test, cfn_mtx_test = get_perds(
            file_name, Model, test_dataloader, args["loss"], num_classes=5
        )

        logits_val, y_val, cfn_mtx_val = get_perds(
            file_name, Model, val_dataloader, args["loss"], num_classes=5
        )

        #         wandb.sklearn.plot_confusion_matrix(y, preds, labels=['0','1','2','3','4'])

        # plot_confusion_matrix(conf_mat=cfn_mtx,
        # class_names=['0', '1', '2', '3', '4'])

        # plt.show()

        return (
            logits_test,
            logits_val,
            y_test,
            y_val,
        )


def sweep():
    sweep_config = {
        "name": "HyperParameterSearch",
        "method": "random",  # Random search
        "metric": {  # We want to maximize val_accuracy
            "name": "test_accuracy",
            "goal": "maximize",
        },
        "parameters": {
            "num_neurons": {
                # Choose from pre-defined values
                "values": [512, 1024, 2048, 4096, 8192]
            },
            "num_layers": {
                # Choose from pre-defined values
                "values": [2, 3, 4, 5, 6]
            },
            "dropout": {"values": [0.2, 0.3, 0.4, 0.5]},
            "lr": {
                # log uniform distribution between exp(min) and exp(max)
                "distribution": "log_uniform",
                "min": -9.21,  # exp(-9.21) = 1e-4
                "max": -6.90,  # exp(-6.90) = 1e-3
            },
            "loss": {"values": ["ce", "corn"]},
        },
    }

    sweep_id = wandb.sweep(sweep_config, project="test_sweep")

    def sweep_iteration():
        # set up W&B logger
        wandb.init()  # required to have access to `wandb.config`
        args = dict(
            model_name="resnet50d",
            num_neurons=wandb.config.num_neurons,
            num_layers=wandb.config.num_layers,
            dropout=wandb.config.dropout,
            lr=wandb.config.lr,
            loss=wandb.config.loss,
            epochs=1,
            gpus=0,
            project="DRD",
            additional_layers=True,
            save_dir="reports",
        )

        logits, preds, y, cfn = helper(
            wab, fast_dev_run, overfit_batches, num_classes=5, **args
        )

    wandb.agent(sweep_id, function=sweep_iteration, count=3)


if __name__ == "__main__":
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    dotenv_path = os.path.join(project_dir, ".env")
    dotenv.load_dotenv(dotenv_path)

    PATH = "../data/processed/"
    dfx = pd.read_csv(
        PATH + "2.Groundtruths/a.IDRiD_Disease_Grading_Training_Labels.csv",
        usecols=["Image name", "Retinopathy grade"],
    )
    df_test = pd.read_csv(
        PATH + "2.Groundtruths/b.IDRiD_Disease_Grading_Testing_Labels.csv",
        usecols=["Image name", "Retinopathy grade"],
    )
    df_train, df_valid = train_test_split(
        dfx, test_size=0.2, random_state=42, stratify=dfx["Retinopathy grade"].values
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_transforms = A.Compose(
        [
            A.Resize(width=250, height=250),
            A.RandomCrop(height=224, width=224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Blur(p=0.15),
            A.CLAHE(p=0.15),
            A.ColorJitter(p=0.15),
            A.Affine(shear=30, rotate=0, p=0.1),
            A.Normalize(
                mean=(0.5211, 0.2514, 0.0809),
                std=(0.2653, 0.1499, 0.0861),
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=224, width=224),
            A.Normalize(
                mean=(0.5211, 0.2514, 0.0809),
                std=(0.2653, 0.1499, 0.0861),
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    test_transforms = A.Compose(
        [
            A.Resize(height=224, width=224),
            A.Normalize(
                mean=(0.5211, 0.2514, 0.0809),
                std=(0.2653, 0.1499, 0.0861),
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    dm = DRDataModule(
        df_train,
        df_valid,
        df_test,
        train_path=PATH + "images_resized",
        valid_path=PATH + "images_resized",
        test_path=PATH + "test_images_resized",
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        test_transforms=test_transforms,
        num_workers=4,
        batch_size=4,
    )

    wab = False
    fast_dev_run = False
    overfit_batches = False

    if wab:
        wandb.login(key=os.getenv("WANDB"))

    args = dict(
        model_name="resnet50d",
        num_neurons=512,
        num_layers=2,
        dropout=0.2,
        lr=3e-4,
        loss="ce",
        epochs=3,
        gpus=0,
        tpus=None,
        project="DRD",
        additional_layers=False,
        save_dir="reports",
    )

    logits_test, logits_val, y_test, y_val = helper(
        wab, fast_dev_run, overfit_batches, num_classes=5, **args
    )

    print(logits_test)
