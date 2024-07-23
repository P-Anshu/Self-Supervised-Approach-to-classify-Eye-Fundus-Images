# DRetino

<p align="center">
    <img src="https://res.cloudinary.com/grohealth/image/upload/$wpsize_!_cld_full!,w_1200,h_630,c_scale/v1588090981/Symptoms-of-Diabetic-Retinopathy.png">
</p>

# DRetino 

A python library to create supervised diabetic retinopathy detection neural nets for ordinal regression using different loss functions.
Dretino is build on pytorch lightning and contains four different losses CrossEntropy, MeanSquared, Coral, Corn



- [Quick-start](#quick-start)




## Quick-start

* <a href="" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


* <a href=""><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open in Kaggle" /></a>


```pycon
from dretino.dataloader.build_features import DRDataModule
from dretino.models.train_model import Model, train

dm = DRDataModule(df_train, df_valid, df_test,
                  train_path,
                  valid_path,
                  test_path,
                  train_transforms,
                  val_transforms,
                  test_transforms,
                  num_workers=4,
                  batch_size=16)

args = dict(
        model_name='resnet50d',
        lr=3e-4,
        loss='mse',
        epochs=50,
        gpus=1,
        project='project_name',
        additional_layers=False
    )

train(Model,dm,**args)
```

## Results

We performed the above shown techniques on 2 diabetic retinopathy datasets

Aptos (contains 3k images)
Eyepacs (contains 30k images)


We first train the model with imagenet initialization on the datasets and then finetune to get the maximum accuracy for the given problem

And comparing the accuracy to supervised techniques

Using Eyepacs (30k images) also shows how easily it is scalable to more data

### Aptos

<div style="display:flex;justify-content:space-around;align-items:center;flex-flow:column;background-color:white;">

![Aptos LE](https://raw.githubusercontent.com/Dineswar11/dretino/master/reports/aptos_lineareval.png)

![Aptos FT](https://raw.githubusercontent.com/Dineswar11/dretino/master/reports/aptos_finetune.png)
</div>


### Eyepacs

<div style="display:flex;justify-content:space-around;align-items:center;flex-flow:column;background-color:white">


![EyePacs LE](https://raw.githubusercontent.com/Dineswar11/dretino/master/reports/eyepacs_linear_eval.png)

![EyePacs FT](https://raw.githubusercontent.com/Dineswar11/dretino/master/reports/eyepacs_finetune.png)


</div>