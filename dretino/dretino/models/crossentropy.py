import timm
import torch
import torch.nn as nn


class ModelCE(nn.Module):
    def __init__(
        self,
        model_name,
        num_classes=5,
        additional_layers=True,
        num_neurons=512,
        n_layers=2,
        dropout_rate=0.2,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.additional_layers = additional_layers
        self.num_neurons = num_neurons
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate

        self.model = timm.create_model(
            self.model_name, pretrained=True, num_classes=self.num_classes
        )

        num_features = self.model.get_classifier().in_features

        if self.additional_layers:
            modules = [
                nn.BatchNorm1d(num_features),
                nn.Linear(in_features=num_features, out_features=self.num_neurons),
                nn.ReLU(),
                nn.BatchNorm1d(self.num_neurons),
                nn.Dropout(self.dropout_rate),
            ]

            for i in range(1, n_layers):
                modules.append(
                    nn.Linear(
                        in_features=self.num_neurons,
                        out_features=int(self.num_neurons / 2),
                    )
                )
                modules.append(nn.ReLU()),
                modules.append(nn.BatchNorm1d(int(self.num_neurons / 2)))
                modules.append(nn.Dropout(self.dropout_rate))
                self.num_neurons = int(self.num_neurons / 2)
            modules.append(
                nn.Linear(in_features=self.num_neurons, out_features=self.num_classes)
            )

            self.model.fc = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


def ce_loss(logits, y):
    """Calculate CrossEntropy Loss

    Parameters
    ----------
    logits : (torch.tensor), Logits returned by the model
    y : (torch.tensor), One hot encoded Ground Truths

    Returns
    -------
    loss : torch.tensor .grad_fn=True
    preds : torch.tensor
    y : torch.tesnor
    """
    y = torch.argmax(y, dim=-1)
    preds = torch.argmax(logits, dim=-1)
    loss = nn.CrossEntropyLoss()(logits, y)
    return loss, preds, y
