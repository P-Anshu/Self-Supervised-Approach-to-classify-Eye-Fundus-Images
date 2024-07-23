import timm
import torch
import torch.nn as nn


class ModelMSE(nn.Module):
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
            modules.append(nn.Linear(in_features=self.num_neurons, out_features=1))

            self.model.fc = nn.Sequential(*modules)
        else:
            self.model.fc = nn.Linear(in_features=num_features, out_features=1)

    def forward(self, x):
        return self.model(x)


def mse_loss(logits, y):
    """Calculate MSE Loss

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
    y = torch.unsqueeze(y, 1).to(torch.float32)
    loss = nn.MSELoss()(logits, y)
    predictions = logits.data
    predictions[predictions < 0.5] = 0
    predictions[(predictions >= 0.5) & (predictions < 1.5)] = 1
    predictions[(predictions >= 1.5) & (predictions < 2.5)] = 2
    predictions[(predictions >= 2.5) & (predictions < 3.5)] = 3
    predictions[(predictions >= 3.5) & (predictions < 1000000000000)] = 4
    preds = predictions.long().view(-1)
    y = y.to(torch.int8).view(-1)
    return loss, preds, y
