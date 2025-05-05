import torch
from torch.nn.modules.loss import _Loss
from monai.losses.dice import DiceLoss


class DiceBCELoss(_Loss):
    """
    Compute both Dice loss and BCE Loss, and return the weighted sum of these two losses.
    The details of Dice loss is shown in ``monai.losses.DiceLoss``.
    The details of BCE Loss is shown in ``torch.nn.BCEWithLogitsLoss``.
    """

    def __init__(
        self,
        sigmoid: bool = True,
        squared_pred: bool = True,
        reduction: str = "mean",
        pos_weight: torch.Tensor | None = None,
        lambda_dice: float = 1.0,
        lambda_bce: float = 1.0,
    ) -> None:
        """
        Args:
            ``pos_weight`` and ``lambda_bce`` are only used for cross entropy loss.
            ``reduction`` is used for both losses and other parameters are only used for dice loss.

            sigmoid: if True, apply a sigmoid function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `CrossEntropyLoss`.
            reduction: {``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``. The dice loss should
                as least reduce the spatial dimensions, which is different from cross entropy loss, thus here
                the ``none`` option cannot be used.

                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
            pos_weight: a rescaling weight given to positive examples for cross entropy loss.
                See ``torch.nn.BCEWithLogitsLoss()`` for more information.
            lambda_dice: the trade-off weight value for dice loss. The value should be no less than 0.0.
                Defaults to 1.0.
            lambda_bce: the trade-off weight value for bce loss. Defaults to 1.0.

        """
        super().__init__()
        self.dice = DiceLoss(sigmoid=sigmoid, squared_pred=squared_pred, reduction=reduction)
        self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_bce

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 nor the same as input.

        """
        if len(input.shape) != len(target.shape):
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {input.shape} and {target.shape}."
            )

        dice_loss = self.dice(input, target).squeeze()
        ce_loss = self.bce(input, target.float())
        ce_loss = ce_loss.mean(dim=tuple(range(1, ce_loss.dim())))
        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_ce * ce_loss

        return total_loss