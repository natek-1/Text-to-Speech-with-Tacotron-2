import torch.nn as nn


class FeaturePredictNetLoss(nn.Module):
    '''
    Loss function for FeaturePredictNet
    '''
    def __init__(self):
        super(FeaturePredictNetLoss, self).__init__()

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (feat_predict, feat_residual_predict, stop_tokens_predict, attention_weights)
            targets: (feat_target, stop_tokens_target)
        Detail:
            stop_tokens_predict: stop_token value before sigmoid, do sigmoid in BCEWithLogitsLoss for numerical stabel
        Returns:
            loss: loss averaged over batch size and sequence length
            loss1: loss from raw results
            loss2: loss from residual results
            stop_loss: loss from stop tokens
        """
        feat_predict, feat_residual_predict, stop_tokens_predict, _ = inputs
        feat_target, stop_tokens_target = targets
        stop_tokens_predict = stop_tokens_predict.view(-1, 1)
        stop_tokens_target = stop_tokens_target.view(-1, 1)

        loss1 = nn.MSELoss()(feat_predict, feat_target) 
        loss2 =nn.MSELoss()(feat_residual_predict, feat_target)
        stop_loss = nn.BCEWithLogitsLoss()(stop_tokens_predict, stop_tokens_target)
        loss = loss1 + loss2 + stop_loss
        return loss, (loss1, loss2, stop_loss)