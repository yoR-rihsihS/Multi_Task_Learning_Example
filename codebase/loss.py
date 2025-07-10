import torch
import torch.nn as nn
import torchvision.ops as ops
import torch.nn.functional as F 

class Criterion(nn.Module):
    def __init__(self, task_specs, alpha=0.25, gamma=2.0):
        super(Criterion, self).__init__()
        self.task_specs = task_specs
        self.alpha = alpha
        self.gamma = gamma
        self.classification_losses = ["cross_entropy_loss", "focal_loss"]
        self.segmentation_losses = ["focal_loss", "dice_loss"]

    def cross_entropy_loss(self, outputs, targets, key):
        if key not in targets:
            return {}
        loss = {
            key+"_ce_loss": F.cross_entropy(outputs[key + "_logits"], targets[key])
        }
        return loss

    def focal_loss(self, outputs, targets, key):
        if key not in targets:
            return {}
        if "classification" in key:
            one_hot_targets = F.one_hot(targets[key], self.task_specs[key]).float()
        else:
            one_hot_targets = targets[key].float()
        loss = {
            key+"_focal_loss": ops.sigmoid_focal_loss(outputs[key + "_logits"], one_hot_targets, self.alpha, self.gamma, "mean")
        }
        return loss

    def dice_loss(self, outputs, targets, key):
        if key not in targets:
            return {}
        p = torch.sigmoid(outputs[key + "_logits"])
        intersection = (p * targets[key]).sum(dim=(2, 3))
        union = p.sum(dim=(2, 3)) + targets[key].sum(dim=(2, 3))
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        loss = {
            key+"_dice_loss" : 1 - dice.mean()
        }
        return loss

    def get_loss(self, loss, outputs, targets, key):
        function_map = {
            "focal_loss": self.focal_loss,
            "cross_entropy_loss": self.cross_entropy_loss,
            "dice_loss": self.dice_loss
        }
        assert loss in function_map, f"Do you really want to compute {loss} loss?"
        return function_map[loss](outputs, targets, key)

    def compute_gate_losses(self, gate_scores):
        def compute_l1_loss(score):
            return score.mean()
        
        def compute_load_balancing_loss(score):
            batch_probs = score.mean(dim=0)
            uniform = torch.full_like(batch_probs, 1.0 / score.shape[1])
            loss = F.kl_div(torch.log(batch_probs + 1e-8), uniform, reduction='batchmean')
            return loss
        
        def compuyte_entropy_loss(score):
            probs = score / (score.sum(dim=1, keepdim=True) + 1e-6)
            ent = -torch.sum(probs * torch.log(probs + 1e-6), dim=1)
            return -ent.mean()

        l1_loss = 0
        for k, v in gate_scores.items():
            l1_loss += compute_l1_loss(v) # to encourage sparcity among experts

        load_balancing_loss = 0
        for k, v in gate_scores.items():
            load_balancing_loss += compute_load_balancing_loss(v) # to encourage fair division of work

        entropy_loss = 0
        for k, v in gate_scores.items():
            entropy_loss += compuyte_entropy_loss(v) # to discourage one expert to always answer

        return {"l1_loss": l1_loss, "load_balancing_loss": load_balancing_loss, "entropy_loss": entropy_loss}


    def forward(self, outputs, targets, gate_scores):
        loss_dict = {}
        for key in self.task_specs:
            if "segmentation" in key:
                for loss in self.segmentation_losses:
                    loss_dict.update(self.get_loss(loss, outputs, targets, key))
            elif "classification" in key:
                for loss in self.classification_losses:
                    loss_dict.update(self.get_loss(loss, outputs, targets, key))
        # loss_dict.update(self.compute_gate_losses(gate_scores))
        return loss_dict, self.compute_gate_losses(gate_scores)
                
