import torch
import torch.nn.functional as F

class BalancedCrossEntropy(torch.nn.Module):
    def __init__(self, freq):
        super(BalancedCrossEntropy, self).__init__()
        self.sample_per_class = freq

    def forward(self, input, labels, reduction='mean'):
        spc = self.sample_per_class.type_as(input)
        spc = spc.unsqueeze(0).expand(input.shape[0], -1)
        input = input + spc.log()
        loss = torch.nn.functional.cross_entropy(input=input, target=labels, reduction=reduction)
        return loss
    
def L2_loss(student_logits, teacher_logits):
    """
    Compute L2-based knowledge distillation loss between student and teacher logits.
    NB : Only compares logits over previously known classes.
    """

    return F.mse_loss(student_logits, teacher_logits, reduction='mean')

def KL_loss(student_logits, teacher_logits, T=2.0, reduction='batchmean'):
    """
    Compute the distillation loss between student and teacher logits with KL divergence.

    Args:
        student_logits (Tensor): Logits from the student model. Shape: [B, C]
        teacher_logits (Tensor): Logits from the teacher model. Shape: [B, C]
        T (float): Temperature parameter. Usually > 1.0
        reduction (str): Reduction method for KLDivLoss ('batchmean' is recommended)
    NB : see https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html
    on log_target = False by default and on use of reduction = 'batchmean' to align with KL math definition.
    
    Returns:
        Tensor: Distillation loss (scalar)
    """
    # Compute softened probabilities
    student_log_probs = F.log_softmax(student_logits / T, dim=1)
    teacher_probs = F.softmax(teacher_logits / T, dim=1)

    # Compute the KL divergence
    loss = F.kl_div(student_log_probs, teacher_probs, reduction=reduction) * (T * T)
    return loss

def cos_loss(student_repr, teacher_repr, reduction='mean'):
    """
    Compute cosine embedding loss between student and teacher representations.
    Input:
        student_repr: Tensor of shape [B, D]
        teacher_repr: Tensor of shape [B, D]
        reduction: 'mean' or 'sum'
    Output:
        Cosine embedding loss (scalar tensor)
    """
    # Target = 1 means we want cosine similarity to be high (aligned representations)
    target = torch.ones(student_repr.size(0), device=student_repr.device)
    loss = F.cosine_embedding_loss(student_repr, teacher_repr, target, reduction=reduction)
    return loss

