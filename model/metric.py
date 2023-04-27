import torch

def create_mask(original_mask, batch_size, window_size):
    pos_matrix = torch.arange(batch_size).reshape((batch_size, 1))
    pos_matrix_bool = pos_matrix < pos_matrix.T
    mask_per_word = torch.abs(pos_matrix - pos_matrix.T) <= window_size
    mask_per_word = torch.logical_and(mask_per_word,pos_matrix_bool)
    mask_per_word = (mask_per_word * original_mask.unsqueeze(1) * original_mask.unsqueeze(0))
    mask_per_word.fill_diagonal_(0)
    return mask_per_word


def split_and_modify_mask(mask):
    """
    Split the mask tensor into two tensors and modify their values as required.

    Args:
        mask (torch.Tensor): Binary tensor of shape [batch_size, batch_size].

    Returns:
        torch.Tensor: Binary tensor of shape [batch_size, batch_size].
    """

    left_tensor = (torch.triu(torch.ones(mask.shape), diagonal=0) + mask).bool().int()
    right_tensor = (torch.tril(torch.ones(mask.shape), diagonal=0) + mask).bool().int()
    mask_left = torch.flip(torch.cumsum(torch.flip(left_tensor == 0, dims=[1]), dim=1) > 0, dims=[1]).int()

    # Apply the mask to the input tensor
    masked_left_tensor = left_tensor * (1 - mask_left)

    mask_right = (torch.cumsum(right_tensor == 0, dim=1) > 0).int()
    # Apply the mask to the input tensor
    masked_right_tensor = right_tensor * (1 - mask_right)

    # Combine the modified left and right tensors with element-wise addition
    modified_mask = torch.logical_and(masked_left_tensor, masked_right_tensor).fill_diagonal_(0).int()

    return modified_mask

# not really recall because we take the top k values of probits and just all values above zero that with target , instead im gonna go with
def recall_k(probits, target, mask,k=5, window=2):

    acc_r = 0.0
    acc_rt = 0.0

    count = 0
    count_rt=0
    mask_per_word = create_mask(mask, len(target), window)
    mask_after_diagonal = split_and_modify_mask(mask_per_word).float()
    # print("mask after diagonal:",mask_after_diagonal)
    # print("shape of mask and x",mask_after_diagonal.shape,x.shape)
    # print("shapes of probits and target before masking:", probits.shape, target.shape)
    # print("target before masking:", target)
    # print("probits before masking:", probits)
    target_masked = torch.matmul(mask_after_diagonal, target.float())
    probits_masked = torch.matmul(mask_after_diagonal, probits.float())
    # print("target masked:", target_masked)
    # print("probits masked:", probits_masked)
    target_masked = target_masked[torch.any(target_masked != 0, dim=1)].bool().float()
    probits_masked = probits_masked[torch.any(probits_masked != 0, dim=1)]
    # print("shapes of probits and target after masking:", probits_masked.shape, target_masked.shape)

    k = 5
    _, tk = torch.topk(probits_masked, k)
    # print("tk is:",tk)
    # print("target masked is:",target_masked)
    tt = torch.gather(target_masked, 1, tk)
    # print("tt is:", tt)
    r = torch.mean(torch.sum(tt, dim=1))/k
    r_t = torch.mean(torch.sum(tt, dim=1))/torch.mean(torch.sum(target_masked, dim=1))

    # print("r is:", r)
    if r != r:
        r = 0
    else:
        acc_r += r.item()
        count += 1
    if r_t != r_t:
        r_t = 0
    else:
        acc_rt += r_t.item()
        count_rt += 1
    return acc_r / count if count > 0 else 0 , acc_rt / count_rt if count_rt > 0 else 0






