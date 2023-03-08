import torch
# torch.set_default_tensor_type(torch.cuda.FloatTensor)

def recall_k(output, target, mask,k=3, window=3):
    # print("target in start is:",target)
    # print("mask in start is:",mask)
    # print("output in start is:",output)
    # TODO: make in config with metric
    k = 3

    acc_r = 0.0
    count = 0
    for i in range(len(mask)):
        # check if current index is masked (end of sentence)
        if mask[i] == 0:
            continue

        # create maski with zeros at the current index and zeros for masked indices
        maski = torch.zeros(len(mask))
        word_ahead = False
        word_before = False
        for j in range(1, window + 1):
            if i + j < len(mask):
                if word_ahead != True:
                    if mask[i + j] == 0:
                        word_ahead = True
                    else:
                        maski[i + j] = 1

            if i - j >= 0:
                if word_before != True:
                    if mask[i - j] == 0:
                        word_before = True
                    else:
                        maski[i - j] = 1

        maski[i] = 0

        # calculate the loss for the current index
        target_i = target * maski.view(len(target), 1)
        output_masked = output * maski.view(len(target), 1)

        forward_output = output_masked[i + 1:]

        forward_output = forward_output[torch.any(forward_output != 0, dim=1)]

        if (forward_output.numel() > 0):
            target_i_forward = target_i[i + 1:]

            target_i_forward = target_i_forward[torch.any(target_i_forward != 0, dim=1)]
            _, tk = torch.topk(forward_output, k)
            # print("tk is:",tk)
            tt = torch.gather(target_i_forward, 1, tk)
            r = torch.mean(torch.sum(tt, dim=1) / torch.sum(target_i_forward, dim=1))
            # print("target is:",target_i_forward)
            # print("tt is:",tt)

            if r != r:
                r = 0
            else:
                acc_r += r.item()
                count += 1
    return acc_r / count if count > 0 else 0
#     bsz = output.shape[0]
#     idx = torch.arange(0, bsz, device=output.device)
#
#     mask = mask.squeeze()
#     print("mask is:",mask)
#     for i in range(window):
#         mi = mask[i + 1:] * mask[i + 1:i + 1 + min(window, len(mask) - i - 1)]
#         print("mi first is:",mi)
#         mi = torch.nn.functional.pad(mi, (1 + i, 1 + i))
#         tm = mi[:-i - 1]
#         im = mi[i + 1:]
#         print("mi is:",mi)
#
#         target_mask = torch.masked_select(idx, tm)
#         input_mask = torch.masked_select(idx, im)
#         #ii = ii.long()
#         output = output[input_mask, :]
#         output = output.float()
#         print("tagert is:",target)
#         print("target mask is:",target_mask)
#         target = target[target_mask, :]
#         target = target.float()
#         # print("target of metric is:",target)
#         # print("output of metric is:",output)
#
#         _, tk = torch.topk(output,7)
#         # print(" taget is:",target)
#         # print("top k for batch:",tk)
#         tt = torch.gather(target, 1, tk)
#         r = torch.mean(torch.sum(tt, dim=1) / torch.sum(target, dim=1))
#         if r != r:
#             r = 0
#     return r





