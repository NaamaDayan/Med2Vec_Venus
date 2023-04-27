import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_tensor_type(torch.cuda.FloatTensor)

def med2vec_loss(inputs, mask, probits, bce_loss, emb_w, ivec, jvec, window, eps=1.0e-8):
    """ returns the med2vec loss
    """

    def create_mask(original_mask,batch_size, window_size):
        pos_matrix = torch.arange(batch_size).reshape((batch_size, 1))
        mask_per_word = torch.abs(pos_matrix - pos_matrix.T) <= window_size
        mask_per_word = (mask_per_word * original_mask.unsqueeze(1) * original_mask.unsqueeze(0))
        mask_per_word.fill_diagonal_(0)
        return mask_per_word

    # def create_mask(original_mask, batch_size, window_size):
    #     pos_matrix = torch.arange(batch_size).reshape((batch_size, 1))
    #     diff = torch.abs(pos_matrix - pos_matrix.T)
    #     mask_per_word = (diff <= window_size).clamp(max=1)
    #     mask_per_word = (mask_per_word * original_mask.unsqueeze(1) * original_mask.unsqueeze(0))
    #     mask_per_word.fill_diagonal_(0)
    #     return mask_per_word

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
        modified_mask = torch.logical_and(masked_left_tensor,masked_right_tensor).fill_diagonal_(0).int()

        return modified_mask




    def visit_loss(x, mask, probits, window):
        '''
        TODO: write a function that gets the full sentence x, mask of sentence ,
        and the probits after forward for the sentence and calculates the bce_Loss for each word in sentence with the
        relevant probits around it.it removes all the "words" that are not relevant to the target word
        - this are words that are of new sentence or word beyond his window
        :param x: one hot vector for each word in batch
        :param mask: boolian vector meant to say which word is a symbol for end of word and shouldn ot be used
        :param probits: this is the probebilty for each word in the batch after forward - shold be (batch_size,voacb_size)
        :param window: all the word that shoud be part of the target word loss are the wrods inside the window[i-window,i+window]
        :return: it should return the bce loss of each target word with all its context words (with no meanning to distance for now)
        '''
        # print("mask and window is:",mask,window)
        mask_per_word = create_mask(mask,len(x),window)
        # print("mask per word is:",mask_per_word)
        mask_after_diagonal = split_and_modify_mask(mask_per_word).float()
        # print("mask after diagonal:",mask_after_diagonal)
        # print("shape of mask and x",mask_after_diagonal.shape,x.shape)
        x_masked = torch.matmul(mask_after_diagonal, x.float()).clamp(0,1)
        protbits_masked = torch.matmul(mask_after_diagonal, probits.float())
        loss = bce_loss(protbits_masked, x_masked.float())
        # print(protbits_masked.shape)
        # print("positive values in probits:",torch.sum(protbits_masked > 0))
        # print("original x is:",x)
        # print("x masked is:",x_masked)
        if(loss.item()<0):
            print("visit loss is negative!")



        # loss = 0
        # for i in range(len(x)):
        #
        #     current_loss = 0
        #     # check if current index is masked (end of sentence)
        #     if mask[i] == 0:
        #         continue
        #
        #     if loss != loss:
        #         import pdb;
        #         pdb.set_trace()
        #
        #     # create maski with zeros at the current index and zeros for masked indices
        #     maski = torch.zeros(len(mask))
        #     # print("maski at creation is:",maski)
        #     word_ahead = False
        #     word_before = False
        #     for j in range(1,window+1):
        #         if i + j < len(mask):
        #             if  word_ahead != True:
        #                 if mask[i + j] == 0:
        #                     word_ahead = True
        #                 else:
        #                     maski[i + j] = 1
        #
        #         if i - j >=0:
        #             if word_before != True:
        #                 if mask[i - j] == 0:
        #                     word_before = True
        #                 else:
        #                     maski[i - j] = 1
        #
        #     maski[i] = 0
        #     # print("maski of i",i, "is:",maski)
        #     # print("maski shape is:",maski.shape)
        #     # print("probits shape is:",probits.shape)
        #     # calculate the loss for the current index
        #     # print(maski[i + 1:].shape)
        #     x_i = x*maski.view(len(x), 1)
        #     # print("probits masked same as x masking:",probits*maski.view(10, 1))
        #     probits_masked = probits*maski.view(len(x), 1)
        #     # print("probits_masked:",probits_masked)
        #     backward_preds = probits_masked[:i]
        #     # print("backward_preds and x",backward_preds,x_i[:i])
        #     # print("clean backward is:",backward_preds[(backward_preds != 0).any(dim=1)])
        #     backward_preds = (backward_preds[(backward_preds != 0).any(dim=1)])
        #
        #
        #     forward_preds = probits_masked[i+1:]
        #     # print("clean forward is:",forward_preds[(forward_preds != 0).any(dim=1)])
        #     forward_preds = (forward_preds[(forward_preds != 0).any(dim=1)])
        #
        #
        #     backward_loss = 0
        #     forward_loss = 0
        #
        #     if (backward_preds.numel() > 0):
        #         x_i_backward = x_i[:i]
        #         x_i_backward = x_i_backward[torch.any(x_i_backward != 0, dim=1)]
        #         backward_loss = bce_loss(backward_preds, x_i_backward.float())
        #         # print("were here at backward at i = ",i)
        #
        #
        #     if(forward_preds.numel()>0):
        #         x_i_forward = x_i[i+1:]
        #         # print(x_i_forward)
        #         x_i_forward = x_i_forward[torch.any(x_i_forward != 0, dim=1)]
        #         # print(forward_preds)
        #         # print(x_i_forward)
        #
        #         forward_loss = bce_loss(forward_preds, x_i_forward.float())
        #         # print("were here at forward at i = ",i)
        #
        #     current_loss = torch.tensor(forward_loss + backward_loss, requires_grad=True, dtype=torch.float)
        #
        #     loss += current_loss
        return loss


    def code_loss(emb_w, ivec, jvec, eps=1.e-6):
        ivec.requires_grad_()
        jvec.requires_grad_()
        norm = torch.sum(torch.exp(torch.mm(emb_w.t(), emb_w)), dim=1)
        if torch.isnan(norm).any() or torch.isinf(norm).any():
            print("we got nan in norm")
            return 0

        norm_ivec = norm[ivec.long()]
        if torch.isnan(norm_ivec).any() or torch.isinf(norm_ivec).any():
            print("we got nan in ivec norm")
            return 0

        cost = -torch.log((torch.exp(torch.sum(emb_w[:, ivec.long()].t() * emb_w[:, jvec.long()].t(), dim=1)) /norm_ivec) + eps)
        cost_mean = torch.mean(cost)
        if (cost_mean.item() != cost_mean.item()):
            cost_mean = torch.zeros(1)
        return cost_mean



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Move all tensors to the device
    mask = mask.to(device)
    probits = probits.to(device)
    emb_w = emb_w.to(device)
    ivec = ivec.to(device)
    jvec = jvec.to(device)
    # print("input is:",inputs ,"ivec is:",ivec,"jvec is:",jvec)

    # vl = visit_loss(inputs, mask, probits, window = 2)
    cl = code_loss(emb_w, ivec.float(), jvec.float(), eps=1.e-6)
    return { 'code_loss': cl}

    # return { 'code_loss': cl,'visit_loss':vl}
