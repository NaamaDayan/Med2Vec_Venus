import torch
import torch.nn as nn
import torch.nn.functional as F

def med2vec_loss(inputs, mask, probits, bce_loss, emb_w, ivec, jvec, window, eps=1.0e-8):
    """ returns the med2vec loss
    """

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


        # print("x with mask is",x*mask.view(10, 1))
        # print("probits are:",probits)
        loss = 0
        # print("full mask is:",mask)
        for i in range(len(x)):

            current_loss =0
            # check if current index is masked (end of sentence)
            if mask[i] == 0:
                continue

            if loss != loss:
                import pdb;
                pdb.set_trace()

            # create maski with zeros at the current index and zeros for masked indices
            maski = torch.zeros(len(mask))
            # print("maski at creation is:",maski)
            word_ahead = False
            word_before = False
            for j in range(1,window+1):
                if i + j < len(mask):
                    if  word_ahead != True:
                        if mask[i + j] == 0:
                            word_ahead = True
                        else:
                            maski[i + j] = 1

                if i - j >=0:
                    if word_before != True:
                        if mask[i - j] == 0:
                            word_before = True
                        else:
                            maski[i - j] = 1

            maski[i] = 0
            # print("maski of i",i, "is:",maski)
            # print("maski shape is:",maski.shape)
            # print("probits shape is:",probits.shape)
            # calculate the loss for the current index
            # print(maski[i + 1:].shape)
            x_i = x*maski.view(len(x), 1)
            # print("probits masked same as x masking:",probits*maski.view(10, 1))
            probits_masked = probits*maski.view(len(x), 1)
            # print("probits_masked:",probits_masked)
            backward_preds = probits_masked[:i]
            # print("backward_preds and x",backward_preds,x_i[:i])
            # print("clean backward is:",backward_preds[(backward_preds != 0).any(dim=1)])
            backward_preds = (backward_preds[(backward_preds != 0).any(dim=1)])


            forward_preds = probits_masked[i+1:]
            # print("clean forward is:",forward_preds[(forward_preds != 0).any(dim=1)])
            forward_preds = (forward_preds[(forward_preds != 0).any(dim=1)])


            backward_loss = 0
            forward_loss = 0

            if (backward_preds.numel() > 0):
                x_i_backward = x_i[:i]
                x_i_backward = x_i_backward[torch.any(x_i_backward != 0, dim=1)]
                backward_loss = bce_loss(backward_preds, x_i_backward.float())
                # print("were here at backward at i = ",i)


            if(forward_preds.numel()>0):
                x_i_forward = x_i[i+1:]
                # print(x_i_forward)
                x_i_forward = x_i_forward[torch.any(x_i_forward != 0, dim=1)]
                # print(forward_preds)
                # print(x_i_forward)

                forward_loss = bce_loss(forward_preds, x_i_forward.float())
                # print("were here at forward at i = ",i)

            current_loss = torch.tensor(forward_loss, requires_grad=True, dtype=torch.float) \
                           + torch.tensor(backward_loss, requires_grad=True, dtype=torch.float)

            loss += current_loss
        return loss

    def code_loss(emb_w, ivec, jvec, eps=1.e-6):
        # print(ivec.dtype)
        ivec.requires_grad_()
        jvec.requires_grad_()
        emb_w.requires_grad_()

        norm = torch.sum(torch.exp(torch.mm(emb_w.t(), emb_w)), dim=1)
        cost = -torch.log((torch.exp(torch.sum(emb_w[:, ivec.long()].t() * emb_w[:, jvec.long()].t(), dim=1)) / norm[ivec.long()]) + eps)
        # print("emb_w shape",emb_w.shape)
        # print("ivec size",len(ivec))
        # print("emb_w with ivec shape",emb_w[:, ivec.long()].shape)
        # print("emb_w with ivec", emb_w[:, ivec.long()])
        cost = torch.mean(cost)
        # print("cost is:",cost)
        return cost

    vl = visit_loss(inputs, mask, probits, window = window)
    cl = code_loss(emb_w, ivec, jvec, eps=1.e-6)
    return { 'code_loss': cl,'visit_loss':vl}
