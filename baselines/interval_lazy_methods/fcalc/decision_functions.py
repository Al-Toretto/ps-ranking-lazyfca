import numpy as np

def alpha_weak(supp_cont,classes, class_lengths, alpha=0., randomize=False):
    ccl = class_lengths.sum() - class_lengths
    criter = np.zeros(shape=(len(class_lengths),supp_cont[0].shape[1]))
    preds = np.full(supp_cont[0].shape[1], -1.)
    for j in range(len(classes)):
        criter[j] = (supp_cont[j][1] <= alpha).sum(axis=-1)
    if randomize:
        criter = criter.T / supp_cont[0].shape[-1]
    else:
        criter = criter.T / class_lengths
    pred_mask = (np.max(criter,axis=-1)[:,None]==criter).sum(axis=-1) < 2
    preds[pred_mask] = classes[np.argmax(criter[pred_mask], axis=-1)]
    return preds

def alpha_weak_support(supp_cont, classes, class_lengths, alpha=0., randomize=False, k=None):
    ccl = class_lengths.sum() - class_lengths
    # indices = np.arange()
    criter = np.zeros(shape=(len(class_lengths),supp_cont[0].shape[1]))
    preds = np.full(supp_cont[0].shape[1], -1.)
    if k is None:
        for j in range(len(classes)):
            criter[j] = (supp_cont[j][0]*(supp_cont[j][1] <= alpha)).sum(axis=-1)
        if randomize:
            criter = criter.T / (supp_cont[0].shape[-1]*class_lengths)
        else:
            criter = criter.T / class_lengths**2
    else:
        for j in range(len(classes)):
            criter[j] = np.sort(supp_cont[j][0]*(supp_cont[j][1] <= alpha), axis=-1)[:,:k].sum(axis=-1)
        if randomize:
            criter = criter.T / (supp_cont[0].shape[-1]*class_lengths)
        else:
            criter = criter.T / class_lengths**2
    pred_mask = (np.max(criter,axis=-1)[:,None]==criter).sum(axis=-1) < 2
    preds[pred_mask] = classes[np.argmax(criter[pred_mask], axis=-1)]
    return preds

def ratio_support(supp_cont, classes, class_lengths, alpha=1., randomize=False, k=None):
    ccl = class_lengths.sum() - class_lengths
    criter = np.zeros(shape=(len(class_lengths),supp_cont[0].shape[1]))
    preds = np.full(supp_cont[0].shape[1], -1.)
    if k is None:
        for j in range(len(classes)):
            sup = (supp_cont[j][0]*(supp_cont[j][1]/ccl[j] * alpha <= supp_cont[j][0]/class_lengths[j])).sum(axis=-1)
            cont = (supp_cont[j][1]*(supp_cont[j][1]/ccl[j] * alpha <= supp_cont[j][0]/class_lengths[j])).sum(axis=-1)+1e-6
            criter[j] = (ccl[j]*sup) / (cont*class_lengths[j])
    else:
        for j in range(len(classes)):
            ind = np.argsort(supp_cont[j][0]*(supp_cont[j][1]/ccl[j] * alpha <= supp_cont[j][0]/class_lengths[j]), axis=-1)[:,-k:]
            sup = (supp_cont[j][0][ind]).sum(axis=-1)
            cont = (supp_cont[j][1][ind]).sum(axis=-1)+1e-6
            criter[j] = (ccl[j]*sup) / (cont*class_lengths[j])
    criter = criter.T
    pred_mask = (np.max(criter,axis=-1)[:,None]==criter).sum(axis=-1) < 2
    preds[pred_mask] = classes[np.argmax(criter[pred_mask], axis=-1)]
    return preds
