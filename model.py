import torch
import torch.nn as nn



class skipGram(nn.Module):
    def __init__(self, vocabSize, embedDim):
        super(skipGram,self).__init__()
        self.numWords = vocabSize
        # self.hidDim fixme should I choose how many hidden dimensions?
        self.embeddingDim = embedDim  # fixme that is something u can choose
        self.embed = nn.Embedding(self.numWords, self.embeddingDim)
        self.bound = .5 / self.embeddingDim
        self.toOut = nn.Linear(self.embeddingDim, self.numWords)
        # self.toHidden = (-2 * self.bound) * torch.rand(self.numWords, self.embeddingDim,
        #                                                requires_grad=True) + self.bound
        #
        # # FIXME do I need to make that float? isn't it already?
        # self.toOut = (-2 * self.bound) * torch.rand(self.embeddingDim, self.numWords,
        #                                             requires_grad=True) + self.bound

    def inputOneHot(self, curr):
        # I want a matrix that has a zero for every word my current word isn't, and 1 for curr word
        oneHot = torch.zeros(self.numWords).long()
        oneHot[curr] = 1.0
        return oneHot
        # FIXME how do I ensure curr is the index of a word?do I even need this?

    def forward(self, curr):
        # know curr is a one hot
        curr = self.embed(curr)
        out = self.toOut(curr)
        # hid = curr.mm(self.toHidden)
        # out = hid.mm(self.toOut)
        return out

#
#
#     def forward(self, command):
#         embeds = self.embeddings(command)  # command should already be a list of embedded words
#         lstmOut, _ = self.lstm(embeds.view(len(command[0]), 1, -1))
#
#         # calculating scores for different possible labels
#         actSpace = self.toAct(lstmOut.view(len(command[0]), -1))
#         targetSpace = self.toTarget(lstmOut.view(len(command[0]), -1))
#         actScores = F.log_softmax(actSpace, dim=1)
#         targetScores = F.log_softmax(targetSpace, dim=1)

       #return actScores[len(actScores) - 1], targetScores[len(targetScores) - 1]
