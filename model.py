import torch
import torch.nn as nn



class skipGram(nn.Module):
    def __init__(self, numWords, embedDim):
        super(skipGram,self).__init__()
        self.vocab_size = numWords
        self.embedding_dim = embedDim
        self.embed = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.toOut = nn.Linear(self.embedding_dim, self.vocab_size)

    def forward(self, curr):
        curr = self.embed(curr)
        out = self.toOut(curr)
        # hid = curr.mm(self.toHidden)
        # out = hid.mm(self.toOut)
        return out

