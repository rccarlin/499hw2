import argparse
import os

import numpy
import numpy as np
import tqdm
import torch
from sklearn.metrics import accuracy_score

from eval_utils import downstream_validation
import utils
import data_utils

from torch.utils.data import TensorDataset

import model as md
import random


def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """

    # read in training data from books dataset
    sentences = data_utils.process_book_dir(args.data_dir)

    # build one hot maps for input and output
    (
        vocab_to_index,
        index_to_vocab,
        suggested_padding_len,
    ) = data_utils.build_tokenizer_table(sentences, vocab_size=args.vocab_size)

    # create encoded input and output numpy matrices for the entire dataset and then put them into tensors
    encoded_sentences, lens = data_utils.encode_data(
        sentences,
        vocab_to_index,
        suggested_padding_len,
    )

    # ================== TODO: CODE HERE ================== #
    # Task: Given the tokenized and encoded text, you need to
    # create inputs to the LM model you want to train.
    # E.g., could be target word in -> context out or
    # context in -> target word out.
    # You can build up that input/output table across all
    # encoded sentences in the dataset!
    # Then, split the data into train set and validation set
    # (you can use utils functions) and create respective
    # dataloaders.
    # ===================================================== #

    # I assume encoded_sentences are the sentences... encoded
    # bigTable = list()
    # lists for train and validation
    train_encoded = list()
    train_out = list()
    val_encoded = list()
    val_out = list()

    for s in encoded_sentences:  # for each sample sentence
        for curr in range(len(s)):  # for each word in the sentence
            if s[curr] == 0:  # no useful words after this
                break
            temp = list()
            for i in range(-1 * args.num_out, args.num_out + 1):
                if (curr + i in range(len(s))) and i != 0:  # in bounds and not the current word
                    if s[curr] != 0 and s[curr + i] != 0:  # neither are padding
                        temp.append(s[curr + i])
            while len(temp) < 2 * args.num_out:  # fixme added padding because it complained about ragged
                temp.append(0)
            # temp now is a list of context words
            split = random.random()
            if split <= .2:  # validation  fixme something you can change
                val_encoded.append(s[curr])
                val_out.append(np.array(temp))
            else:  # training
                train_encoded.append(s[curr])
                train_out.append(np.array(temp))

    train_encoded = np.array(train_encoded)
    train_out = np.array(train_out)
    val_encoded = np.array(val_encoded)
    val_out = np.array(val_out)

    # once out here, bigTable has been made for everything.... time to break into training and validation

    trainDS = torch.utils.data.TensorDataset(torch.from_numpy(train_encoded), torch.from_numpy(train_out))
    valDS = torch.utils.data.TensorDataset(torch.from_numpy(val_encoded), torch.from_numpy(val_out))
    train_loader = torch.utils.data.DataLoader(trainDS, shuffle=True, batch_size=args.batch_size)  # FIXME
    val_loader = torch.utils.data.DataLoader(valDS, shuffle=True, batch_size=args.batch_size)
    return train_loader, val_loader, index_to_vocab



def setup_model(args):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your CBOW or Skip-Gram model.
    # ===================================================== #
    # is there an embedding dim?
    model = md.skipGram(args.vocab_size, 3)  # fixme can change embedding dim
    return model


def setup_optimizer(args, model):
    """
    return:
        - criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for predictions. 
    # Also initialize your optimizer.
    # ===================================================== #
    criterion = torch.nn.BCEWithLogitsLoss()  # FIXME
    optimizer = torch.optim.SGD(model.parameters(), lr=.05)  # FIXME what should the lr be?
    return criterion, optimizer


def train_epoch(
    args,
    model,
    loader,
    optimizer,
    criterion,
    device,
    training=True,
):
    model.train()
    epoch_loss = 0.0

    # keep track of the model predictions for computing accuracy
    pred_labels = []
    target_labels = []

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for (inputs, labels) in tqdm.tqdm(loader):
        # put model inputs to device
        inputs, labels = inputs.to(device).long(), labels.to(device).long()

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        pred_logits = model(inputs)  # pred_logits is batch by vocab... oh I knew that lmao

        # loss function is expecting batch * vocab
        # labels is not that, so you need to loop over labels and make it something loss will like
        lossLabels = torch.zeros(len(pred_logits), args.vocab_size)
        # populating lossLabels
        for l in range(len(labels)):  # okay so these are the max len 4 lists
            for n in labels[l]:
                lossLabels[l][n] = 1
        # calculate prediction loss
        # fixme, using my own array so it's the right size
        loss = criterion(pred_logits.squeeze(), lossLabels)
        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # logging
        epoch_loss += loss.item()

        # compute metrics
        pred_labels.extend(pred_logits.cpu())  # so this is attaching what the model spits out, will do math
        # on it later
        target_labels.extend(labels.cpu())  # originally target_labels.extend(labels.cpu().numpy())

    # acc = accuracy_score(pred_labels, target_labels)  # this prob won't be accurate but I don't super care
    epoch_loss /= len(loader)

    return epoch_loss, pred_labels, target_labels


def validate(args, model, loader, optimizer, criterion, device):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():
        val_loss, pred_labels, target_labels = train_epoch(
            args,
            model,
            loader,
            optimizer,
            criterion,
            device,
            training=False,
        )
        # turning model output into actual word predictions
        exAcc = 0
        index = np.arange(args.vocab_size)  # so I don't loose track of what word these numbers correlate to
        for p in range(len(pred_labels)):  # for each example, cos it extended instead of appended
            together = sorted(zip(pred_labels[p], index), reverse=True)[:(2 * args.num_out)]
            temp = list()
            for t in together:
                temp.append(t[1])
            inCommon = 0
            for t in temp:
                if t in target_labels[p]:
                    inCommon += 1
            exAcc += (inCommon / len(target_labels[p]))
        # fixme issue: if I pick top window, that limits what I predict, and like what are the chances that
        # target is just 4? or is that not an issue?
        val_acc = exAcc / len(pred_labels)

    return val_loss, val_acc


def main(args):
    device = utils.get_device(args.force_cpu)

    # load analogies for downstream eval
    external_val_analogies = utils.read_analogies(args.analogies_fn)

    if args.downstream_eval:
        word_vec_file = os.path.join(args.output_dir, args.word_vector_fn)
        assert os.path.exists(word_vec_file), "need to train the word vecs first!"
        downstream_validation(word_vec_file, external_val_analogies)
        return

    # get dataloaders
    train_loader, val_loader, i2v = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(args)
    print(model)

    # get optimizer
    criterion, optimizer = setup_optimizer(args, model)

    for epoch in range(args.num_epochs):
        print("training 1 epoch")
        # train model for a single epoch
        print(f"Epoch {epoch}")
        train_loss, p, t = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
        )
        if epoch % args.save_every == 0:
            ckpt_file = os.path.join(args.output_dir, "model.ckpt")
            print("saving model to ", ckpt_file)
            torch.save(model, ckpt_file)

        #fixme, not returning that
        print(f"train loss : {train_loss}")
        # print(f"train loss : {train_loss} | train acc: {train_acc}")

        if epoch % args.val_every == 0:
            val_loss, val_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                criterion,
                device,
            )
            print(f"val loss : {val_loss} | val acc: {val_acc}")

            # ======================= NOTE ======================== #
            # Saving the word vectors to disk and running the eval
            # can be costly when you do it multiple times. You could
            # change this to run only when your training has concluded.
            # However, incremental saving means if something crashes
            # later or you get bored and kill the process you'll still
            # have a word vector file and some results.
            # ===================================================== #

            # save word vectors
            word_vec_file = os.path.join(args.output_dir, args.word_vector_fn)
            print("saving word vec to ", word_vec_file)
            utils.save_word2vec_format(word_vec_file, model, i2v)

            # evaluate learned embeddings on a downstream task
            downstream_validation(word_vec_file, external_val_analogies)

        if epoch % args.save_every == 0:
            ckpt_file = os.path.join(args.output_dir, "model.ckpt")
            print("saving model to ", ckpt_file)
            torch.save(model, ckpt_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outs", help="where to save training outputs")
    parser.add_argument("--data_dir", type=str, default="books", help="where the book dataset is stored")
    parser.add_argument(
        "--downstream_eval",
        action="store_true",
        help="run downstream eval on trained word vecs",
    )
    # ======================= NOTE ======================== #
    # If you adjust the vocab_size down below 3000, there 
    # may be analogies in the downstream evaluation that have
    # words that are not in your vocabulary, resulting in
    # automatic (currently) zero score for an ABCD where one
    # of A, B, C, or D is not in the vocab. A visible warning
    # will be generated by the evaluation loop for these examples.
    # ===================================================== #
    parser.add_argument(
        "--vocab_size", type=int, default=300, help="size of vocabulary"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument(
        "--analogies_fn", type=str, default= "analogies_v3000_1309.json",help="filepath to the analogies json file"
    )
    parser.add_argument(
        "--word_vector_fn", type=str, help="filepath to store the learned word vectors",
        default='learned_word_vectors.txt'
    )
    parser.add_argument(
        "--num_epochs", default=9, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "--val_every",
        default=100,
        type=int,
        help="number of epochs between every eval loop",
    )
    parser.add_argument(
        "--save_every",
        default=1,
        type=int,
        help="number of epochs between saving model checkpoint",
    )
    parser.add_argument(
        "--num_out",
        default=2,
        type=int,
        help="half size of context",
    )
    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #

    args = parser.parse_args()
    main(args)
