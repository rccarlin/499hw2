import data_utils
import pandas as pd

def main():
    sentences = data_utils.process_book_dir("books")
    # sentences is a list of lists, where first element is sentence and second element is book id

    # build one hot maps for input and output
    (
        vocab_to_index,
        index_to_vocab,
        suggested_padding_len,
    ) = data_utils.build_tokenizer_table(sentences, 3000)
    # create encoded input and output numpy matrices for the entire dataset and then put them into tensors
    encoded_sentences, lens = data_utils.encode_data(
        sentences,
        vocab_to_index,
        suggested_padding_len,
    )

    # calculate num words ber book... oh shoot is that char....?
    numWords = 0
    book2words = dict()  # maps a book id to number of words in it
    words2books = dict()  # maps word to a dictionary of book to num occurrences
    words2occ = dict()  # maps words to number of times we see it total
    for s in range(len(encoded_sentences)):  # s is index of current sentences
        for w in encoded_sentences[s]:  # for each word in sentence
            if w > 2:  # this is a real word, not a buffer
                numWords += 1
                currBook = sentences[s][1]

                # Populating book2words
                if currBook in book2words:  # we have seen this book before
                    book2words[currBook] += 1
                else:  # new book
                    book2words[currBook] = 1

                currWord = index_to_vocab[w]
                # populating words2books
                if currWord in words2books:  # have seen this word before
                    if currBook in words2books[currWord]:  # have seen this word in this book before
                        words2books[currWord][currBook] += 1
                    else:  # we have seen this word but not in this book
                        words2books[currWord][currBook] = 1
                else:  # new word
                    temp = dict()
                    temp[currBook] = 1
                    words2books[currWord] = temp

                # populating words2occ
                if currWord in words2occ:  # have seen this word before
                    words2occ[currWord] += 1
                else:  # new word
                    words2occ[currWord] = 1


    for key in book2words:
        print(key, ":", book2words[key] / numWords)

    # now to get pairs of words and tie them to their respective book
    # listFirst= list()
    # listSecond = list()
    # listBook = list()
    # for s in range(len(encoded_sentences)):
    #     for curr in range(len(encoded_sentences[s])):
    #         for i in range(-2, 0):  # do i actually need to have (i have, and have i)
    #             if curr + i in range(len(encoded_sentences[s])):
    #                 sortedl = [encoded_sentences[s][curr], encoded_sentences[s][curr + i]]
    #                 sortedl.sort()
    #                 listFirst.append(sortedl[0])
    #                 listSecond.append(sortedl[1])
    #                 listBook.append(sentences[s][1])
    # tempDict = {"Word_one": listFirst, "Word_two": listSecond, "BookID": listBook}
    # df1 = pd.DataFrame(tempDict)
    # df1.to_csv("wordPairs.csv")

    # pairs2Books = dict()  # key = tuple, value = dictionary of book to occurrences
    # for s in range(len(encoded_sentences)):
    #     for curr in range(len(encoded_sentences[s])):
    #         for i in range(-2, 3):
    #             if (curr + i in range(len(encoded_sentences[s]))) and i != 0:
    #                 temp = (encoded_sentences[s][curr], encoded_sentences[s][curr + i])
    #                 if temp in pairs2Books:  # we've seen this pair before
    #                     if sentences[s][1] in pairs2Books[temp]:  # weve seen this pair in this book
    #                         pairs2Books[temp][str(sentences[s][1])] += 1
    #                     else:  # we've seen this pair but not in this book
    #                         pairs2Books[temp][str(sentences[s][1])] = 1
    #                 else:  # we have never seen this pair before
    #                     newDict = dict()
    #                     newDict[str(sentences[s][1])] = 1
    #                     pairs2Books[temp] = newDict

    # f = open("")
    # okay so I know the encoded sentences line up with the non encoded sentences

main()