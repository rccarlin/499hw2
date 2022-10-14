 discuss your implementation choices and 
 document the performance of your model (both training 
and validation performance, both in vitro and in vivo) under the conditions you 
settled on (e.g., what hyperparameters you chose) and discuss why these are a good set.
 
. Finally, in this report I'd also like you to do a 
little bit of analysis of the released code and discuss what's going on. In particular, what are the in vitro and in 
vivo tasks being evaluated? On what metrics are they measured? If there are assumptions or simplifications that can 
affect the metrics, what are they and what might go "wrong" with them that over- or under-estimate model performance?
## Implementation choices:
### (hyperparams and weird quirks with my code)
1. Skip-gram model: I didn't realize it would be way more work than the CBOW and by the time I noticed, I was too 
   invested
2. Widow size 2: I wanted to look beyond the immediately surrounding words to hopefully get better syntactic context of
   the target word (looking more than one ahead allows for multi-word phrases to be included in the context window).
   However, I didn't want the window to be too big because I knew my computer wouldn't be able to handle it.
3. 80/20 train/val split: with such a big data set, I felt 20% was plenty for train (and takes up less space than 30, 
   which I had originally considered doing).
4. I had to add padding to my output labels because python complained when I *only* included words/<start>/<end>. I
   doubt this changed too much down the line since a) there was max 2 "0" per output list and b) "0" just acts as 
   another sentence endpoint marker. If I was working with lines instead of sentences, I think this would be a bigger
   issue as the padding could be randomly in the center of sentences.
5. Criterion = Binary Cross Entropy with logits: I wanted cross entropy to avoid explicitly calling 
   a max function for loss, the model spits out logits so I assumed I should use something to handle that, and the "binary" part
   was added after class on Wednesday.
6. Optimizer = stochastic gradient decent: was what we've used in the past and also was recommended by the internet for 
   skipgram.
7. Learning rate = .05: I know lrs are typically between .1 and .01 and I wanted to start with a medium value. I wasn't 
   sure if I should have changed it based on the results of the model, so I decided to keep it
8. Changed accuracy function: In class, we discussed how the starter code's accuracy function wouldn't be accurate (lol)
   with skip gram, so I had to calculate it by hand using # in common between predicted & target / # in target. I wish
   I had been able to have different sized targets so I could penalize the model for predicting too many/ few, but alas.
   One issue I encountered was that this accuracy function requires a batch * 2widow representation of the model output,
   and converting into this took ~5 hr per 20k input. I was told that the training accuracy was not as important as the
   validation accuracy so I decided to only calculate accuracy for validation (sorry). 
9. Various vocab size: My computer ran out of memory when vocab_size = 3000 so I tested with size 300 and 1000.
   Obviously 1000 is the better of the two since it is closer to the 3000 I was supposed to be using.
10.Embedding Dimension = 3: oh noooo I put in 3 as a placeholder to test the function and then I left it. Well, at least
   that saved some precious space in memory...

## Accuracy results
Unfortunately I don't have any in vivo results as the in vitro stuff keeps taking 7 hours and then crashes on the starter 
code and I didn't finish my code far enough ahead of time to run and debug for a week.

That being said, I did manage to record some loss and accuracy scores (training accuracy was not computed because
converting the model's output into a batch x 2*window array took an obscene amount of time):

Vocab size 300, validation after 5 epochs:
val loss : 0.06810295467168881 | val acc: 0.3154260385376218
(I forgot to grab the train numbers for this one :'( )

Vocab size 1000, validation after 10 epochs:
train loss: 0.3016729356002149
train loss: 0.08663147930091201
(not 100% sure why there are only two when I said 10 trainings before validation but there we go)
val loss : 0.06581418702267544 | val acc: 0.26363829066756966