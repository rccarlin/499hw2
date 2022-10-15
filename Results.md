# this is just a duplication of readme 
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
10. Embedding Dimension = 3: oh noooo I put in 3 as a placeholder to test the function and then I left it. Well, at least
   that saved some precious space in memory...

## Accuracy results
Unfortunately I don't have any in vivo results because I can't seem to get it to run and I feel it's too late to ask.
</br>That being said, I did manage to record a few loss and accuracy scores:

Vocab size 300, validation after 5 epochs:</br>
val loss : 0.06810295467168881 | val acc: 0.3154260385376218</br>
(I forgot to grab the train numbers for this one :'( )
</br>

Vocab size 1000, validation after 10 epochs:</br>
train loss: 0.3016729356002149</br>
train loss: 0.08663147930091201</br>
(not 100% sure why there are only two when I said 10 trainings before validation but there we go)</br>
val loss : 0.06581418702267544 | val acc: 0.26363829066756966
</br>

Sadly I don't think my program managed to spit out enough numbers for me to say if my hyperparameters are good or not;
but considering I had to greatly reduce vocab size (and I made embedding dimension really small), I'm going to conclude 
that my model is not good.


##Evaluation
If there are assumptions or simplifications that can affect the metrics, what are they and what might go "wrong" with 
them that over- or under-estimate model performance?

In vitro: Given a word, can the model predict what words are likely to be seen around given word? Performance is judged 
on how close the predicted context is to the target context. "Closeness" can be determined by how confident the model is
in predicting labels that are actually correct/ not predicting ones that are wrong. This may cause a problem if we assume
words only have one type of context (for my small window, that's like assuming record the action and record the object 
are the same). When we provide only the current word, how should the model know which sense you mean? The model would 
be unfairly penalized for not being confident enough in the "correct" context and being too confident in the other 
(technically-right-but-still-wrong) context. (underestimate) </br>
One simplification my model in particular makes is that all target labels are the same length and I force the model to 
only predict that many labels, regardless of if the model implies more/fewer should have been picked. I can see this 
affecting my calculation of accuracy, which cares about the length of target labels. I believe this issue can overestimate
accuracy if the model wasn't super confident in the contexts we said it predicted (i.e., the model only had 3 strong
contexts but we forced it to predict more).


In vivo: Can we perform word vector math on the vectors learned during the in vitro task to create analogies that
make sense? Given 4 words a, b, c, and d, we calculate C + B - A and look for the words close to the end of that vector.
If d is nearby, we count that as "correct", and if d is the closest word, we count that as "exact." Overall performance
is judged on # correct / # tested and # exact / # tested, for each type (ex semantic) and subtype of analogy (ex 
semantic, part of). There is again a concern about words with multiple meanings; the analogies [typically] only make
sense for one sense per word. If the vectors we use are forced to represent multiple senses, our vector math probably 
won't result in the desired answer. This method of rating performance also "penalizes" the model when a synonym of d is
the closest word to C + B - A. Technically, if a synonym is the top result, d is probably nearby so this would be a
point towards correct (but not exact). This underestimates performance because answers that should be equivalent to the
target are not scored the same as the target. Think about how unfair this would be to ask a human: "Closets are to 
clothes as the US is to...?" and California was partial credit but the Correct answer was Wisconsin.

