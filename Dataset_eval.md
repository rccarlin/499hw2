# Exploring the Training Dataset

At first, I wanted to study to frequency and distribution (possibly via a co-occurrence matrix of sorts) of 
word-context pairs, but I realized this would take a long time and would probably be fairly sparse. So I turned to
individual words. The idea being the words (and books!) fed into the model affect what word vectors the model learns, 
and you know what they say, weird in, weird out (??)

## Big questions
1. Are certain books dominating the training data, and what are the consequences of this?
2. Do the top 3000 most common words behave like we'd expect (common, used everywhere frequently), and does it even
   matter anyway?

   
### Q1: Books
I started by counting the number of words in each book provided. Since different books can have different styles of
writing, the same word may have a different "usual" context across different books. Thus, a concerning 
over-representation of a book may skew the word vectors learned.

If each book had an equal number of words, we would expect each book to contribute ~3% of total words in the directory.
Here is a list of each book and their % contribution. "*" flag books that contribute at least double 3%


1080 :  0.00190 </br>
11 :    0.00867</br>
1184 :  0.13619 ****</br>
1232 :  0.01545</br>
1260 :  0.05538</br>
1342 :  0.03663</br>
1400 :  0.05512</br>
1497 :  0.06435 **</br>
16328 : 0.01202</br>
1661 :  0.03157</br>
174 :   0.02410</br>
19097 : 0.01177</br>
1952 :  0.00268</br>
205 :   0.03489</br>
25344 : 0.02551</br>
2542 :  0.00037</br>
2554 :  0.00039</br>
2591 :  0.03061</br>
2600 :  0.16628 ****</br>
2701 :  0.06336 **</br>
345 :   0.04809</br>
408 :   0.02103</br>
43 :    0.00843</br>
4300 :  0.00045</br>
46 :    0.00927</br>
5200 :  0.00035</br>
6130 :  0.05632</br>
64317 : 0.01502</br>
84 :    0.02293</br>
98 :    0.04083

Possible areas of concern are the books War and Peace (~17%) and the Count of Monte Cristo (~14%), as they are much longer and
thus contribute more words than the average book. That being said, these aren't offensively high percentages and many words
alone won't mess up word vectors. What matters is style! </br>

I am not going to read all these books to assess their style, but I can look at their original language and 
publication date (I will not argue how time and place affect art, so just trust me). War and Peace was witten in 
Russian in 1865 and the Count of Monte Cristo is a French novel from 1844. So both are old and are translations. But 
again, that alone isn't an issue.</br> 

After some math, I found that 
~48% of the words are from translated books and
~19% of the words are from non-19th century writing. Honestly, I'm not horribly sure what to make of this; I think maybe
the translation part's impact is too hard to track since all the texts are in English and the translations happened at 
various times. But! Since most of the texts are 19th century, I assume the word vectors for words that were common then
will have better vectors than newer or older words (if those even make it into the top 3000). 

Ultimately, I think the only way to ensure the books don't weirdly sway the model is to check the words and their sources.

### Q2: Words
I made dictionaries to keep track of the top 3000 words and how often they occur in each book. My goal was to identify
words that maybe shouldn't be in the top 3000 for whatever reason. Maybe these words took the spot of a word that will 
come up more often in vivo. Here are some of my favorite findings (in order from most common --> least common):

- "The" is the second most common word (unsurprising), but it only appears 4 times in 2542, 2554, and 2500, and only 6
  times in 4300. I checked these and found out they were not books, but code! While I'm not sure why these are part of
  the training set, I am not worried about them pushing weird words into the top 3000 since they are all very short. 
  This phenomenon happens with other common words like "of" and "to."
- Near the top 100 mark is the word "shall," which I feel is a lot more common in older books than now, supporting my
  hypothesis from earlier that common old words may get vectors because of the 19th century bias in the data. ("thou" is
  also here, but much later)
- "Came" and "go" are ranked one behind the other, so are "hand" and "eyes." I think this is fun but irrelevant.
- "Man" (6471 occurrences) and "men" (3130) are much more common than "woman" (1069) and "women" (635). This may have
  to do with the stories themselves or the time they were written, but this gender imbalance may cause issues when 
  performing gender analogies.
- "gutenbergtm" should NOT be here if we wanted the dataset to be just the books. Even if this was a word, what would 
  the point of training on it be? Is this going to ever be an in vivo question? Is the context of this word ever going 
  to change (it always happens 56 times)? I feel like this is a sign that some word vectors may be rigged if said word is a 
  part of the legal/ non-book stuff on each file. "Copyright" and "agreement" both appear 499 times, suggesting they 
  may only appear together and never apart (though I would have thought "agreement" could find other contexts). 
  "wwwgutenbergorg" happens 5 times a book.
- "Monte" is in roughly the top 300, but only appears in 3 books (174, 1184, 64317), so maybe The Count of Monte Cristo
  *was* too powerful for this data set
- War and Peace singlehandedly got at lest one word on the list ("historians")
- "td," "tr," and "div" only appear in the code "books" so I guess they were long enough files to push nonsense into
  the model.
- A word had to appear at least 89 times to be in the top 3000. Thankfully (and maybe surprisingly), I'd say most of
  those words are fairly common, so I am now less worried about the model than I had been before this exercise. 
- A lot of the words that appear in only a few book are names. Names probably won't be too helpful in vivo, but it would
  probably be too much work to remove them.


