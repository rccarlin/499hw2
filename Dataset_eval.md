sentence[0] returned ['the project gutenberg ebook of a modest proposal by jonathan swift this ebook is for the use...
 ... of anyone anywhere in the united states and most other parts of the world at no cost and with almost no...
  restrictions whatsoever', '1080']

Right off the bat, this isn't a sentence, and also isn't that useful if we actually care about the content
I wonder if the 30 times this happens will affect anything?

Big questions:
1. is there an uneven distribution of word pairs from different books, and do we think that changes anything?
2. do the most common pairs make sense?

Number of words (nondistinct) for each book:  WAIT are these chars?

Percentages
1080 : 0.00190
11 : 0.00867
1184 : 0.13619
1232 : 0.01545
1260 : 0.05538
1342 : 0.03663
1400 : 0.05512
1497 : 0.06435
16328 : 0.01202
1661 : 0.03157
174 : 0.02410
19097 : 0.01177
1952 : 0.00268
205 : 0.034889
25344 : 0.02551
2542 : 0.00037
2554 : 0.00039
2591 : 0.03061
2600 : 0.16628
2701 : 0.06336
345 : 0.04809
408 : 0.02103
43 : 0.00843
4300 : 0.00045
46 : 0.00927
5200 : 0.00035
6130 : 0.05632
64317 : 0.01502
84 : 0.02293
98 : 0.04083

if it was even, would expect each book to contribute ~.03
Possible areas of concern are books War and Peace (17%) and Count of Monte Cristo (14%)

Now to see the distribution of word pairs (wrt books?)
As I was testing something, I got another project gutenberg line so this might mess it way up...

I tried multiple times to analyze the whole data set but that's .... 175,181 sentences, with like, many pairs per....
and I'm inpatient

I think I will just pull a representative sample... how about 10%? but will that capture the strange things?

JK, let's look at just the words and their occurrences!
The whole output is in provided txt file, but I will point out highlights here:

- the is most common word, but only has 4 occurrences in 2542, 2554, and 2500, and 6 in 4300. Checked these and sure
enough, they aren't books. Since there are only 4 of these types of files, I'm not too worried about this "messing up"
my training unless they somehow manage to push words into the 3000 word vocab that don't belong there.

- Project is the second most common word... that isn't saying project wouldn't be in the vocab without Project Gutenberg,
just I suspect a lot of these instances are not actually the books, in which case that may mess up the learned vector/
neighbors for the word project (and probably gutenberg). And surprise surprise, gutenberg is next... certainly not a
helpful word in the real word, but neither is "cruncher" but that's in the vocab as well. How much do I actually think
these are messing the learning up tho? ebook...

- finally seeing a word that deserves to be common, "of"

- why is johnathan swift on here? oh those are actually separate! cos the johnathan swift book (1080) only has each 3
times. but also, it's not that common, are these not in order????







