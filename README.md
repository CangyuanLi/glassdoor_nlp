# Step 1: Clean the Text

This is handled by ```get_tokens(), get_lemmas(), and clean_lemmas()```.

* ```get_tokens()```:
    * Removes punctuation
    * Removes numbers
    * Splits string into list of words by space
    * Lowercases all words
    * Removes stopwords

* ```get_lemmas()```:
    * Obtain roots of each token

* ```clean_lemmas()```:
    * Retain only relevant words

# Step 2: Pair Adjectives with Nouns

This is handled by ```pair_adj_noun()```. In many cases, adjectives can be misleading. For example,
if a review says "It was a great day until I got fired", the word cloud may incorrectly count
the positive adjective "great". Or a review may say "There was a great amount of strife", and so on.
Other adjectives are simply uninformative. Two reviews could both contain "many" but say the exact
opposite thing. For example, "There are many good things" vs. "There are many bad things". Therefore,
whenever possible, pair these unclear adjectives with a following noun.

* ```pair_adj_noun()```
    * When encounter an unclear adjective, look forward one word, and if that word is a noun, concatenate the two with an underscore