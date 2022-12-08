# Imports

from pathlib import Path
import re
import time
import typing

import gensim
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from wordcloud import WordCloud

# Options

pd.options.mode.chained_assignment = None  # default="warn", removes settingwithcopywarning

# Paths

BASE_PATH = Path(__file__).resolve().parents[1]
DAT_PATH = BASE_PATH / "Data"
OUT_PATH = BASE_PATH / "Output"

# Globals

numeric = typing.Union[int, float]

nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

CUSTOM_STOP = [
    "company", "work", "working", "ive", "review", "lot", "employee", "u",
    "dont", "nothing", "way", "store", "thing", "le", "doesnt", "get",
    "youre", "con", "going", "new"
]
STOPWORDS = nltk.corpus.stopwords.words("english")
STOPWORDS = STOPWORDS + CUSTOM_STOP
TOKEEP = {"JJ", "NN", "VBG", "VBN", "RBR"}
VAGUE_ADJECTIVES = {
    "great", "good", "bad", "poor", "heavy", "low", "little", "nice", 
    "high", "many", "better", "best", "much"
}
FINAL_STOP = VAGUE_ADJECTIVES.union({"cant"})
SELECTED_COLS = ["review_cons"]
SELECTED_FILTERS = ["pre_8qtrs", "post_8qtrs"]

# Functions

def dict_to_csv(my_dict: dict, path: Path) -> None:
    """Utility funciton to write dictionary
    to csv.

    Args:
        path (Path): path to file
    """
    with open(path, "w") as f:
        for key in my_dict.keys():
            f.write("%s, %s\n" % (key, my_dict[key]))

def get_tokens(string: str, stopwords: list=STOPWORDS) -> list:
    """Keep only letters and tokenize

    Args:
        string (str): The review
        stopwords (list, optional): Words to remove. Defaults to STOPWORDS.

    Returns:
        list: A list of tokens (each word is a token)
    """
    newstr = re.sub(r"[^\w\s]", "", string)
    newstr = re.sub(r"[0-9]+", "", newstr)
    # tokenize 
    tokens = nltk.tokenize.word_tokenize(newstr)
    tokens = [word.lower() for word in tokens if not word.lower() in stopwords]

    return tokens

def get_lemmas(tokens: list, stopwords: list=STOPWORDS) -> list:
    """Obtain roots of each token
    For example, "works -> work"

    Args:
        tokens (list): List of tokens
        stopwords (list, optional): Words to remove. Defaults to STOPWORDS.

    Returns:
        list: A list of token roots
    """
    lemmas = [nltk.stem.WordNetLemmatizer().lemmatize(word) for word in tokens]
    # remove duplicate paragraphs from lemmas
    unique_lemmas = []
    for par in lemmas:
        if par not in unique_lemmas:
            if par not in stopwords:
                unique_lemmas.append(par)
    lemmas = unique_lemmas

    return lemmas

def clean_lemmas(lemmas: list, tokeep: set=TOKEEP) -> list:
    """Generate a tuple of (word, part of speech)

    Args:
        lemmas (list): A list of lemmas
        tokeep (set, optional): A list of parts of speech to keep. Defaults to TOKEEP.

    Returns:
        list: A list of tuples, each tuple being (word, pos)
    """
    tagged_lemmas = nltk.pos_tag(lemmas)
    clean = [(word, tag) for (word, tag) in tagged_lemmas if any(pas in tag for pas in tokeep)]

    return clean 

def pair_adj_noun(lemmas: list) -> list:
    """Pair uninformative adjectives with a noun.
    This only looks forward one word, and misses adjectives
    that are at the end of sentences. Turns ["great", "day"]
    into ["great_day"]. Removes second word and separtes by underscore
    to be consistent with Gensim.

    Args:
        lemmas (list): List of lemmas

    Returns:
        list: A list of words, with uninformative adjectives paired with following nouns
    """
    new_lemmas = []
    words_to_remove = []

    for idx, entry in enumerate(lemmas):
        word = entry[0]
        tag = entry[1]

        # look ahead to next entry to see if it is noun
        try:
            next_entry = lemmas[idx + 1]
        except IndexError:
            next_entry = (";;;", ";;;") # sentinel value

        next_word = next_entry[0]
        next_tag = next_entry[1]

        if "JJ" in tag and word in VAGUE_ADJECTIVES and "NN" in next_tag:
            new_lemmas.append(word + "_" + next_word)
            words_to_remove.append(next_word)
        else:
            new_lemmas.append(word)
        
    new_lemmas = [x for x in new_lemmas if x not in words_to_remove]
    
    return new_lemmas

def luminosity_func(
    x: numeric, 
    min_freq: numeric, 
    max_freq: numeric, 
    min: numeric=40, 
    max: numeric=80
) -> numeric:
    """Custom scaling for determining the luminosity of word color.
    I want higher frequencies to be darker and vice versa. As luminosity
    increases, shade gets lighter.

    Args:
        x (numeric): Word frequency
        min_freq (numeric): Minimum frequency value
        max_freq (numeric): Maximum frequency value
        min (numeric, optional): Start point on HSL scale. Defaults to 40.
        max (numeric, optional): End point on HSL scale. Defaults to 80.

    Returns:
        numeric: luminosity value
    """
    y = max + ((min - max) / (max_freq - min_freq)) * x

    return y

def my_color_func(dictionary: dict, col: str):
    """Function to color wordcloud. Red for cons, 
    greens for pros. Luminosity is defined by 
    luminosity_func.

    Args:
        dictionary (dict): The frequency dictionary generated by WordCloud
        col (str): The column name

    Returns:
        function: A function to pass into the .recolor method
    """
    min_freq = min(dictionary.values())
    max_freq = max(dictionary.values())
    def my_tf_color_func_inner(word, font_size, position, orientation, random_state=None, **kwargs):
        luminosity = luminosity_func(dictionary[word], min_freq, max_freq, 40, 90)
        if "cons" in col:
            hsl_string = "hsl(0, 100%%, %d%%)" % (luminosity)
        elif "pros" in col:
            hsl_string = "hsl(100, 100%%, %d%%)" % (luminosity)

        return hsl_string

    return my_tf_color_func_inner

def get_word_clouds(
    df: pd.DataFrame, 
    col: str, path: Path, 
    freq_path: Path=None, 
    recolor: bool=True
) -> None:
    """Generates phrases (gensim.models) and create worcloud.
    Gensim combines tokens with an underscore and removes the 2nd token,
    e.g. ["New", "York", "blah"] -> ["New_York", "blah"]
    WordCloud handles the frequency.

    Original arguments to WordCloud are:
        prefer_horizontal=0.5,
        stopwords=STOPWORDS,
        max_words=30,
        max_font_size=40, 
        scale=3,
        background_color="white",
        random_state=100

    Args:
        df (pd.DataFrame): input data
        path (Path): output path
    """
    phrases = gensim.models.phrases.Phrases(
        df["paired_lemmas"].tolist(), 
        threshold=5, 
        connector_words=gensim.models.phrases.ENGLISH_CONNECTOR_WORDS
    )

    text = " ".join(df["joined_lemmas"].tolist())
    
    wc = WordCloud(
        prefer_horizontal=1,
        stopwords=FINAL_STOP,
        max_words=40,
        max_font_size=40,
        scale=3,
        background_color="white",
        random_state=100
    )
    
    cloud = wc.generate(" ".join(phrases[text.split()]))

    if recolor == True:
        cloud = cloud.recolor(color_func=my_color_func(cloud.words_, col))
    plt.axis("off")
    plt.imshow(cloud)
    plt.savefig(path, dpi=500, bbox_inches="tight")
    plt.close()

    if freq_path is not None:
        dict_to_csv(my_dict=cloud.words_, path=freq_path)

def generate_cloud(
    df: pd.DataFrame, 
    col: str, 
    path: Path, 
    freq_path: Path=None,
    recolor: bool=True
) -> None:
    """Wrapper for previous functions

    Args:
        df (pd.DataFrame): input data
        col (str): input column
        path (Path): output path
    """
    df["tokens"] = [get_tokens(str(x)) for x in df[col]]
    df["lemmas"] = [clean_lemmas(get_lemmas(x)) for x in df["tokens"]]
    df["paired_lemmas"] = df["lemmas"].apply(pair_adj_noun)
    df["joined_lemmas"] = [" ".join(x) for x in df["paired_lemmas"]]

    get_word_clouds(df=df, col=col, path=path, freq_path=freq_path, recolor=recolor)

def main(
    file: Path=DAT_PATH / "All_Review_Text.csv",
    cols: list=SELECTED_COLS, 
    filters: list=SELECTED_FILTERS
) -> None:
    # Filter to only current employees

    start = time.time()

    df = pd.read_csv(file, parse_dates=["DealDate", "review_date"])

    end = time.time()
    print(f"Done reading in file in {round(end - start, 2)} seconds.")

    start = time.time()

    current = df.loc[df["review_employee_status"] == "Current Employee"]

    # Generate dummies

    current["pre"] = np.select(
        condlist=[
            (current["ever"].isnull()) | (current["post"].isnull()),
            (current["ever"] == True) & (current["post"] == False),
        ],
        choicelist=[pd.NA, True],
        default=False
    )

    current["control"] = np.select(
        condlist=[
            current["ever"].isnull(),
            current["ever"] == False
        ],
        choicelist=[pd.NA, True],
        default=False
    )

    current["pre_4qtrs"] = np.select(
        condlist=[
            (current["review_date"].isnull() | current["DealDate"].isnull() | current["pre"].isnull()),
            (current["pre"] == True) & (current["review_date"] < current["DealDate"] - pd.DateOffset(years=1))
        ],
        choicelist=[pd.NA, True],
        default=False
    )

    current["pre_8qtrs"] = np.select(
        condlist=[
            (current["review_date"].isnull() | current["DealDate"].isnull() | current["pre"].isnull()),
            (current["pre"] == True) & (current["review_date"] < current["DealDate"] - pd.DateOffset(years=2))
        ],
        choicelist=[pd.NA, True],
        default=False
    )
    
    current["post_4qtrs"] = np.select(
        condlist=[
            (current["review_date"].isnull() | current["DealDate"].isnull() | current["post"].isnull()),
            (current["post"] == True) & (current["review_date"] > current["DealDate"] + pd.DateOffset(years=1))
        ],
        choicelist=[pd.NA, True],
        default=False
    )

    current["post_8qtrs"] = np.select(
        condlist=[
            (current["review_date"].isnull() | current["DealDate"].isnull() | current["post"].isnull()),
            (current["post"] == True) & (current["review_date"] > current["DealDate"] + pd.DateOffset(years=2))
        ],
        choicelist=[pd.NA, True],
        default=False
    )

    end = time.time()
    print(f"Done filtering and creating dummies in {round(end - start, 2)} seconds.")

    sample_size_dict = dict()
    for col in cols:
        for filter in filters:
            start = time.time()

            col_filter = f"{col}_{filter}"

            current_filtered = current[[col, filter]]
            mask = (
                (current_filtered[col].notna()) &
                (current_filtered[filter] == True)
            )
            current_filtered = current_filtered.loc[mask]

            sample_size_dict[col_filter] = current_filtered.shape[0]

            generate_cloud(
                df=current_filtered, 
                col=col, 
                path=OUT_PATH / f"{col_filter}_wc.png",
                freq_path=OUT_PATH / f"{col_filter}_freqs.csv",
                recolor=True
            )

            end = time.time()
            print(f"Done with {col_filter} in {round(end - start, 2)} seconds.")
    
    dict_to_csv(my_dict=sample_size_dict, path=OUT_PATH / "sample_sizes.csv")

if __name__ == "__main__":
    main()