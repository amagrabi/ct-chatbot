""" Get the hipchat data somehow..."""

from get_data import get_data
df = get_data()


# Most common words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

df = df[df.room == 'Berlin']

# Split in train and data and valid
train = open("test/data/berlin_train", "w")
test = open("test/data/berlin_test", "w")
valid = open("test/data/berlin_valid", "w")
mylist = df.message.values # 7010
train.write("\n".join(map(lambda x: str(x), mylist[0:4000])))
test.write("\n".join(map(lambda x: str(x), mylist[4001:5500])))
valid.write("\n".join(map(lambda x: str(x), mylist[5501:-1])))
train.close()
test.close()
valid.close()

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
messages = vectorizer.fit_transform(df.message.values).build_analyzer()

tokenize_text = CountVectorizer(stop_words='english', lowercase=True, max_df=0.8, min_df=0.2).build_analyzer()
tokenize_text(df.message.values)



