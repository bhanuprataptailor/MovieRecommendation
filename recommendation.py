from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
# text1 = ["London Paris London", "Paris Paris London"]

# # print(1)
# print(cv.get_feature_names())
# print(cv_fit.toarray())
# similarity = cosine_similarity(cv_fit.toarray())
# print(similarity)
df = pd.read_csv('movie_dataset.csv')

features = ['keywords', 'cast', 'genres', 'director']

def combine_features(row):
	return str(row['cast'])+" "+str(row["keywords"])+" "+str(row["genres"])+" "+str(row['director'])

def get_title_from_index(index):
	return df[df.index==index]['title'].values[0]

def get_index_from_title(title):
	return df[df.title==title]['index'].values[0]

df["combined_features"] = df.apply(combine_features, axis=1)
print(df['combined_features'].head())
cv = CountVectorizer()
cv_fit=cv.fit_transform(df["combined_features"])
# print(cv_fit.toarray())
similarity = cosine_similarity(cv_fit.toarray())
# print(len(similarity[0]))
movie_user_like = input('Enter the movie\n')
movie_index = get_index_from_title(movie_user_like)
similar_movies = list(enumerate(similarity[movie_index]))
sorted_result = sorted(similar_movies, key=lambda x:x[1], reverse=True)
i =0 
for result in sorted_result:
	if result[1]>0.01:
		i=i+1
		print(i, get_title_from_index(result[0]))
