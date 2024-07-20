import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds


def recommend_films(user_id, ratings_matrix, top_n=5):
    recommended_id = ratings_matrix.sort_values(by=user_id, axis=0, ascending=False).index[:top_n]

    file_path = 'movies.csv'
    df = pd.read_csv(file_path)
    
    recommended_movies = df[df['movieId'].isin(recommended_id)]
    recommended_movies = recommended_movies[['title', 'genres']]
    
    return recommended_movies


def SVD(matrix):
    S_right = np.dot(matrix.T, matrix)
    S_left = np.dot(matrix, matrix.T)

    values_right, V = np.linalg.eigh(S_right)
    values_left, U = np.linalg.eigh(S_left)

    index_right = np.argsort(values_right)[::-1]
    index_left = np.argsort(values_left)[::-1]
    values_right = values_right[index_right]
    values_left = values_left[index_left]
    U = U[:, index_left]
    V = V[:, index_right]
    
    singular_values = np.sqrt(values_right)

    S = np.zeros(matrix.shape)
    np.fill_diagonal(S, singular_values)

    reconstructed_matrix = np.dot(U, np.dot(S, V.T))

    return U, S, V.T, reconstructed_matrix


A = np.array([[5, 3, 2], [1, 4, 7]])
U, S, Vt, A_rec = SVD(A)

file_path = 'ratings.csv'
df = pd.read_csv(file_path)

ratings_matrix = df.pivot(index='userId', columns='movieId', values='rating')
ratings_matrix = ratings_matrix.dropna(thresh=25, axis=0)
ratings_matrix = ratings_matrix.dropna(thresh=25, axis=1)
print(ratings_matrix)

ratings_matrix_filled = ratings_matrix.fillna(2.5)
R = ratings_matrix_filled.values
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

U, sigma, Vt = svds(R_demeaned, k=3)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
for i in range(0,50):
    ax.scatter(U[i,0], U[i,1], U[i,2])
plt.show()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
for i in range(0,50):
    ax.scatter(Vt[0,i], Vt[1,i], Vt[2,i])
plt.show()

U, sigma, Vt = svds(R_demeaned, k=3)
sigma = np.diag(sigma)

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns=ratings_matrix.columns, index=ratings_matrix.index)
print(preds_df)

preds_df[ratings_matrix.notna()] = np.nan
print(preds_df)

print(recommend_films(10, preds_df, 5))