import unittest
import pandas as pd
from recommendation_system import load_data, create_user_item_matrix, compute_cosine_similarity, recommend_movies

class TestRecommendationSystem(unittest.TestCase):
    def test_load_data(self):
        ratings, movies = load_data()
        self.assertEqual(ratings.shape[0], 100000)  # Assuming 100,000 ratings
        self.assertEqual(movies.shape[0], 10000)  # Assuming 10,000 movies

    def test_create_user_item_matrix(self):
        ratings, _ = load_data()
        user_item_matrix = create_user_item_matrix(ratings)
        self.assertEqual(user_item_matrix.shape[0], 6000)  # Assuming 6,000 users
        self.assertEqual(user_item_matrix.shape[1], 4000)  # Assuming 4,000 movies

    def test_recommend_movies(self):
        ratings, movies = load_data()
        user_item_matrix = create_user_item_matrix(ratings)
        cosine_sim = compute_cosine_similarity(user_item_matrix)
        recommendations = recommend_movies(1, user_item_matrix, cosine_sim, movies)
        self.assertGreater(len(recommendations), 0)
        self.assertTrue(all(isinstance(movie, str) for movie in recommendations))

if __name__ == "__main__":
    unittest.main()
