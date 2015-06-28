"""
TODO: fill in
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor

class Preprocess(object):
	
	def __init__(self, yelp_ids_path, reviews_path, labels_path):
		self.yelp_ids_path = yelp_ids_path
		self.reviews_path = reviews_path
		self.labels_path = labels_path

	def get_ids(self):
		"""
		TODO: Fill in
		"""
		id_map = pd.read_csv(self.yelp_ids_path)
		id_dict = {}
		for i, row in id_map.iterrows():
			boston_id = row["restaurant_id"]
			non_null_mask = ~pd.isnull(row.ix[1:])
			yelp_ids = row[1:][non_null_mask].values
			for yelp_id in yelp_ids:
			    id_dict[yelp_id] = boston_id

		return id_dict


	def reviews_to_df(self):
		with open(self.reviews_path) as review_file:
			review_json = '[' + ','.join(review_file.readlines()) + ']'
		reviews = pd.read_json(review_json)
		reviews.drop(['review_id', 'type', 'user_id', 'votes'], 
		             inplace=True, 
		             axis=1)
		
		id_dict = self.get_ids()
		map_to_boston_ids = lambda yelp_id: id_dict[yelp_id] if yelp_id in id_dict else np.nan
		reviews.business_id = reviews.business_id.map(map_to_boston_ids)
		reviews.columns = ["restaurant_id", "date", "stars", "text"]
		reviews = reviews[pd.notnull(reviews.restaurant_id)]
		return reviews


	def flatten_reviews(self):
	    """ label_df: inspection dataframe with date, restaurant_id
	        reviews: dataframe of reviews
	        
	        Returns all of the text of reviews previous to each
	        inspection listed in label_df.
	    """
	    reviews = self.reviews_to_df()
	    label_df = pd.read_csv(self.labels_path, index_col=0)
	    reviews_dictionary = {}
	    N = len(label_df)
	    for i, (pid, row) in enumerate(label_df.iterrows()):
	        pre_inspection_mask = (reviews.date < row.date) & (reviews.restaurant_id == row.restaurant_id)

	        # pre-inspection reviews
	        pre_inspection_reviews = reviews[pre_inspection_mask]
	        all_text = ' '.join(pre_inspection_reviews.text)
	        reviews_dictionary[pid] = all_text
	        
	        if i % 2500 == 0:
	            print '{} out of {}'.format(i, N)
	    # return series in same order as the original data frame
	    return pd.Series(reviews_dictionary)[label_df.index]


	def build_x(self):
		"""
		TODO: fill in
		"""
		train_text = self.flatten_reviews()
		vec = TfidfVectorizer(stop_words='english',max_features=1500, use_idf=True, ngram_range=(1, 3))
		train_tfidf = vec.fit_transform(train_text)
		return train_tfidf

	def build_y(self):
		train_targets = train_labels[['*', '**', '***']].astype(np.float64)
		return train_targets

	# def get_uniq_index_xy():
	# 	pass

	# def combine_xy():
	# 	return x, y 

class Predict(object):
	"""
	TODO
	"""
	def __init__(X, y):
		self.X = X
		self.y = y


	def rf_regressor(self):
		X = X.toarray() # Convert X from sparse to array
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

		model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)
		model.fit(X_train, y_train)
		return model.score(X_test, y_test).round(2)


def main():
	yelp_ids_file = 'data/yelp_boston_academic_dataset/restaurant_ids_to_yelp_ids.csv' 
	reviews_file = 'data/yelp_boston_academic_dataset/yelp_academic_dataset_review.json'
	labels_file = 'data/yelp_boston_academic_dataset/train_labels.csv'
	preprocess = Preprocess(yelp_ids_file, reviews_file, labels_file)
	X = preprocess.build_x()
	y = preprocess.build_y()
	predict = Predict(X, y)
	score = predict.rf_regressor()
	print score


if __name__ == '__main__':
	main()









