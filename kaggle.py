import sys
import numpy as np
from sklearn import svm
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier

def rf(x_train, x_test, y_train, y_test):
	print 'training random forest...'
	clf = RandomForestClassifier(n_estimators=30, random_state=40)
	clf.fit(x_train, y_train)
	print 'done.'

	print 'prediciting the labels of test data...'
	rf_result = clf.predict(x_test)
	print 'done.'

	success = 0
	for i in range (0, len(rf_result)):
		if y_test[i] == rf_result[i]:
			success += 1
	print 'RF Success Rate: ' + str(100 * (float(success)/len(y_test))) + '%'

def rf_quiz(x_train, y_train, x_test, output):
	print 'training random forest...'
	clf = RandomForestClassifier(n_estimators=300, max_features=150, random_state=40)
	clf.fit(x_train, y_train)
	print 'done.'

	print 'prediciting the labels of test data...'
	rf_result = clf.predict(x_test)
	print 'done.'

	print 'writing results into submission.csv...'
	fo = open(output, "w")
	fo.write('Id,Prediction\n')
	ID = 1
	for result in rf_result:
		s = str(ID) + ','+ str(int(result)) + '\n'
		fo.write(s)
		ID += 1
	print 'done.'

def knn(x_train, x_test, y_train, y_test):
	print 'training knn...'
	knn_clf = neighbors.KNeighborsClassifier()
	knn_clf.fit(x_train, y_train)
	print 'done.'

	print 'prediciting the labels of test data...'
	knn_result = knn_clf.predict(x_test)
	print 'done.'

	success = 0
	for i in range (0, len(knn_result)):
		if y_test[i] == knn_result[i]:
			success += 1
	print 'KNN Success Rate: ' + str(100 * (float(success)/len(y_test))) + '%'

def svm(x_train, x_test, y_train, y_test):
	print 'training svm...'
	svm_clf = svm.LinearSVC()
	svm_clf.fit(x_train, y_train)
	print 'done.'

	print 'prediciting the labels of test data...'
	svm_result = svm_clf.predict(x_test)
	print 'done.'

	success = 0
	for i in range (0, len(svm_result)):
		if y_test[i] == svm_result[i]:
			success += 1
	print 'SVM Success Rate: ' + str(100 * (float(success)/len(y_test))) + '%'

def process_train_data(data, categorical_features):
	print 'vectorizing training data...'
	x_data = []
	y_data = []
	for i in range (0, len(data)):
		features = data[i]

		c_f = 0
		new_features = []
		for j in range (0, len(features)-1):
			feature = features[j]

			try:
				feature = float(feature)
				new_features.append(feature)
			except:
				categories = categorical_features[c_f]
				zeros = np.zeros(len(categories))
				index = categories.index(feature)
				zeros[index] = 1
				feature = list(zeros)
				new_features += feature
				c_f += 1

		x_data.append(new_features)
		y_data.append(float(features[-1]))
	print 'done.'

	return x_data, y_data

def process_test_data(data, categorical_features):
	print 'vectorizing test data...'
	x_data = []
	y_data = []
	for i in range (0, len(data)):
		features = data[i]

		c_f = 0
		new_features = []
		for j in range (0, len(features)):
			feature = features[j]

			try:
				feature = float(feature)
				new_features.append(feature)
			except:
				categories = categorical_features[c_f]
				zeros = np.zeros(len(categories))
				index = categories.index(feature)
				zeros[index] = 1
				feature = list(zeros)
				new_features += feature
				c_f += 1

		x_data.append(new_features)
	print 'done.'

	return x_data

def main(argv):
	feature_0 = ['dctc', 'def', 'dem', 'demnum', 'el', 'indef', 'null', 'num', 
	'numpro', 'poss', 'posspro', 'pro', 'relpro']
	feature_5 = ['acomp', 'advcl', 'advmod', 'agent', 'amod', 'attr', 'aux', 
	'cc', 'ccomp', 'conj_and', 'conj_but', 'conj_just', 'conj_negcc', 'conj_or', 
	'csubj', 'dep', 'det', 'dobj', 'expl', 'iobj', 'mark', 'na', 'neg', 'nn', 
	'npadvmod', 'nsubj', 'nsubjpass', 'num', 'partmod', 'pcomp', 'pobj', 
	'poss', 'predet', 'prep', 'prep_about', 'prep_above', 'prep_across', 
	'prep_across_from', 'prep_after', 'prep_ahead_of', 'prep_along', 
	'prep_alongside', 'prep_apart_from', 'prep_around', 'prep_as', 'prep_at', 
	'prep_away_from', 'prep_because', 'prep_because_of', 'prep_before', 
	'prep_below', 'prep_beneath', 'prep_beside', 'prep_between', 'prep_beyond', 
	'prep_by', 'prep_close_to', 'prep_down', 'prep_far_from', 'prep_following', 
	'prep_for', 'prep_from', 'prep_if', 'prep_in', 'prep_in_front_of', 
	'prep_inside', 'prep_instead_of', 'prep_into', "prep_it's", 'prep_like', 
	'prep_near', 'prep_nearer', 'prep_next_to', 'prep_of', 'prep_off', 'prep_on', 
	'prep_on_top_of', 'prep_onto', 'prep_opposite', 'prep_out_of', 'prep_outside', 
	'prep_outside_of', 'prep_over', 'prep_past', 'prep_than', 'prep_that', 
	"prep_that's", 'prep_through', 'prep_to', 'prep_toward', 'prep_towards', 
	'prep_under', 'prep_underneath', 'prep_until', 'prep_up', "prep_where's", 
	'prep_with', 'prep_with_respect_to', 'prep_within', 'prepc_beneath', 'prepc_by', 
	'prepc_of', 'prepc_to', 'prepc_towards', 'prepc_under', 'prepc_with', 'quantmod', 
	'rcmod', 'rel', 'root', 'tmod', 'xcomp']
	feature_7 = ['f', 'g']
	feature_8 = ['acknowledge', 'align', 'check', 'clarify', 'explain', 'instruct', 
	'query_w', 'query_yn', 'ready', 'reply_n', 'reply_w', 'reply_y', 'uncodable']
	feature_9 = ['dctc', 'def', 'dem', 'demnum', 'el', 'indef', 'null', 'num', 
	'numpro', 'poss', 'posspro', 'pro', 'relpro']
	feature_14 = ['acomp', 'advcl', 'advmod', 'agent', 'amod', 'attr', 'aux', 'cc', 
	'ccomp', 'conj_and', 'conj_but', 'conj_just', 'conj_negcc', 'conj_or', 'csubj', 
	'dep', 'det', 'dobj', 'expl', 'iobj', 'mark', 'na', 'neg', 'nn', 'npadvmod', 
	'nsubj', 'nsubjpass', 'num', 'partmod', 'pcomp', 'pobj', 'poss', 'predet', 
	'prep', 'prep_about', 'prep_above', 'prep_across', 'prep_across_from', 
	'prep_after', 'prep_ahead_of', 'prep_along', 'prep_alongside', 'prep_apart_from', 
	'prep_around', 'prep_as', 'prep_at', 'prep_away_from', 'prep_because', 'prep_because_of', 
	'prep_before', 'prep_below', 'prep_beneath', 'prep_beside', 'prep_between', 
	'prep_beyond', 'prep_by', 'prep_close_to', 'prep_down', 'prep_far_from', 
	'prep_following', 'prep_for', 'prep_from', 'prep_if', 'prep_in', 'prep_in_front_of', 
	'prep_inside', 'prep_instead_of', 'prep_into', "prep_it's", 'prep_like', 'prep_near', 
	'prep_nearer', 'prep_next_to', 'prep_of', 'prep_off', 'prep_on', 'prep_on_top_of', 
	'prep_onto', 'prep_opposite', 'prep_out_of', 'prep_outside', 'prep_outside_of', 
	'prep_over', 'prep_past', 'prep_than', 'prep_that', "prep_that's", 'prep_through', 
	'prep_to', 'prep_toward', 'prep_towards', 'prep_under', 'prep_underneath', 'prep_until', 
	'prep_up', "prep_where's", 'prep_with', 'prep_with_respect_to', 'prep_within', 'prepc_beneath', 
	'prepc_by', 'prepc_of', 'prepc_to', 'prepc_towards', 'prepc_under', 'prepc_with', 'quantmod', 
	'rcmod', 'rel', 'root', 'tmod', 'xcomp']
	feature_16 = ['f', 'g']
	feature_17 = ['acknowledge', 'align', 'check', 'clarify', 'explain', 'instruct', 'query_w', 
	'query_yn', 'ready', 'reply_n', 'reply_w', 'reply_y', 'uncodable']
	feature_56 = ['acomp', 'advcl', 'advmod', 'agent', 'amod', 'ccomp', 'conj', 'conj_and', 
	'conj_but', 'conj_just', 'conj_negcc', 'conj_or', 'csubj', 'dep', 'dobj', 'infmod', 'na', 
	'neg', 'nn', 'npadvmod', 'nsubj', 'nsubjpass', 'partmod', 'pcomp', 'pobj', 'prep', 'prep_a--', 
	'prep_about', 'prep_above', 'prep_across', 'prep_across_from', 'prep_after', 'prep_along', 
	'prep_around', 'prep_as', 'prep_at', 'prep_away_from', 'prep_before', 'prep_below', 
	'prep_beneath', 'prep_beside', 'prep_between', 'prep_beyond', 'prep_by', 'prep_down', 
	'prep_for', 'prep_from', 'prep_if', 'prep_in', 'prep_into', "prep_it's", 'prep_like', 
	'prep_near', 'prep_next_to', 'prep_of', 'prep_off', 'prep_on', 'prep_opposite', 'prep_out', 
	'prep_over', 'prep_past', 'prep_than', 'prep_that', "prep_that's", "prep_they're", 
	'prep_through', 'prep_to', 'prep_towards', 'prep_under', 'prep_underneath', 'prep_until', 
	'prep_up', 'prep_with', 'prep_within', 'prepc_about', 'prepc_above', 'prepc_along', 
	'prepc_around', 'prepc_as', 'prepc_at', 'prepc_below', 'prepc_beneath', 'prepc_between', 
	'prepc_down', 'prepc_from', 'prepc_in', 'prepc_instead_of', 'prepc_like', 'prepc_next_to', 
	'prepc_of', 'prepc_on', 'prepc_over', 'prepc_past', 'prepc_than', 'prepc_to', 'prepc_underneath', 
	'prepc_until', 'prepc_with', 'rcmod', 'root', 'tmod', 'xcomp']
	feature_57 = ['acomp', 'advcl', 'advmod', 'agent', 'amod', 'ccomp', 'conj', 'conj_and', 
	'conj_but', 'conj_just', 'conj_negcc', 'conj_or', 'csubj', 'dep', 'dobj', 'infmod', 'na', 
	'neg', 'nn', 'npadvmod', 'nsubj', 'nsubjpass', 'partmod', 'pcomp', 'pobj', 'prep', 
	'prep_a--', 'prep_about', 'prep_above', 'prep_across', 'prep_across_from', 'prep_after', 
	'prep_along', 'prep_around', 'prep_as', 'prep_at', 'prep_away_from', 'prep_before', 
	'prep_below', 'prep_beneath', 'prep_beside', 'prep_between', 'prep_beyond', 'prep_by', 
	'prep_down', 'prep_for', 'prep_from', 'prep_if', 'prep_in', 'prep_into', "prep_it's", 
	'prep_like', 'prep_near', 'prep_next_to', 'prep_of', 'prep_off', 'prep_on', 'prep_opposite', 
	'prep_out', 'prep_over', 'prep_past', 'prep_starting', 'prep_than', 'prep_that', "prep_that's", 
	"prep_they're", 'prep_through', 'prep_to', 'prep_towards', 'prep_under', 'prep_underneath', 
	'prep_until', 'prep_up', 'prep_with', 'prep_within', 'prepc_about', 'prepc_above', 'prepc_along', 
	'prepc_around', 'prepc_as', 'prepc_at', 'prepc_below', 'prepc_beneath', 'prepc_between', 'prepc_down', 
	'prepc_from', 'prepc_in', 'prepc_instead_of', 'prepc_like', 'prepc_next_to', 'prepc_of', 'prepc_on', 
	'prepc_over', 'prepc_past', 'prepc_than', 'prepc_to', 'prepc_underneath', 'prepc_with', 'rcmod', 
	'root', 'tmod', 'xcomp']

	categorical_features = [feature_0, feature_5, feature_7, feature_8, feature_9, 
	feature_14, feature_16, feature_17, feature_56, feature_57]

	print 'extracting data from ' + argv[0] + '...'
	fo = open(argv[0], "r")
	s = fo.read()
	fo.close()
	s = s.split('\n')
	print 'removing bigram features...'
	data = []
	for i in range (1, len(s)-1):
		l = s[i].split(',')

		# remove bigram features
		for j in range(5):
			del l[10]
		del l[-7] # del l[-6] for test data 'quiz.csv'
		
		data.append(l)
	print 'done.'

	x_train, y_train = process_train_data(data, categorical_features)

	print 'extracting data from ' + argv[1] + '...'
	fo = open(argv[1], "r")
	s = fo.read()
	fo.close()
	s = s.split('\n')
	print 'removing bigram features...'
	data = []
	for i in range (1, len(s)-1):
		l = s[i].split(',')

		# remove bigram features
		for j in range(5):
			del l[10]
		del l[-6]
		
		data.append(l)
	print 'done.'
	
	x_test = process_test_data(data, categorical_features)

	#svm(x_train, x_test, y_train, y_test)
	#knn(x_train, x_test, y_train, y_test)

	#rf(x_train, x_test, y_train, y_test)
	rf_quiz(x_train, y_train, x_test, argv[2])

if __name__ == "__main__":
	main(sys.argv[1:])