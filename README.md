# learner1




models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('NB', GaussianNB()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVM', SVC(gamma='auto')))
models.append(('CART', DecisionTreeClassifier()))
models.append(('BAG', BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=1), n_estimators=100, random_state=1)))
models.append(('FOREST',RandomForestClassifier(n_jobs=2, random_state=0)))
models.append(('ADA',AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2, random_state=1), n_estimators=20, random_state=1)))
models.append(('GBoost',GradientBoostingClassifier()))
