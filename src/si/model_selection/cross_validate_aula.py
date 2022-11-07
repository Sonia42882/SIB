
def cross_validate (model, dataset, scoring, cv, test_size):
    scores = {"seeds": [], "train":[], "test":[]}
    for i in range(cv):
        random_state = np.random.randint(0, 1000)
        scores["seeds"].append(random_state)
        train, test = train_test_split(dataset = dataset, test_size = test_size, random_state = random_state)
        model.fit(train)
        if scoring is None:
            scores["train"].append(model.score(train))
            scores["test"].append(model.score(test))
        else:
            y_train = train.y
            y_test = test.y
            scores["train"].append(scoring(y_train, model.predict(train)))
            scores["test"].append(scoring(y_test, model.predict(test)))
    return scores


