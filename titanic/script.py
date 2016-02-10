import pandas 
import numpy as np

titanic = pandas.read_csv("train.csv")
titanic_test = pandas.read_csv("test.csv")

titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
titanic_test["Age"] = titanic_test["Age"].fillna(titanic_test["Age"].median())

titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1
#----------------------------------------------------------------------

titanic["Embarked"] = titanic["Embarked"].fillna("S")
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")
titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2
#------------------------------------------------------------------------

titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic["Fare"].median())

#---------------------------------------------------------------------------
titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]
titanic["SexPclass"] = titanic["Sex"] * titanic["Pclass"]
titanic_test["SexPclass"] = titanic_test["Sex"] * titanic_test["Pclass"]
titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]
#-----------------------------------------------------------------------------
# The .apply method generates a new series
titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))
titanic_test["NameLength"] = titanic_test["Name"].apply(lambda x: len(x))

#-------------------------------------------------------------------------

import re

# A function to get the title from a name.
def get_title(name):
    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

# Get all the titles and print how often each one occurs.
titles = titanic["Name"].apply(get_title)
#print(pandas.value_counts(titles))
# Map each title to an integer.  Some titles are very rare, and are compressed into the same codes as other titles.
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k,v in title_mapping.items():
    titles[titles == k] = v

# Verify that we converted everything.
#print(pandas.value_counts(titles))

# Add in the title column.
titanic["Title"] = titles

titles = titanic_test["Name"].apply(get_title)
# We're adding the Dona title to the mapping, because it's in the test set, but not the training set
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 10}
for k,v in title_mapping.items():
    titles[titles == k] = v
titanic_test["Title"] = titles
#---------------------------------------------------------------------------------------------------------------------------
titanic['Cabin']= titanic.Cabin.fillna("U")
titanic['Cabin']= titanic.Cabin.apply(lambda x: str(x)[-1])
"""
for i in range(len(titanic['Cabin'].unique())):
    titanic.loc[titanic["Cabin"] == titanic['Cabin'].unique()[i], "Cabin"] = i
"""
"""    
titanic_test['Cabin']= titanic_test.Cabin.fillna("U")
titanic_test['Cabin']= titanic_test.Cabin.apply(lambda x: str(x)[0])
titanic_test.loc[titanic_test["Cabin"] == "A" or titanic_test["Cabin"] == "B" , "Embarked"] = 0
titanic_test.loc[titanic_test["Cabin"] == , "Embarked"] = 0

for i in range(len(titanic_test['Cabin'].unique())):
    titanic_test.loc[titanic_test["Cabin"] == titanic_test['Cabin'].unique()[i], "Cabin"] = i

#--------------------------------------------------------------------------------------------------------
import operator

# A dictionary mapping family name to id
family_id_mapping = {}

# A function to get the id given a row
def get_family_id(row):
    # Find the last name by splitting on a comma
    last_name = row["Name"].split(",")[0]
    # Create the family id
    family_id = "{0}{1}".format(last_name, row["FamilySize"])
    # Look up the id in the mapping
    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            current_id = 1
        else:
            # Get the maximum id from the mapping and add one to it if we don't have an id
            current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
        family_id_mapping[family_id] = current_id
    return family_id_mapping[family_id]

# Get the family ids with the apply method
family_ids = titanic.apply(get_family_id, axis=1)

# There are a lot of family ids, so we'll compress all of the families under 3 members into one code.
family_ids[titanic["FamilySize"] < 3] = -1

# Print the count of each unique id.
#print(pandas.value_counts(family_ids))

titanic["FamilyId"] = family_ids
	
family_ids = titanic_test.apply(get_family_id, axis=1)
family_ids[titanic_test["FamilySize"] < 3] = -1
titanic_test["FamilyId"] = family_ids
titanic_test["NameLength"] = titanic_test["Name"].apply(lambda x: len(x))

#-----------------------------------------------------------------------

#------------------------------------------------------------------------
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId", "SexPclass","Cabin"]
algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare","Embarked", "FamilySize", "Title", "FamilyId", "Cabin", "SexPclass"]],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]

# Initialize the cross validation folds
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    train_target = titanic["Survived"].iloc[train]
    full_test_predictions = []
    # Make predictions for each algorithm on each fold
    for alg, predictors in algorithms:
        # Fit the algorithm on the training data.
        alg.fit(titanic[predictors].iloc[train,:], train_target)
        # Select and predict on the test fold.  
        # The .astype(float) is necessary to convert the dataframe to all floats and avoid an sklearn error.
        test_predictions = alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)
    # Use a simple ensembling scheme -- just average the predictions to get the final classification.
    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
    # Any value over .5 is assumed to be a 1 prediction, and below .5 is a 0 prediction.
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)

# Put all the predictions together into one array.
predictions = np.concatenate(predictions, axis=0)

# Compute accuracy by comparing to the training data.
accuracy = sum(predictions == titanic["Survived"]) / float(len(predictions))
print(accuracy)

algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]] 
]

full_predictions = []
for alg, predictors in algorithms:
    # Fit the algorithm using the full training data.
    alg.fit(titanic[predictors], titanic["Survived"])
    # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error.
    predictions = alg.predict_proba(titanic_test[predictors].astype(float))[:,1]
    full_predictions.append(predictions)

# The gradient boosting classifier generates better predictions, so we weight it higher.
predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4
predictions[predictions <= .5] = 0
predictions[predictions > .5] = 1
predictions = predictions.astype(int)


# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pandas.DataFrame({"PassengerId": titanic_test["PassengerId"], "Survived": predictions})
submission.to_csv("kaggle.csv", index=False)
"""
