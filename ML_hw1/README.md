Commandline arguments:  
--data_dir: The directory of the dataset (default="bank/")  
--model: Name of the model to create, there are currently 5 types of models, namely, DecisionTreeClassifier, AdaBoost, BaggedTrees, RandomForest, LMS (default="DecisionTreeClassifier")  
--feature_sample: Number of feature to consider for each split (default=None)  
--data_sample: Number of examples to sample from dataset (default=None)  
--num_trees: Number of trees to create (default=500)  
--max_depth: Maximum depth of each tree (default=-1)  
--split_scheme: Scheme to choose best feature to split on (default="Information Gain")  
--train_error: If user want to get the train error (default=True)  
--test_error: If user want to get the test error (default=True)  

Example run to create a random forest with 4 features each split and 1000 example per tree:  
python3 main.py --model=RandomForest --num_trees=10 --data_sample=1000 --feature_sample=4  

After you run, the model will be save as RandomForest.pickle