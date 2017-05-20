# Question_Identifier
question_identifier is a machine learning module which is trained to classify a given sentence into five categories.
  * Affirmation
  * Unknown
  * What
  * When
  * Who

## Input Data
Input data should be in a text file(ideally kept at data folder). It should be comma separated. Ideally it should have two clumns. 
  * Text
  * Category
## Packages to run module
list of packages required to run this module has been published in "requirement.txt". Please install the packages before running this module.

## Usage
python run_pipeline.py --input data/LabelledData.txt <--param value>

Other parameters(optional) given to model
  * model:  Model name could be one of 1.random_forest (default), 2. ada_boost 3. naive_bayes
  * test_ratio: Faction of data to validate model, default is 0.3
  * max_df: This will remove words whose freq is more in corpus, default  is 0.8
  * tree: No of tree in model, default is 100
  * jobs: No of jobs to run parallel, default is 2
  * tree_adaboost: This is used only for adaboost method. no of iteration of boosting, default is 5
  * train_test: This is boolean parameter. if true, it will train and test model. if fasle, it will test already selected model for complete dataset.

