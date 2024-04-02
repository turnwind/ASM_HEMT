
def suggest_metrics(report):

    # Note: The prompt is generated and optimized by ChatGPT (GPT-4)
    prompt = f"""
    The classification problem under investigation is based on a network 
    intrusion detection dataset. This dataset contains Probe attack type, 
    which are all grouped under the "attack" class (label: 1). Conversely, 
    the "normal" class is represented by label 0. Below are the dataset's 
    characteristics:
    {report}.

    For this specific inquiry, you are tasked with recommending a suitable 
    hyperparameter optimization metric for training a XGBoost model. It is 
    crucial that the model should accurately identify genuine threats (attacks) 
    without raising excessive false alarms on benign activities. They are equally 
    important. Given the problem context and dataset characteristics, suggest 
    only the name of one of the built-in metrics: 
    - 'accuracy'
    - 'roc_auc' (ROCAUC score)
    - 'f1' (F1 score)
    - 'balanced_accuracy' (It is the macro-average of recall scores per class 
    or, equivalently, raw accuracy where each sample is weighted according to 
    the inverse prevalence of its true class) 
    - 'average_precision'
    - 'precision'
    - 'recall'
    - 'neg_brier_score'


    Please first briefly explain your reasoning and then provide the 
    recommended metric name. Your recommendation should be enclosed between 
    markers [BEGIN] and [END], with standalone string for indicating the 
    metric name.
    Do not provide other settings or configurations.
    """

    return prompt


def data_report(df, num_feats, bin_feats, nom_feats):
    """
    Generate data characteristics report.

    Inputs:
    -------
    df: dataframe for the dataset.
    num_feats: list of names of numerical features.
    bin_feats: list of names of binary features.
    nom_feats: list of names of nominal features.

    Outputs:
    --------
    report: data characteristics report.
    """

    # Label column
    target = df.iloc[:, -1]
    features = df.iloc[:, :-1]

    # General dataset info
    num_instances = len(df)
    num_features = features.shape[1]

    # Class imbalance analysis
    class_counts = target.value_counts()
    class_distribution = class_counts / num_instances

    # Create report
    # Note: The format of the report is generated and optimized
    # by ChatGPT (GPT-4)
    report = f"""Data Characteristics Report:

- General information:
  - Number of Instances: {num_instances}
  - Number of Features: {num_features}

- Class distribution analysis:
  - Class Distribution: {class_distribution.to_string()}

- Feature analysis:
  - Feature names: {features.columns.to_list()}
  - Number of numerical features: {len(num_feats)}
  - Number of binary features: {len(bin_feats)}
  - Binary feature names: {bin_feats}
  - Number of nominal features: {len(nom_feats)}
  - Nominal feature names: {nom_feats}
"""

    return report