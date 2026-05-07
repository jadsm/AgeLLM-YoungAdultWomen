# process the data
import os
import json
import pandas as pd
import numpy as np
import re
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt

def retreive_data(model_path):
    with open(os.path.join("outputs", model_path), "r") as f:
        raw_string = f.read().replace("```","").replace("json","")
        json_formatted = "[" + raw_string.replace("\n", "").replace("}", "},").strip(",") + "]"
        data_list = json.loads(json_formatted)
        

        dfa = pd.DataFrame(data_list)
        dfa.columns = [f.replace(" ","_") for f in dfa.columns]
        dfa['model'] = model_path.replace(".txt","").replace("output_","")
        dfa['patient'] = np.arange(1,8,1)
        print(f"Processed {model_path}")
        return dfa
    
# read the data
a = []
filter_conditions = "seed58"
for model_path in os.listdir("outputs"):
    if not model_path.endswith(".txt") or filter_conditions not in model_path:
        continue
    a.append(retreive_data(model_path))   

df = pd.concat(a, ignore_index=True)

# model pseudonyms
model_mapping = {
    "gemini-3.1-pro": "Gemini Pro",
    "gemini-3.1-pro-preview": "Gemini Pro",
    "gemini-3.1-flash-lite-preview": "Gemini Flash",
    "claude-sonnet-4-6": "Claude Sonnet",
    "qwen3-next-80b-a3b-instruct-maas": "Qwen 3",
    "gpt-oss-20b-maas": "GPT 20b"
}

df['cohort'] = None 
def assign_cohort(name):
    idx = df['model'].str.contains(name)
    df.loc[idx, 'cohort'] = name
    # get the trial number
    if "seed" in filter_conditions:
        df.loc[idx, 'trial'] = df.loc[idx, 'model'].str.extract(r'_(\d+)_seed', expand=False).astype(float)
        df.loc[idx,'model'] = df.loc[idx,'model'].str.replace(r"_(\d+)_"+filter_conditions, "",regex=True).str.strip("_")
    # remove the filter conditions 
    df.loc[idx,'model'] = df.loc[idx,'model'].str.replace(filter_conditions, "").str.strip("_")
    df.loc[idx,'model'] = df.loc[idx,'model'].str.replace(name, "").str.strip("_")
    return df
df = assign_cohort('women_pre_natal')
df = assign_cohort('women_post_natal')
df['cohort'] = df['cohort'].fillna("nature_paper")
df['model'] = df['model'].map(model_mapping)
if 'trial' in df.columns:
    df['trial'] = df['trial'].astype(int)

# calculate real age from text
def get_real_age(x):

    match = re.search(r"(\d+(?:\.\d+)?)-year-old", x)
    
    if match:
        age = match.group(1)
        # age_int = int(age)
        return age
    else:
        print("No age found in the text.")
        return None
    
real_age_mapping = { (row['patient'], row['cohort']): get_real_age(row['inference_process']) 
                    for _, row in df.query("model == 'Claude Sonnet'").iterrows() }
df['chronological_age'] = df.set_index(['patient', 'cohort']).index.map(real_age_mapping).astype(float)

df_reasoning = df.loc[:,['patient', 'model','inference_process', 'key_indicators']]
df.drop(columns=['inference_process', 'key_indicators'], inplace=True)
df_reasoning.to_csv("outputs/reasoning_and_indicators.csv", index=False)

dftall = df.melt(id_vars=['patient','cohort', 'model'], var_name='age_type', value_name='age_value')
dftall.groupby(["patient","age_type"])['age_value'].std()

# discrepancies in repeats
agecols = [col for col in df.columns if 'age' in col]
aux = df.groupby(['cohort','model','patient'])[agecols].std().reset_index()
aux.to_csv(f"outputs/discrepancies_repeats{filter_conditions}.csv")
aux.groupby(['model'])[agecols].agg(['mean','median','min','max']).T.to_csv(f"outputs/discrepancies_repeats{filter_conditions}_agg.csv")
aux.groupby(['model'])[agecols].median().T.to_csv(f"outputs/discrepancies_repeats{filter_conditions}_median.csv")


# concordance index
# This creates a table where each column is a different model's prediction
df_wide = dftall.pivot(index=['patient','cohort'], columns=['age_type','model'], values='age_value')

### METRIC 1: Intraclass Correlation Coefficient (ICC)
# This tells you how much the 5 models agree with each other (0 to 1 scale)
# for f in dftall['age_type'].unique():
#     dfaux = dftall.query(f"age_type == '{f}'")
#     icc = pg.intraclass_corr(data=dfaux, targets='patient', raters='model', ratings='age_value')
#     print(f"ICC Results for {f}:")
#     print(icc) # ICC3k measures consistency of fixed 'raters'


### METRIC 2: Age Acceleration (Δ-Age)
# We calculate the 'gap' for each model. 
# A good model should have a Δ-Age distribution that isn't wildly skewed.

dfnew_out = pd.DataFrame()
dfnew = pd.DataFrame()
for age in df_wide.columns.levels[0].unique():
    for model in df_wide.columns.levels[1].unique():
        if age != 'chronological_age':
            dfnew = pd.DataFrame()
            dfnew['delta'] = df_wide[(age, model)] - df_wide[('chronological_age',model)]
            dfnew['age'] = age            
            dfnew['model'] = model
            dfnew_out = pd.concat([dfnew_out, dfnew.reset_index()], ignore_index=True)
print("\nMean Age Acceleration per Model:")
# print(dfnew_out.groupby(['model','age'])['delta'].describe())
print(dfnew_out.groupby(['model','age'])['delta'].describe()['mean'].reset_index().pivot(columns='model',index='age'))
dfnew_out.groupby(['cohort','model','age'])['delta'].describe().to_csv(f'outputs/age_acceleration_summary{filter_conditions}.csv')
dfnew_out.pivot_table(columns=['cohort','model'],index='age').to_csv(f'outputs/age_acceleration_summary_cohort{filter_conditions}.csv')
dfnew_out.pivot_table(columns=['cohort','model'],index='age',aggfunc='median').to_csv(f'outputs/age_acceleration_summary_median_cohort{filter_conditions}.csv')

### METRIC 3: Inter-Model Correlation Heatmap
# This visualizes which models are "twins" and which is the outlier.
# ATTENTION: if this is not empty, it will filter just the ones indicated
filter_categories = ['women_post_natal', 'women_pre_natal']
corr_means = pd.DataFrame()
for age in df_wide.columns.levels[0].unique():
    plt.figure(figsize=(8, 6))
    
    subset = df_wide[df_wide.index.get_level_values(1).isin(filter_categories)] if len(filter_categories)!=0 else df_wide.copy()
    subset = subset[age] 

    correlation_matrix = subset.corr()
    # correlation_matrix = df_wide[[(col[0], col[1]) for col in df_wide.columns if col[0] == age]].corr()
    # mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(
        correlation_matrix, 
        # mask=mask, 
        annot=True, 
        fmt=".2f", 
        cmap='coolwarm', 
        vmin=0, vmax=1,
        cbar=False
    )
    # Fix labels: Rotate and align
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title(age.replace("_"," ").capitalize() + " Correlation Matrix", pad=20)
    plt.tight_layout()
    # save figures as svgs
    plt.savefig(f'outputs/correlation_matrix_{age}.svg')
    corr_means[age] = correlation_matrix.mean().round(2)
corr_means.to_csv(f'outputs/correlation_means_women{filter_conditions}.csv')
a = 0