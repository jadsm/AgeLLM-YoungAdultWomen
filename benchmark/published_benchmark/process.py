# process the data
import os
import json
import pandas as pd
import numpy as np
import re
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt


# read the data
a = []
for model_path in os.listdir("outputs"):
    if not model_path.endswith(".txt"):
        continue
    with open(os.path.join("outputs", model_path), "r") as f:
        raw_string = f.read().replace("```","").replace("json","")
        json_formatted = "[" + raw_string.replace("\n", "").replace("}", "},").strip(",") + "]"
        data_list = json.loads(json_formatted)
        

        dfa = pd.DataFrame(data_list)
        dfa.columns = [f.replace(" ","_") for f in dfa.columns]
        dfa['model'] = model_path.replace(".txt","").replace("output_","")
        dfa['patient'] = np.arange(1,8,1)
        a.append(dfa)
        print(f"Processed {model_path}")

df = pd.concat(a, ignore_index=True)

# model pseudonyms
model_mapping = {
    "gemini-3.1-pro": "Gemini Pro",
    "gemini-3.1-flash-lite-preview": "Gemini Flash",
    "claude-sonnet-4-6": "Claude Sonnet",
    "qwen3-next-80b-a3b-instruct-maas": "Qwen 3",
    "gpt-oss-20b-maas": "GPT 20b"
}

df['model'] = df['model'].map(model_mapping)


# calculate real age from text
def get_real_age(x):

    match = re.search(r"This (\d+)-year-old", x)
    
    if match:
        age = match.group(1)
        age_int = int(age)
        return age_int
    else:
        print("No age found in the text.")
        return None
    
real_age_mapping = {ii+1:i for ii,i in enumerate(df.query("model == 'Claude Sonnet'")['inference_process'].apply(get_real_age).values)}
df['chronological_age'] = df['patient'].map(real_age_mapping)

df_reasoning = df.loc[:,['patient', 'model','inference_process', 'key_indicators']]
df.drop(columns=['inference_process', 'key_indicators'], inplace=True)

dftall = df.melt(id_vars=['patient', 'model'], var_name='age_type', value_name='age_value')

dftall.groupby(["patient","age_type"])['age_value'].std()


# concordance index

# This creates a table where each column is a different model's prediction
df_wide = dftall.pivot(index='patient', columns=['age_type','model'], values='age_value')

### METRIC 1: Intraclass Correlation Coefficient (ICC)
# This tells you how much the 5 models agree with each other (0 to 1 scale)
for f in dftall['age_type'].unique():
    dfaux = dftall.query(f"age_type == '{f}'")
    icc = pg.intraclass_corr(data=dfaux, targets='patient', raters='model', ratings='age_value')
    print(f"ICC Results for {f}:")
    print(icc) # ICC3k measures consistency of fixed 'raters'


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
            dfnew_out = pd.concat([dfnew_out, dfnew], ignore_index=True)
print("\nMean Age Acceleration per Model:")
print(dfnew_out.groupby(['model','age'])['delta'].describe())
dfnew_out.groupby(['model','age'])['delta'].describe().to_csv('outputs/age_acceleration_summary.csv')

### METRIC 3: Inter-Model Correlation Heatmap
# This visualizes which models are "twins" and which is the outlier.
corr_means = pd.DataFrame()
for age in df_wide.columns.levels[0].unique():
    plt.figure(figsize=(8, 6))
    subset = df_wide[age] 
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
        cbar_kws={"shrink": .8}
    )
    # Fix labels: Rotate and align
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title(age.replace("_"," ").capitalize() + " Correlation Matrix", pad=20)
    plt.tight_layout()
    # save figures as svgs
    plt.savefig(f'outputs/correlation_matrix_{age}.svg')
    corr_means[age] = correlation_matrix.mean().round(2)
corr_means.to_csv('outputs/correlation_means.csv')
a = 0