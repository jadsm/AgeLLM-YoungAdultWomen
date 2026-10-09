# process the data
import os
import json
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.path.join(HERE, "outputs")
# _number_patients = 20
_prefix_to_remove = r"output\_\d+yo\_"
_suffix_to_remove = "_prompt0.txt"
SOURCE_DATA = os.path.join(HERE, "data", "source_data")
# source file per cohort (same inputs as main.py), used to read the chronological age
source_files = {
    "8yo": "gusto_y8_selected20.jsonl",
    "10yo": "gusto_y10_selected20.jsonl",
    "12yo": "gusto_y12_selected20.jsonl",
    "women_pre_natal": "women_before_pregnancy.jsonl",
    "women_post_natal": "women_after_pregnancy.jsonl",
}

def parse_minimax(raw_string):
    pattern = r"\{.*?\}"
    extracted = "".join(re.findall(pattern, raw_string, re.DOTALL))
    return extracted


def retreive_data(model_path):
    print('retrieving...',model_path)
    with open(os.path.join(OUTPUTS, model_path), "r",encoding='utf-8') as f:
        raw_string = f.read().replace("```","").replace("json","")
        raw_string = parse_minimax(raw_string) if 'minimax' in model_path else raw_string

        json_formatted = "[" + raw_string.replace("\n", "").replace("}", "},").strip(",") + "]"
        data_list = json.loads(json_formatted)

        dfa = pd.DataFrame(data_list)
        dfa.columns = [f.replace(" ","_") for f in dfa.columns]
        dfa['model'] = re.sub(_prefix_to_remove, "", model_path.replace(_suffix_to_remove,""))
        dfa['patient'] = np.arange(1,dfa.shape[0]+1,1)
        # age group from the file name, e.g. output_8yo_... -> 8yo
        age_group = re.search(r"output_(\d+yo)_", model_path)
        dfa['cohort'] = age_group.group(1) if age_group else None
        print(f"Processed {model_path}")
        return dfa
    
# read the data
a = []
# filter_conditions = "seed[0-4]"
# filter_conditions = "prompt[0-2]"
filter_conditions = "yo_"
children_cohort = True
exclusion_filters = ""
for model_path in os.listdir(OUTPUTS):
    if not model_path.endswith(".txt") or not re.search(rf"{filter_conditions}", model_path) or (len(exclusion_filters)>0 and re.search(rf"{exclusion_filters}", model_path)):
        continue
    a.append(retreive_data(model_path))   

df = pd.concat(a, ignore_index=True)

# model pseudonyms
model_mapping = {
    "gemini-3.1-pro": "Gemini Pro",
    "gemini-3.1-pro-preview": "Gemini Pro",
    "gemini-3.1-flash-lite-preview": "Gemini Flash",
    "gemini-3.8-flash":"Gemini Flash",
    "gemini-3.5-flash-lite": "Gemini Flash Lite",
    "claude-sonnet-4-6": "Claude Sonnet",
    "claude-sonnet-5": "Claude Sonnet",
    "qwen3-next-80b-a3b-instruct-maas": "Qwen 3",
    "gpt-oss-20b-maas": "GPT 20b",
    'minimax-m2-maas':'minimax'
}

def assign_cohort(name):
    idx = df['model'].str.contains(name)
    df.loc[idx, 'cohort'] = name
    # get the trial number
    if "seed" in filter_conditions:
        df.loc[idx, 'trial'] = df.loc[idx, 'model'].str.extract(r'_(\d+)_seed', expand=False).astype(float)
        df.loc[idx,'model'] = df.loc[idx,'model'].str.replace(r"_(\d+)_"+filter_conditions, "",regex=True).str.strip("_")
    # remove the filter conditions 
    df.loc[idx,'model'] = df.loc[idx,'model'].str.replace(filter_conditions, "",regex=True).str.strip("_")
    df.loc[idx,'model'] = df.loc[idx,'model'].str.replace(name, "").str.strip("_")
    return df
df = assign_cohort('women_pre_natal')
df = assign_cohort('women_post_natal')
df['cohort'] = df['cohort'].fillna("nature_paper")
df['model'] = df['model'].map(model_mapping)
if 'trial' in df.columns:
    df['trial'] = df['trial'].astype(int)

# chronological age from the "Age: X years old" line of the source input text
def get_chronological_ages(cohort):
    with open(os.path.join(SOURCE_DATA, source_files[cohort]), "r") as f:
        texts = [json.loads(line)["input"] for line in f.read().splitlines()]
    return {patient: float(re.search(r"Age: ?(\d+(?:\.\d+)?)", text).group(1))
            for patient, text in enumerate(texts, start=1)}

# every "Name: value" field of the source input text (one per line, or several per line separated by ". ")
_variable_pattern = re.compile(r"(?:^|(?<=\. ))([A-Za-z][^:.\n]*?):\s*(.*?)(?=\.\s+[A-Za-z][^:.\n]*?:|\.?$)", re.MULTILINE)
def get_all_variables(cohort):
    with open(os.path.join(SOURCE_DATA, source_files[cohort]), "r") as f:
        texts = [json.loads(line)["input"] for line in f.read().splitlines()]
    return pd.DataFrame([{name.strip(): value.strip() for name, value in _variable_pattern.findall(text)} for text in texts],
                        index=pd.RangeIndex(1, len(texts) + 1, name="patient"))

real_age_mapping = {(patient, cohort): age
                    for cohort in df['cohort'].unique()
                    for patient, age in get_chronological_ages(cohort).items()}
df['chronological_age'] = df.set_index(['patient', 'cohort']).index.map(real_age_mapping).astype(float)

get_all_variables(cohort)

# make sure all ages are floats
for c in df.columns:
    if 'age' in c:
        df[c] = df[c].astype(float)

df_reasoning = df.loc[:,['cohort','patient', 'model','inference_process', 'key_indicators']]
df.drop(columns=['inference_process', 'key_indicators'], inplace=True)
df_reasoning.to_csv(OUTPUTS+f"/reasoning_and_indicators{filter_conditions}.csv", index=False)

# discrepancies in repeats
agecols = [col for col in df.columns if 'age' in col]
aux = df.groupby(['cohort','model','patient'])[agecols].std().reset_index()
aux.to_csv(OUTPUTS+f"/discrepancies_repeats{filter_conditions}.csv")
aux.groupby('model')[agecols].agg(['mean','median','min','max']).T.to_csv(OUTPUTS+f"/discrepancies_repeats{filter_conditions}_agg.csv")
aux.groupby(['model'])[agecols].median().T.to_csv(OUTPUTS+f"/discrepancies_repeats{filter_conditions}_median.csv")

# transpose the data for further analysis
dftall = df.melt(id_vars=['patient','cohort', 'model'], var_name='age_type', value_name='age_value')
dftall.groupby(["patient","age_type"])['age_value'].std()

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
dfnew_out.groupby(['cohort','model','age'])['delta'].describe().to_csv(OUTPUTS+f'/age_acceleration_summary{filter_conditions}.csv')
dfnew_out.pivot_table(columns=['cohort','model'],index='age').to_csv(OUTPUTS+f'/age_acceleration_summary_cohort{filter_conditions}.csv')
dfnew_out.pivot_table(columns=['cohort','model'],index='age',aggfunc='median').to_csv(OUTPUTS+f'/age_acceleration_summary_median_cohort{filter_conditions}.csv')

#### children over time - children only
if children_cohort:
    cohort_sort = ['8yo','10yo','12yo']
    dfnew_out['age'] = dfnew_out['age'].str.replace('_age','')

    alt.Chart(dfnew_out).mark_line(size=.5).encode(x=alt.X('cohort',sort=cohort_sort,title=None),
                                            y=alt.Y('delta',title=None),
                                            detail='patient').properties(height=80,width=80).facet(
                                            column=alt.Column('age',title=None),
                                            row=alt.Row('model',title='Age Acceleration')).save(f'{OUTPUTS}/children_timeline_all_patients.html')


    alt.Chart(dfnew_out).mark_line(size=1).encode(x=alt.X('cohort',sort=cohort_sort,title=None),
                                            y=alt.Y('delta',title=None),
                                            color='model').properties(height=80,width=80).facet(
                                            column=alt.Column('age',title=None),
                                            row=alt.Row('patient',title='Age Acceleration')).save(f'{OUTPUTS}/children_timeline_all_patients2.html')


    # mean ± std per cohort
    dfnew_out_agg = dfnew_out.groupby(['age','cohort','model'])['delta'].agg(['mean','std']).reset_index()
    dfnew_out_agg['lo'] = dfnew_out_agg['mean'] - dfnew_out_agg['std']
    dfnew_out_agg['hi'] = dfnew_out_agg['mean'] + dfnew_out_agg['std']

    base = alt.Chart().encode(x=alt.X('cohort:N',sort=cohort_sort,title=None))
    lines = base.mark_line(size=1).encode(y=alt.Y('mean:Q',title=None))
    dots = base.mark_circle(size=20,opacity=1).encode(y='mean:Q')
    errs = base.mark_errorbar(ticks=True).encode(y='lo:Q',y2='hi:Q')
    alt.layer(errs,lines,dots,data=dfnew_out_agg).properties(height=80,width=80).facet(
                        column=alt.Column('age:N',title=None),
                        row=alt.Row('model:N',title='Age Acceleration')).save(f'{OUTPUTS}/children_timeline_pooled.html')



df_hepatic = dfnew_out.query('age == "hepatic"')
alt.Chart(df_hepatic).mark_circle().encode(x=alt.X('delta').title(None),
                                           color=alt.Color('cohort',
                                                           scale=alt.Scale(domain=['8yo','10yo', '12yo'],
                                                           range=['#00C4CC','#FFC300','#FF5733'])),
                                           y='patient:N',
                                           column=alt.Column('model').title('Hepatic Age Acceleration')).properties(height=200,width=100).save(f'{OUTPUTS}/children_hepatic_age_acceleration.html')

# example children 3 and 6
df_reasoning['cohort'] = df_reasoning['cohort'].str.replace('yo','').astype(int)
df_reas_aux = df_reasoning.query("model == 'Gemini Flash Lite' and patient in [3,6]").sort_values(by=['patient','cohort']).drop(columns=['model']).reset_index(drop=True)
df_reas_aux.to_csv(f'{OUTPUTS}/children_reasoning_examples.csv', index=False)




### METRIC 3: Inter-Model Correlation Heatmap
# This visualizes which models are "twins" and which is the outlier.
# ATTENTION: if this is not empty, it will filter just the ones indicated
# filter_categories = ['women_post_natal', 'women_pre_natal']
filter_categories = ['8yo', '10yo', '12yo']
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
    plt.savefig(f'{OUTPUTS}/correlation_matrix_{age}.svg')
    corr_means[age] = correlation_matrix.mean().round(2)
corr_means.to_csv(f'{OUTPUTS}/correlation_means_women{filter_conditions}.csv')
a = 0