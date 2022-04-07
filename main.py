######### Machinery #######
## if we need to recalculate
import streamlit as st
# st.set_page_config(layout='wide')
from io import BytesIO

import pandas as pd
import numpy as np

from itertools import product

import altair as alt

import pickle

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='arviz')
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['xtick.major.size'] = 6
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True

# min reference date
truncation_date = pd.to_datetime('2021-08-25', format='%Y-%m-%d')

st.title("Analysis of SARS-CoV-2 spread in Taiwan, April-August 2021")
st.markdown("*Supporting materials* for Akhmetzhanov AR, Cheng H, Linton NM, Ponce L, Jian S, Lin L. Transmission dynamics and effectiveness of control measures during a surge of COVID-19 cases in Taiwan, 2021.")
st.markdown("<font color='red'>(*Work in progress*)</font>", unsafe_allow_html=True)

##########################

st.header("")
counties_to_show = ['New Taipei City','Taipei City', 'Keelung City']
## loading the processed data
df = pd.read_csv(truncation_date.strftime("%Y%m%d")+"-data.csv")
for col in ['Onset', 'Confirm']:
    df[col] = pd.to_datetime(df[col])
df['County'] = [x if x in ['Keelung City','New Taipei City','Taipei City'] else 'Others' for x in df.ResidCounty_eng]
mindate = df.Confirm.values[0] - pd.DateOffset(days=int(df.ConfirmDay.values[0]))
truncation_day = (truncation_date - mindate).days
counties_to_show = np.r_[counties_to_show, ['Others']]

## generating data frame with case counts
for idx, county_ in enumerate(counties_to_show):
    df_ = df.loc[lambda d: (d.County==county_)&(~pd.isnull(d.Onset)), 'OnsetDay'].value_counts(sort=False)\
        .reindex(np.arange(1,truncation_day+1), fill_value=0).reset_index().rename(columns = {'index':'t', 'OnsetDay':'n'})\
        .merge(df.loc[lambda d: (d.County==county_)&(pd.isnull(d.Onset)), 'ConfirmDay'].value_counts(sort=False).reindex(np.arange(1,truncation_day+1), fill_value=0)\
           .reset_index().rename(columns = {'index':'t', 'ConfirmDay':'n_unobs'}))
    df_['County'] = county_
    df_cases = df_ if idx==0 else df_cases.append(df_)
df_cases['d'] = [mindate + pd.DateOffset(days=t) for t in df_cases['t']]

##---
clrs_ = ["#d6604d", "#fee391", "#4393c3", "lightgrey"]

df_ = df_cases.copy()
df_['County_order'] = df_['County'].replace(
    {val: i for i, val in enumerate(counties_to_show)}
)

datemin_plt = pd.to_datetime("2021-04-29")
datemax_plt = truncation_date

chart = alt.Chart(df_).mark_bar(clip=True,width=3).encode(
    x = alt.X('d', axis=alt.Axis(title="Reporting date in 2021"), scale=alt.Scale(domain=[datemin_plt, datemax_plt])),
    y = alt.Y('n', title="Confirmed cases"),
    color = alt.Color('County', scale=alt.Scale(range=clrs_), sort=alt.EncodingSortField('County_order', order='ascending')),
    order = 'County_order'
)
st.altair_chart(chart.configure_view(strokeWidth=0).configure_axis(grid=False, domain=False).properties(width=600, height=350))

with st.expander("Descriptive statistics"):
    for county in counties_to_show[:-1]:
        st.markdown("#### " + county + 
            "\n* Number of cases: %d (%.1f%% of all cases in Taiwan)" % (df.loc[lambda d: d.ResidCounty_eng==county].shape[0], 
                round(df.loc[lambda d: d.ResidCounty_eng==county].shape[0] / df.shape[0] * 100, 1)) + 
            "\n* By age groups\n" +
            "  * <17 y.o.: %d (%.1f%%)" % (df.loc[lambda d: (d.ResidCounty_eng==county)&(d.Age<17)].shape[0],
                round(df.loc[lambda d: (d.ResidCounty_eng==county)&(d.Age<17)].shape[0] / df.loc[lambda d: (d.ResidCounty_eng==county)].shape[0] * 100, 1)) + "\n" + 
            "  * 17-34 y.o.: %d (%.1f%%)" % (df.loc[lambda d: (d.ResidCounty_eng==county)&(d.Age>=17)&(d.Age<35)].shape[0],
                round(df.loc[lambda d: (d.ResidCounty_eng==county)&(d.Age>=17)&(d.Age<35)].shape[0] / df.loc[lambda d: (d.ResidCounty_eng==county)].shape[0] * 100, 1)) + "\n" +
            "  * 35-64 y.o.: %d (%.1f%%)" % (df.loc[lambda d: (d.ResidCounty_eng==county)&(d.Age>=35)&(d.Age<65)].shape[0],
                round(df.loc[lambda d: (d.ResidCounty_eng==county)&(d.Age>=35)&(d.Age<65)].shape[0] / df.loc[lambda d: (d.ResidCounty_eng==county)].shape[0] * 100, 1)) + "\n" + 
            "  * 65+ y.o.: %d (%.1f%%)" % (df.loc[lambda d: (d.ResidCounty_eng==county)&(d.Age>=35)&(d.Age<65)].shape[0],
                round(df.loc[lambda d: (d.ResidCounty_eng==county)&(d.Age>=65)].shape[0] / df.loc[lambda d: (d.ResidCounty_eng==county)].shape[0] * 100, 1)) + "\n" +
            "* By sex\n" +
            "  * Women %d (%.1f%%)" % (df.loc[lambda d: (d.ResidCounty_eng==county)&(d.Gender=='女')].shape[0],
                round(df.loc[lambda d: (d.ResidCounty_eng==county)&(d.Gender=='女')].shape[0] / df.loc[lambda d: (d.ResidCounty_eng==county)].shape[0] * 100, 1)) + "\n" + 
            "  * Men: %d (%.1f%%)" % (df.loc[lambda d: (d.ResidCounty_eng==county)&(d.Gender=='男')].shape[0],
                round(df.loc[lambda d: (d.ResidCounty_eng==county)&(d.Gender=='男')].shape[0] / df.loc[lambda d: (d.ResidCounty_eng==county)].shape[0] * 100, 1)) + "\n" +
            "* Among them %d cases were known to be asymptomatic" % df.loc[lambda d: (d.ResidCounty_eng==county)&(d.Asymptomatic=='Yes')].shape[0])


st.header("A. Reporting delay")

st.subheader("By symptom onset date")
st.text("(so-called forward-looking interval)")
st.markdown("* Data")
## creating a dataframe for heatmap of the reporting delay
df_ = df.loc[lambda d: ~pd.isnull(d.Onset), ['ID', 'Onset','Confirm']].copy()
df_['Δ'] = (df_.Confirm - df_.Onset).dt.days
Δmin = -1
Δmax = 21
df_['Δ'] = [Δ if Δ<Δmax else Δmax for Δ in df_.Δ]
df_['Δ'] = [Δ if Δ>Δmin else Δmin for Δ in df_.Δ]
df_['Δ_str'] = ["%d-"%Δmin if Δ<=Δmin else ("%d+"%Δmax if Δ>=Δmax else str(Δ)) for Δ in df_.Δ]
df_['Δ'] = [Δmin if Δ<=Δmin else (Δmax if Δ>=Δmax else Δ) for Δ in df_.Δ]
df_ = df_.loc[:, ['Onset','Δ']].groupby(['Onset','Δ']).value_counts(sort=False).reset_index().rename(columns={0:'n'})\
    .merge(pd.DataFrame(product(pd.date_range(start=mindate, end=truncation_date), np.arange(Δmin, Δmax+1, 1)), columns = ['Onset', 'Δ']),  how='outer')\
    .sort_values(['Onset','Δ'])#.fillna(0)
df_['OnsetDay'] = (df_.Onset - mindate).dt.days
df_['Onset_str'] = df_.Onset.dt.strftime("%m/%d")
df_['Onset_str_ticks'] = [Onset_str if x % 7 == 1 else "" for x, Onset_str in zip(df_.OnsetDay, df_.Onset_str)]

heatmap = alt.Chart(df_).mark_rect(clip=True).encode(
    x=alt.X('Onset_str:O', title="Symptom onset date", axis=alt.Axis(values=list(df_['Onset_str_ticks']), labelAngle=-35), sort=list(df_.Onset_str.drop_duplicates())),
    y=alt.Y('Δ:O', scale=alt.Scale(reverse=True), title="Delay"),
    color=alt.Color('n', scale=alt.Scale(scheme='Spectral', reverse=True))
).properties(width=600, height=320)

nulls = heatmap.transform_filter(
  "!isValid(datum.n)"
).mark_rect(opacity=0).encode(
  alt.Color('n:N', legend=None)
)

df_line = df_.loc[:,['Onset', 'Onset_str', 'Onset_str_ticks']].drop_duplicates()
df_line['y'] = (truncation_date - df_line.Onset).dt.days + 1
line = alt.Chart(df_line.loc[lambda d: (d.y<=Δmax)&(d.y>=Δmin)])\
    .mark_line(strokeDash=[6, 6], size=1, color='lightgrey').encode(
        x = alt.X('Onset_str:O', sort=list(df_.Onset_str.drop_duplicates())), y='y:O')

chart = heatmap + nulls + line
st.altair_chart(chart)

st.subheader("By reporting date")
st.text("(so-called backward-looking interval)")
st.markdown("* Data")

## creating a dataframe for heatmap of the reporting delay
df_ = df.loc[lambda d: ~pd.isnull(d.Onset), ['ID', 'Onset','Confirm']].copy()
df_['Δ'] = (df_.Confirm - df_.Onset).dt.days
df_['Δ'] = [Δ if Δ<Δmax else Δmax for Δ in df_.Δ]
df_['Δ'] = [Δ if Δ>Δmin else Δmin for Δ in df_.Δ]
df_['Δ_str'] = ["%d-"%Δmin if Δ<=Δmin else ("%d+"%Δmax if Δ>=Δmax else str(Δ)) for Δ in df_.Δ]
df_['Δ'] = [Δmin if Δ<=Δmin else (Δmax if Δ>=Δmax else Δ) for Δ in df_.Δ]
df_ = df_.loc[:, ['Confirm','Δ']].groupby(['Confirm','Δ']).value_counts(sort=False).reset_index().rename(columns={0:'n'})\
    .merge(pd.DataFrame(product(pd.date_range(start=mindate, end=truncation_date), np.arange(Δmin, Δmax+1, 1)), columns = ['Confirm', 'Δ']),  how='outer')\
    .sort_values(['Confirm','Δ'])#.fillna(0)
df_['ConfirmDay'] = (df_.Confirm - mindate).dt.days
df_['Confirm_str'] = df_.Confirm.dt.strftime("%m/%d")
df_['Confirm_str_ticks'] = [Confirm_str if x % 7 == 1 else "" for x, Confirm_str in zip(df_.ConfirmDay, df_.Confirm_str)]

heatmap = alt.Chart(df_).mark_rect(clip=True).encode(
    x=alt.X('Confirm_str:O', title="Reporting date", axis=alt.Axis(values=list(df_['Confirm_str_ticks']), labelAngle=-35), sort=list(df_.Confirm_str.drop_duplicates())),
    y=alt.Y('Δ:O', scale=alt.Scale(reverse=True), title="Delay"),
    color=alt.Color('n', scale=alt.Scale(scheme='Spectral', reverse=True))
).properties(width=600, height=320)

nulls = heatmap.transform_filter(
  "!isValid(datum.n)"
).mark_rect(opacity=0).encode(
  alt.Color('n:N', legend=None)
)

chart = heatmap + nulls 
st.altair_chart(chart)

st.header("B. Effective reproduction number, $R_t$")


st.subheader("B.1. $R_t$ by date of symptom onset")

st.subheader("B.2. $R_t$ by date of infection")

