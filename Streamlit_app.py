import streamlit as st
import pandas as pd
import io
import plotly.express as px
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
import numpy as np
import time

st.markdown("<h1 style='text-align: center; color: white;'>Data Visualization Using StreamLit</h1>", unsafe_allow_html=True)
st.markdown("<div id='linkto_top'></div>", unsafe_allow_html=True)



def datf_inf(df):
    df.columns = df.columns.str.replace(' ', '_')
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()

    df_info = s.split('\n')

    counts = []
    names = []
    nn_count = []
    dtype = []
    for i in range(5, len(df_info) - 3):
        line = df_info[i].split()
        counts.append(line[0])
        names.append(line[1])
        nn_count.append(line[2])
        dtype.append(line[4])

    df_info_dataframe = pd.DataFrame(
        data={'#': counts, 'Column': names, 'Non-Null Count': nn_count, 'Data Type': dtype})
    return df_info_dataframe.drop('#', axis=1)

def sidebar_multiselect_container(massage, arr, key):
    container = st.sidebar.container()
    select_all_button = st.sidebar.checkbox("Select all for " + key + " plots")
    if select_all_button:
        selected_num_cols = container.multiselect(massage, arr, default=list(arr))
    else:
        selected_num_cols = container.multiselect(massage, arr, default=arr[0])

    return selected_num_cols
def datf_nval(df):
    res = pd.DataFrame(df.isnull().sum()).reset_index()
    res['Percentage'] = round(res[0] / df.shape[0] * 100, 2)
    res['Percentage'] = res['Percentage'].astype(str) + '%'
    return res.rename(columns={'index': 'Column', 0: 'Number of null values'})



file_format = st.radio('Select file format:', ('csv', 'excel'), key='file_format')
dataset = st.file_uploader(label='')

if dataset:
    if file_format == 'csv':
        df = pd.read_csv(dataset)
    else:
        df = pd.read_excel(dataset)

    st.subheader('Dataframe:')
    n, m = df.shape
    st.write(f'<p style="font-size:130%">Dataset contains {n} rows and {m} columns.</p>', unsafe_allow_html=True)
    st.dataframe(df)

    df_info = ['Info', 'Null Info', 'Box Plots', 'Descriptive Analysis', 'Automated EDA']


    sdbar = st.sidebar.multiselect("EDA Options: ", df_info)

    if 'Info' in sdbar:
        st.subheader('Info:')
        c1, c2, c3 = st.columns([1, 2, 1])
        c2.dataframe(datf_inf(df))

    if 'Null Info' in sdbar:
        st.subheader('NA Value Information:')
        if df.isnull().sum().sum() == 0:
            st.write('There is not any NA value in your dataset.')
        else:
            c1, c2, c3 = st.columns([0.5, 2, 0.5])
            c2.dataframe(datf_nval(df), width=1500)
            st.markdown('')

    if 'Descriptive Analysis' in sdbar:
        st.subheader('Descriptive Analysis:')
        st.dataframe(df.describe())
    num_columns = df.select_dtypes(exclude='object').columns
    if 'Box Plots' in sdbar:
        if len(num_columns) == 0:
            st.write('There is no numerical columns in the data.')
        else:
            selected_num_cols = sidebar_multiselect_container('Choose columns for Box plots:', num_columns,
                                                                        'Box')
            st.subheader('Box plots')
            i = 0
            while (i < len(selected_num_cols)):
                c1, c2 = st.columns(2)
                for j in [c1, c2]:

                    if (i >= len(selected_num_cols)):
                        break

                    fig = px.box(df, y=selected_num_cols[i])
                    j.plotly_chart(fig, use_container_width=True)
                    i += 1

    if 'Automated EDA' in sdbar:
        datf = df
        st.write("Please Wait for Few Seconds.....")

        pr = df.profile_report(dark_mode=True)

        progress_bar = st.progress(0)
        status_text = st.empty()
        chart = st.line_chart(np.random.randn(10, 2))


        st_profile_report(pr)
        st.balloons()
st.markdown("<div id='linkto_top'></div>", unsafe_allow_html=True)


pages = ["Page 1","Page 2"]
section = st.sidebar.radio('', pages)

if section == "Page 1":                  # This is the beginning of my first page

    # add the link at the bottom of each page
    st.markdown("<a href='#linkto_top'>Link to top</a>", unsafe_allow_html=True)