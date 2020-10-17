## COVID-19 Virus Analysis and Diagnosis

### COVID-19 (Sars Cov 2) virus spread analysis using Python and diagnosis from radiography using machine learning model

Author: [Bui Quang Bao](https://buiquangbao.github.io/)

⚠ This research is intended for **educational purposes only**. Please do not use the results without real clinical researches!

![png](assets/diag-combine.gif)

### Table of Contents

1. [Introduction](#Introduction)
   - [COVID-19 virus pandemic](#COVID-19-virus-pandemic)
   - [Analysis and Diagnosis of COVID-19](#Analysis-and-Diagnosis-of-COVID-19)
2. [Dataset](#Dataset)
   - [Data resources](#Data-Resources)
   - [Data processing and classification](#Data-processing-and-classification)
3. [Method](#Method)
   - [Overall Approach](#Overall-Approach)
   - [Detailed Design](#Detailed-Design)
4. [Implementation: COVID-19 virus spread analysis](#COVID-19-virus-spread-analysis)
   - [Data Processing: Confirmed cases](#Confirmed-cases)
   - [Data Processing: Deaths by COVID-19](#Deaths-by-COVID-19)
   - [Data Processing: Recovered](#Recovered)
   - [Data Processing: GDP per capita](#GDP-per-capita)
   - [Join all dataset into one](#Join-all-dataset-into-one)
   - [Calculate Correlation Matrix](#Calculate-Correlation-Matrix)
   - [Visualization from data](#Visualization-from-data)
5. [Implementation: COVID-19 virus diagnosis from radiography](#COVID-19-virus-diagnosis-from-radiography)
   - [Samples Gathering](#Samples-Gathering)
   - [Model Training](#Model-Training)
   - [Model Embedding](#Model-Embedding)
   - [Testing](#Testing)
7. [Results](#Results)
   - [Analysis Conclusion](#Analysis-Conclusion)
   - [Diagnosis Accuracy](#Diagnosis-Accuracy)
   - [Strengths of the method](#Strengths-of-the-method)
   - [Limitations of the method](#Limitations-of-the-method)
   - [Important Notice](#Important-Notice)
8. [References](#References)

## Introduction
#### COVID-19 virus pandemic
Coronavirus disease (COVID-19) is an infectious global disease. The virus was identified in Wuhan, China in December 2019. COVID-19 disease is caused by the virus severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), a new virus in humans causing respiratory illness which can be spread from person-to-person.

The World Health Organization (WHO) declared the COVID-19 outbreak a Public Health Emergency of International Concern (PHEIC) on 30 January 2020 and a pandemic on 11 March 2020.
#### Analysis and Diagnosis of COVID-19
Analysis, or the process of turning raw data into insights, can provide us with a view of how the virus influences the communities. Those insights can indicate if the development of a country affects its virus spread as well as the relationship between the infection rate, the death rate and the recovered rate of COVID-19 pandemic in some specific countries and on the global.

The quick diagnosis method is significant in preventing the spread of the Covid-19 pandemic. With the dataset of approximately 1000 images of chest X-ray collected, we can build a machine learning model to predict whether a new X-ray image is from a COVID-19 infected patient or a non-infected person.

<!-- Dataset -->
## Dataset
#### Data Resources
- [Novel Coronavirus (COVID-19) Cases Data](https://data.humdata.org/dataset/novel-coronavirus-2019-ncov-cases) compiled by the Johns Hopkins University Center for Systems Science and Engineering ([JHU CCSE](https://data.humdata.org/organization/jhucsse))
  - time_series_covid19_confirmed_global.csv
  - time_series_covid19_deaths_global.csv
  - time_series_covid19_recovered_global.csv
- [GDP (Gross Domestic Product) per capita](https://ourworldindata.org/grapher/gdp-per-capita-worldbank) published by World Bank – World Development Indicators
  - gdp-per-capita-worldbank.csv
- [X-ray and CT images of COVID-19 patients dataset](https://github.com/ieee8023/covid-chestxray-dataset) on Github by Joseph Paul Cohen, Postdoctoral Fellow, Mila, University of Montreal
  - [Approximately 1000 X-ray and CT images](https://github.com/ieee8023/covid-chestxray-dataset/tree/master/images) for traing and testing
#### Data processing and classification
```
📁dataset
├── 📁dataset_analysis
│   ├── time_series_covid19_confirmed_global.csv
│   ├── time_series_covid19_deaths_global.csv
│   ├── time_series_covid19_recovered_global.csv
│   └── gdp-per-capita-worldbank.csv
└── 📁dataset_radiography
    ├── 📁data_for_training
    │   ├── 📁covid_infected
    │   │   └── More than 800 JPEG images
    │   └── 📁normal
    │       └── More than 800 JPEG images
    └── 📁data_for_testing
        ├── 📁covid_infected
        │   └── 100 JPEG images
        └── 📁normal
            └── 100 JPEG images
```

## Method
### Overall Approach
#### Python
Python is an interpreted, object-oriented, high-level programming language with dynamic semantics. Its high-level built-in data structures, combined with dynamic typing and dynamic binding, make it very attractive for Rapid Application Development. Python is commonly used to streamline large complex data sets.

Python has a long list of powerful libraries available for all users. That’s a key factor that gives a strong push for Python at all, and in the data science, too. Libraries such as [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/) are very powerful for data manipulation and libraries such as [MatplotLib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/) provide great visualization.
#### Machine Learning
Machine learning is an application of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves.

The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide. The primary aim is to allow the computers to learn automatically without human intervention or assistance and adjust actions accordingly.

A machine learning model is a file that has been trained to recognize certain types of patterns. You train a model over a set of data, providing it an algorithm that it can use to reason over and learn from those data. Once you have trained the model, you can use it to reason over data that it hasn't seen before, and make predictions about those data.
#### TensorFlow
[TensorFlow](https://www.tensorflow.org/) is an end-to-end open-source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries and community resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML-powered applications.

[TensorFlow.js](https://www.tensorflow.org/js) is a library for machine learning in JavaScript. Develop ML models in JavaScript, and use ML directly in the browser.
### Detailed Design
#### Jupyter Notebook
The Jupyter Notebook is an open-source web application that allows us to create and share documents that contain live code, equations, visualizations and narrative text. Uses include: data cleaning and transformation, numerical simulation, statistical modeling, data visualization, machine learning, and much more. 

The Jupyter Notebook is used to support interactive data science and scientific computing across many programming languages, especially Python.
#### Python Modules
- [pandas](https://pandas.pydata.org/) (data analysis and manipulation)
- [numpy](https://numpy.org/) (scientific computing)
- [seaborn](https://seaborn.pydata.org/) (statistical data visualization)
- [matplotlib](https://matplotlib.org/) (visualizations)
#### Teachable Machine
Teachable Machine is a web-based tool that makes creating machine learning models fast, easy, and accessible to everyone. The models made with Teachable Machine are real [TensorFlow.js](https://www.tensorflow.org/js) models that work anywhere javascript runs, so they play nice with tools like Glitch, P5.js, Node.js & more.
#### ml5.js
[ml5.js](https://ml5js.org/) aims to make machine learning approachable for a broad audience of artists, creative coders, and students. The folks building TensorFlow.js figured out that it is possible to use the web browser's built-in graphics processing unit (GPU) to do calculations that would otherwise run very slowly using the central processing unit (CPU). The ml5.js library provides access to machine learning algorithms and models in the browser, building on top of TensorFlow.js with no other external dependencies. 
#### p5.js
Using the metaphor of a sketch, [p5.js](https://p5js.org/) has a full set of drawing functionality, consider the whole browser page as your sketch, including HTML5 objects for text, input, video, webcam, and sound.

## COVID-19 virus spread analysis
<!-- COVID-19 virus spread analysis -->
#### Note
- The latest datasets can be used (from the sources I mentioned above), as long as the files' structure hasn't been changed.
- Adjust the values in `.sort_values` method to the latest date in the datasets.
- Change the `name_country` in `.loc[country_name]` method to see the data of what country you want.
- `.head(1000)` lists all the rows in dataset. Change `1000` to the number of rows you want to list. `.head()` will list 5 rows by default.
- Use `dataset_name.shape` to see the number of rows and columns of the dataset.

#### Import modules

```python
!pip3 install seaborn
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
print('All modules are imported successfully!')
```

    Requirement already satisfied: seaborn in c:\users\admin\appdata\local\programs\python\python38-32\lib\site-packages (0.11.0)
    Requirement already satisfied: scipy>=1.0 in c:\users\admin\appdata\local\programs\python\python38-32\lib\site-packages (from seaborn) (1.5.2)
    Requirement already satisfied: pandas>=0.23 in c:\users\admin\appdata\local\programs\python\python38-32\lib\site-packages (from seaborn) (1.1.2)
    Requirement already satisfied: numpy>=1.15 in c:\users\admin\appdata\local\programs\python\python38-32\lib\site-packages (from seaborn) (1.19.2)
    Requirement already satisfied: matplotlib>=2.2 in c:\users\admin\appdata\local\programs\python\python38-32\lib\site-packages (from seaborn) (3.3.2)
    Requirement already satisfied: pytz>=2017.2 in c:\users\admin\appdata\local\programs\python\python38-32\lib\site-packages (from pandas>=0.23->seaborn) (2020.1)
    Requirement already satisfied: python-dateutil>=2.7.3 in c:\users\admin\appdata\local\programs\python\python38-32\lib\site-packages (from pandas>=0.23->seaborn) (2.8.1)
    Requirement already satisfied: cycler>=0.10 in c:\users\admin\appdata\local\programs\python\python38-32\lib\site-packages (from matplotlib>=2.2->seaborn) (0.10.0)
    Requirement already satisfied: pillow>=6.2.0 in c:\users\admin\appdata\local\programs\python\python38-32\lib\site-packages (from matplotlib>=2.2->seaborn) (7.2.0)
    Requirement already satisfied: certifi>=2020.06.20 in c:\users\admin\appdata\local\programs\python\python38-32\lib\site-packages (from matplotlib>=2.2->seaborn) (2020.6.20)
    Requirement already satisfied: kiwisolver>=1.0.1 in c:\users\admin\appdata\local\programs\python\python38-32\lib\site-packages (from matplotlib>=2.2->seaborn) (1.2.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.3 in c:\users\admin\appdata\local\programs\python\python38-32\lib\site-packages (from matplotlib>=2.2->seaborn) (2.4.7)
    Requirement already satisfied: six>=1.5 in c:\users\admin\appdata\roaming\python\python38\site-packages (from python-dateutil>=2.7.3->pandas>=0.23->seaborn) (1.15.0)
    All modules are imported successfully!
    

### Confirmed cases

#### Import file

Import `time_series_covid19_confirmed_global.csv`


```python
# Importing data
dataset_cases = pd.read_csv("time_series_covid19_confirmed_global.csv")

# Filter
# dataset = dataset[dataset.date.eq("2020-09-18")]

# Delete useless column
dataset_cases.drop(["Lat", "Long", "Province/State"],axis=1, inplace=True)

# Set index
# dataset.set_index("Country/Region", inplace=True)
dataset_cases = dataset_cases.groupby("Country/Region").sum()

dataset_cases.head(1000)
```


<!-- <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1/22/20</th>
      <th>1/23/20</th>
      <th>1/24/20</th>
      <th>1/25/20</th>
      <th>1/26/20</th>
      <th>1/27/20</th>
      <th>1/28/20</th>
      <th>1/29/20</th>
      <th>1/30/20</th>
      <th>1/31/20</th>
      <th>...</th>
      <th>9/9/20</th>
      <th>9/10/20</th>
      <th>9/11/20</th>
      <th>9/12/20</th>
      <th>9/13/20</th>
      <th>9/14/20</th>
      <th>9/15/20</th>
      <th>9/16/20</th>
      <th>9/17/20</th>
      <th>9/18/20</th>
    </tr>
    <tr>
      <th>Country/Region</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Afghanistan</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>38544</td>
      <td>38572</td>
      <td>38606</td>
      <td>38641</td>
      <td>38716</td>
      <td>38772</td>
      <td>38815</td>
      <td>38855</td>
      <td>38872</td>
      <td>38883</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>10704</td>
      <td>10860</td>
      <td>11021</td>
      <td>11185</td>
      <td>11353</td>
      <td>11520</td>
      <td>11672</td>
      <td>11816</td>
      <td>11948</td>
      <td>12073</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>47216</td>
      <td>47488</td>
      <td>47752</td>
      <td>48007</td>
      <td>48254</td>
      <td>48496</td>
      <td>48734</td>
      <td>48966</td>
      <td>49194</td>
      <td>49413</td>
    </tr>
    <tr>
      <th>Andorra</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1301</td>
      <td>1301</td>
      <td>1344</td>
      <td>1344</td>
      <td>1344</td>
      <td>1438</td>
      <td>1438</td>
      <td>1483</td>
      <td>1483</td>
      <td>1564</td>
    </tr>
    <tr>
      <th>Angola</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3092</td>
      <td>3217</td>
      <td>3279</td>
      <td>3335</td>
      <td>3388</td>
      <td>3439</td>
      <td>3569</td>
      <td>3675</td>
      <td>3789</td>
      <td>3848</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>West Bank and Gaza</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>27919</td>
      <td>28664</td>
      <td>29256</td>
      <td>29906</td>
      <td>30574</td>
      <td>31362</td>
      <td>32250</td>
      <td>33006</td>
      <td>33843</td>
      <td>34401</td>
    </tr>
    <tr>
      <th>Western Sahara</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>Yemen</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1999</td>
      <td>2003</td>
      <td>2007</td>
      <td>2009</td>
      <td>2011</td>
      <td>2013</td>
      <td>2016</td>
      <td>2019</td>
      <td>2022</td>
      <td>2024</td>
    </tr>
    <tr>
      <th>Zambia</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>13112</td>
      <td>13214</td>
      <td>13323</td>
      <td>13466</td>
      <td>13539</td>
      <td>13720</td>
      <td>13819</td>
      <td>13887</td>
      <td>13928</td>
      <td>14022</td>
    </tr>
    <tr>
      <th>Zimbabwe</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7429</td>
      <td>7453</td>
      <td>7479</td>
      <td>7508</td>
      <td>7526</td>
      <td>7531</td>
      <td>7576</td>
      <td>7598</td>
      <td>7633</td>
      <td>7647</td>
    </tr>
  </tbody>
</table>
<p>188 rows × 241 columns</p>
</div> -->


```python
# This dataset_cases_filtered will be used later
dataset_cases_filtered = pd.DataFrame(dataset_cases["9/18/20"])
# Rename
dataset_cases_filtered = dataset_cases_filtered.rename(columns={'9/18/20': 'confirmed_cases'})
```

#### Confirmed cases in specific countries


```python
dataset_cases.loc["Vietnam"].plot(figsize=(14, 6), legend = True, title = 'Confirmed cases in Vietnam, Singapore, Japan and China')
dataset_cases.loc["Singapore"].plot()
dataset_cases.loc["Japan"].plot()
dataset_cases.loc["China"].plot()
# Identify which line belong to which country
plt.legend()
```




    <matplotlib.legend.Legend at 0x25052178>




    
![png](assets/output_11_1.png)
    


*Confirmed cases in Vietnam, Singapore, Japan and China*

#### Countries with the highest confirmed cases

Sort the `dataset_cases` descending by the value in the newest column.


```python
# Sort the dataset descending
dataset_cases = dataset_cases.sort_values(by=['9/18/20'], ascending=False)
dataset_cases.head(1000)
```




<!-- <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1/22/20</th>
      <th>1/23/20</th>
      <th>1/24/20</th>
      <th>1/25/20</th>
      <th>1/26/20</th>
      <th>1/27/20</th>
      <th>1/28/20</th>
      <th>1/29/20</th>
      <th>1/30/20</th>
      <th>1/31/20</th>
      <th>...</th>
      <th>9/9/20</th>
      <th>9/10/20</th>
      <th>9/11/20</th>
      <th>9/12/20</th>
      <th>9/13/20</th>
      <th>9/14/20</th>
      <th>9/15/20</th>
      <th>9/16/20</th>
      <th>9/17/20</th>
      <th>9/18/20</th>
    </tr>
    <tr>
      <th>Country/Region</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>US</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>7</td>
      <td>...</td>
      <td>6360212</td>
      <td>6396100</td>
      <td>6443652</td>
      <td>6485123</td>
      <td>6520122</td>
      <td>6553652</td>
      <td>6592342</td>
      <td>6630051</td>
      <td>6674411</td>
      <td>6723933</td>
    </tr>
    <tr>
      <th>India</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>4465863</td>
      <td>4562414</td>
      <td>4659984</td>
      <td>4754356</td>
      <td>4846427</td>
      <td>4930236</td>
      <td>5020359</td>
      <td>5118253</td>
      <td>5214677</td>
      <td>5214677</td>
    </tr>
    <tr>
      <th>Brazil</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>4197889</td>
      <td>4238446</td>
      <td>4282164</td>
      <td>4315687</td>
      <td>4330455</td>
      <td>4345610</td>
      <td>4382263</td>
      <td>4419083</td>
      <td>4455386</td>
      <td>4495183</td>
    </tr>
    <tr>
      <th>Russia</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>1037526</td>
      <td>1042836</td>
      <td>1048257</td>
      <td>1053663</td>
      <td>1059024</td>
      <td>1064438</td>
      <td>1069873</td>
      <td>1075485</td>
      <td>1081152</td>
      <td>1086955</td>
    </tr>
    <tr>
      <th>Colombia</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>686851</td>
      <td>694664</td>
      <td>702088</td>
      <td>708964</td>
      <td>716319</td>
      <td>721892</td>
      <td>728590</td>
      <td>736377</td>
      <td>743945</td>
      <td>750471</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Laos</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>22</td>
      <td>22</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
    </tr>
    <tr>
      <th>Saint Kitts and Nevis</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
    </tr>
    <tr>
      <th>Holy See</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
    </tr>
    <tr>
      <th>Western Sahara</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>MS Zaandam</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
<p>188 rows × 241 columns</p>
</div> -->




```python
# Cast the index of dataframe (name of countries) into list
countries = list(dataset_cases.index)

# List 10 countries with the highest confirmed cases
count = 0
for country in countries:
    dataset_cases.loc[country].plot(figsize=(14, 6), legend = True, title = 'Countries with the highest confirmed cases')
    count += 1
    if count == 10:
        break

plt.legend()
```




    <matplotlib.legend.Legend at 0x25083f88>




    
![png](assets/output_16_1.png)
    


*Countries with the highest confirmed cases*

#### Countries with the lowest confirmed cases


```python
# Sort the dataset ascending
dataset_cases = dataset_cases.sort_values(by=['9/18/20'])
dataset_cases.head(1000)
```




<!-- <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1/22/20</th>
      <th>1/23/20</th>
      <th>1/24/20</th>
      <th>1/25/20</th>
      <th>1/26/20</th>
      <th>1/27/20</th>
      <th>1/28/20</th>
      <th>1/29/20</th>
      <th>1/30/20</th>
      <th>1/31/20</th>
      <th>...</th>
      <th>9/9/20</th>
      <th>9/10/20</th>
      <th>9/11/20</th>
      <th>9/12/20</th>
      <th>9/13/20</th>
      <th>9/14/20</th>
      <th>9/15/20</th>
      <th>9/16/20</th>
      <th>9/17/20</th>
      <th>9/18/20</th>
    </tr>
    <tr>
      <th>Country/Region</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MS Zaandam</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
    </tr>
    <tr>
      <th>Western Sahara</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>Holy See</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
    </tr>
    <tr>
      <th>Saint Kitts and Nevis</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
    </tr>
    <tr>
      <th>Laos</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>22</td>
      <td>22</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Colombia</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>686851</td>
      <td>694664</td>
      <td>702088</td>
      <td>708964</td>
      <td>716319</td>
      <td>721892</td>
      <td>728590</td>
      <td>736377</td>
      <td>743945</td>
      <td>750471</td>
    </tr>
    <tr>
      <th>Russia</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>1037526</td>
      <td>1042836</td>
      <td>1048257</td>
      <td>1053663</td>
      <td>1059024</td>
      <td>1064438</td>
      <td>1069873</td>
      <td>1075485</td>
      <td>1081152</td>
      <td>1086955</td>
    </tr>
    <tr>
      <th>Brazil</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>4197889</td>
      <td>4238446</td>
      <td>4282164</td>
      <td>4315687</td>
      <td>4330455</td>
      <td>4345610</td>
      <td>4382263</td>
      <td>4419083</td>
      <td>4455386</td>
      <td>4495183</td>
    </tr>
    <tr>
      <th>India</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>4465863</td>
      <td>4562414</td>
      <td>4659984</td>
      <td>4754356</td>
      <td>4846427</td>
      <td>4930236</td>
      <td>5020359</td>
      <td>5118253</td>
      <td>5214677</td>
      <td>5214677</td>
    </tr>
    <tr>
      <th>US</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>7</td>
      <td>...</td>
      <td>6360212</td>
      <td>6396100</td>
      <td>6443652</td>
      <td>6485123</td>
      <td>6520122</td>
      <td>6553652</td>
      <td>6592342</td>
      <td>6630051</td>
      <td>6674411</td>
      <td>6723933</td>
    </tr>
  </tbody>
</table>
<p>188 rows × 241 columns</p>
</div> -->




```python
# Cast the index of dataframe (name of countries) into list
countries = list(dataset_cases.index)

# List 10 countries with the lowest confirmed cases
count = 0
for country in countries:
    dataset_cases.loc[country].plot(figsize=(14, 6), legend = True, title = 'Countries with the lowest confirmed cases')
    count += 1
    if count == 10:
        break

plt.legend()
```




    <matplotlib.legend.Legend at 0x261a1670>




    
![png](assets/output_20_1.png)
    


*Countries with the lowest confirmed cases*

#### Look at the data in some specific countries about confirmed cases
Calculate the first derivative to see the spread rate

##### Infection rate in Vietnam


```python
# Calculate the first derivative of the curve
dataset_cases.loc["Vietnam"].diff().plot(figsize=(14, 6), legend = True, title = 'Infection rate in Vietnam')
```




    <AxesSubplot:title={'center':'Infection rate in Vietnam'}>




    
![png](assets/output_24_1.png)
    



```python
# Find the max_infection_rate for Vietnam
dataset_cases.loc["Vietnam"].diff().max()
```




    50.0



The `max_infection_rate` in Vietnam is **50.0**

##### Compare the Infection rate in some countries: Vietnam, United State, Italy, Japan and China


```python
# Calculate the first derivative of the curve

dataset_cases.loc["US"].diff().plot(figsize=(14, 6), legend = True, color = 'orange', title = 'Compare the Infection rate in some countries: Vietnam, United State, Italy, Japan and China')
dataset_cases.loc["Italy"].diff().plot(legend = True, color = 'purple')
dataset_cases.loc["China"].diff().plot(legend = True, color = 'brown')
dataset_cases.loc["Japan"].diff().plot(legend = True, color = 'green')
dataset_cases.loc["Vietnam"].diff().plot(legend = True)
```




    <AxesSubplot:title={'center':'Compare the Infection rate in some countries: Vietnam, United State, Italy, Japan and China'}>




    
![png](assets/output_28_1.png)
    



```python
# Find the max_infection_rate for US
dataset_cases.loc["US"].diff().max()
```




    77255.0



The `max_infection_rate` in US is **77255.0**

#### Calculate the `max_infection_rate` of all countries and add to `dataset_cases`


```python
# Cast the index of dataframe (name of countries) into list
countries = list(dataset_cases.index)

#Calculate max_infection_rate of all countries
max_infection_rate = []
for country in countries:
    max_infection_rate.append(dataset_cases.loc[country].diff().max())

#Add new column
dataset_cases["max_infection_rate"] = max_infection_rate

dataset_cases.head(1000)
```




<!-- <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1/22/20</th>
      <th>1/23/20</th>
      <th>1/24/20</th>
      <th>1/25/20</th>
      <th>1/26/20</th>
      <th>1/27/20</th>
      <th>1/28/20</th>
      <th>1/29/20</th>
      <th>1/30/20</th>
      <th>1/31/20</th>
      <th>...</th>
      <th>9/10/20</th>
      <th>9/11/20</th>
      <th>9/12/20</th>
      <th>9/13/20</th>
      <th>9/14/20</th>
      <th>9/15/20</th>
      <th>9/16/20</th>
      <th>9/17/20</th>
      <th>9/18/20</th>
      <th>max_infection_rate</th>
    </tr>
    <tr>
      <th>Country/Region</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MS Zaandam</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>Western Sahara</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>Holy See</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>Saint Kitts and Nevis</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Laos</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>22</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Colombia</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>694664</td>
      <td>702088</td>
      <td>708964</td>
      <td>716319</td>
      <td>721892</td>
      <td>728590</td>
      <td>736377</td>
      <td>743945</td>
      <td>750471</td>
      <td>15318.0</td>
    </tr>
    <tr>
      <th>Russia</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>1042836</td>
      <td>1048257</td>
      <td>1053663</td>
      <td>1059024</td>
      <td>1064438</td>
      <td>1069873</td>
      <td>1075485</td>
      <td>1081152</td>
      <td>1086955</td>
      <td>11656.0</td>
    </tr>
    <tr>
      <th>Brazil</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>4238446</td>
      <td>4282164</td>
      <td>4315687</td>
      <td>4330455</td>
      <td>4345610</td>
      <td>4382263</td>
      <td>4419083</td>
      <td>4455386</td>
      <td>4495183</td>
      <td>69074.0</td>
    </tr>
    <tr>
      <th>India</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>4562414</td>
      <td>4659984</td>
      <td>4754356</td>
      <td>4846427</td>
      <td>4930236</td>
      <td>5020359</td>
      <td>5118253</td>
      <td>5214677</td>
      <td>5214677</td>
      <td>97894.0</td>
    </tr>
    <tr>
      <th>US</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>7</td>
      <td>...</td>
      <td>6396100</td>
      <td>6443652</td>
      <td>6485123</td>
      <td>6520122</td>
      <td>6553652</td>
      <td>6592342</td>
      <td>6630051</td>
      <td>6674411</td>
      <td>6723933</td>
      <td>77255.0</td>
    </tr>
  </tbody>
</table>
<p>188 rows × 242 columns</p>
</div> -->




```python
# Keep needed column
dataset_cases = pd.DataFrame(dataset_cases["max_infection_rate"])

dataset_cases.head(1000)
```




<!-- <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>max_infection_rate</th>
    </tr>
    <tr>
      <th>Country/Region</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MS Zaandam</th>
      <td>7.0</td>
    </tr>
    <tr>
      <th>Western Sahara</th>
      <td>4.0</td>
    </tr>
    <tr>
      <th>Holy See</th>
      <td>3.0</td>
    </tr>
    <tr>
      <th>Saint Kitts and Nevis</th>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Laos</th>
      <td>3.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>Colombia</th>
      <td>15318.0</td>
    </tr>
    <tr>
      <th>Russia</th>
      <td>11656.0</td>
    </tr>
    <tr>
      <th>Brazil</th>
      <td>69074.0</td>
    </tr>
    <tr>
      <th>India</th>
      <td>97894.0</td>
    </tr>
    <tr>
      <th>US</th>
      <td>77255.0</td>
    </tr>
  </tbody>
</table>
<p>188 rows × 1 columns</p>
</div> -->



### Deaths by COVID-19

#### Import file

Import `time_series_covid19_deaths_global.csv`


```python
# Importing data
dataset_deaths = pd.read_csv("time_series_covid19_deaths_global.csv")

# Delete useless column
dataset_deaths.drop(["Lat", "Long", "Province/State"],axis=1, inplace=True)

# Set index
# dataset.set_index("Country/Region", inplace=True)
dataset_deaths = dataset_deaths.groupby("Country/Region").sum()

dataset_deaths.head(1000)
```




<!-- <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1/22/20</th>
      <th>1/23/20</th>
      <th>1/24/20</th>
      <th>1/25/20</th>
      <th>1/26/20</th>
      <th>1/27/20</th>
      <th>1/28/20</th>
      <th>1/29/20</th>
      <th>1/30/20</th>
      <th>1/31/20</th>
      <th>...</th>
      <th>9/9/20</th>
      <th>9/10/20</th>
      <th>9/11/20</th>
      <th>9/12/20</th>
      <th>9/13/20</th>
      <th>9/14/20</th>
      <th>9/15/20</th>
      <th>9/16/20</th>
      <th>9/17/20</th>
      <th>9/18/20</th>
    </tr>
    <tr>
      <th>Country/Region</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Afghanistan</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1420</td>
      <td>1420</td>
      <td>1420</td>
      <td>1420</td>
      <td>1420</td>
      <td>1425</td>
      <td>1426</td>
      <td>1436</td>
      <td>1436</td>
      <td>1437</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>322</td>
      <td>324</td>
      <td>327</td>
      <td>330</td>
      <td>334</td>
      <td>338</td>
      <td>340</td>
      <td>343</td>
      <td>347</td>
      <td>353</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1581</td>
      <td>1591</td>
      <td>1599</td>
      <td>1605</td>
      <td>1612</td>
      <td>1620</td>
      <td>1632</td>
      <td>1645</td>
      <td>1654</td>
      <td>1659</td>
    </tr>
    <tr>
      <th>Andorra</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>53</td>
      <td>53</td>
      <td>53</td>
      <td>53</td>
      <td>53</td>
      <td>53</td>
      <td>53</td>
      <td>53</td>
      <td>53</td>
      <td>53</td>
    </tr>
    <tr>
      <th>Angola</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>126</td>
      <td>130</td>
      <td>131</td>
      <td>132</td>
      <td>134</td>
      <td>136</td>
      <td>139</td>
      <td>143</td>
      <td>144</td>
      <td>147</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>West Bank and Gaza</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>192</td>
      <td>198</td>
      <td>204</td>
      <td>210</td>
      <td>221</td>
      <td>226</td>
      <td>229</td>
      <td>243</td>
      <td>244</td>
      <td>250</td>
    </tr>
    <tr>
      <th>Western Sahara</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Yemen</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>576</td>
      <td>580</td>
      <td>582</td>
      <td>582</td>
      <td>583</td>
      <td>583</td>
      <td>583</td>
      <td>583</td>
      <td>585</td>
      <td>585</td>
    </tr>
    <tr>
      <th>Zambia</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>300</td>
      <td>300</td>
      <td>306</td>
      <td>312</td>
      <td>312</td>
      <td>320</td>
      <td>324</td>
      <td>326</td>
      <td>326</td>
      <td>329</td>
    </tr>
    <tr>
      <th>Zimbabwe</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>222</td>
      <td>222</td>
      <td>224</td>
      <td>224</td>
      <td>224</td>
      <td>224</td>
      <td>224</td>
      <td>224</td>
      <td>224</td>
      <td>224</td>
    </tr>
  </tbody>
</table>
<p>188 rows × 241 columns</p>
</div> -->



Sort the `dataset_deaths` descending by the value in the newest column.


```python
dataset_deaths = dataset_deaths.sort_values(by=['9/18/20'], ascending=False)
dataset_deaths.head(1000)
```




<!-- <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1/22/20</th>
      <th>1/23/20</th>
      <th>1/24/20</th>
      <th>1/25/20</th>
      <th>1/26/20</th>
      <th>1/27/20</th>
      <th>1/28/20</th>
      <th>1/29/20</th>
      <th>1/30/20</th>
      <th>1/31/20</th>
      <th>...</th>
      <th>9/9/20</th>
      <th>9/10/20</th>
      <th>9/11/20</th>
      <th>9/12/20</th>
      <th>9/13/20</th>
      <th>9/14/20</th>
      <th>9/15/20</th>
      <th>9/16/20</th>
      <th>9/17/20</th>
      <th>9/18/20</th>
    </tr>
    <tr>
      <th>Country/Region</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>US</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>190859</td>
      <td>191766</td>
      <td>192979</td>
      <td>193693</td>
      <td>194071</td>
      <td>194493</td>
      <td>195781</td>
      <td>196763</td>
      <td>197633</td>
      <td>198570</td>
    </tr>
    <tr>
      <th>Brazil</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>128539</td>
      <td>129522</td>
      <td>130396</td>
      <td>131210</td>
      <td>131625</td>
      <td>132006</td>
      <td>133119</td>
      <td>134106</td>
      <td>134935</td>
      <td>135793</td>
    </tr>
    <tr>
      <th>India</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>75062</td>
      <td>76271</td>
      <td>77472</td>
      <td>78586</td>
      <td>79722</td>
      <td>80776</td>
      <td>82066</td>
      <td>83198</td>
      <td>84372</td>
      <td>84372</td>
    </tr>
    <tr>
      <th>Mexico</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>69049</td>
      <td>69649</td>
      <td>70183</td>
      <td>70604</td>
      <td>70821</td>
      <td>71049</td>
      <td>71678</td>
      <td>71978</td>
      <td>72179</td>
      <td>72803</td>
    </tr>
    <tr>
      <th>United Kingdom</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>41683</td>
      <td>41697</td>
      <td>41703</td>
      <td>41712</td>
      <td>41717</td>
      <td>41726</td>
      <td>41753</td>
      <td>41773</td>
      <td>41794</td>
      <td>41821</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Cambodia</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Holy See</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Saint Kitts and Nevis</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Timor-Leste</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Dominica</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>188 rows × 241 columns</p>
</div> -->



#### Deaths by COVID-19 in specific countries


```python
dataset_deaths.loc["Vietnam"].plot(figsize=(14, 6), legend = True, title = 'Deaths by COVID-19 in Vietnam, Singapore, Japan and China')
dataset_deaths.loc["Singapore"].plot()
dataset_deaths.loc["Japan"].plot()
dataset_deaths.loc["China"].plot()
# Identify which line belong to which country
plt.legend()
```




    <matplotlib.legend.Legend at 0x26246f40>




    
![png](assets/output_39_1.png)
    


*Deaths by COVID-19 in Vietnam, Singapore, Japan and China*

#### Countries with the highest deaths by COVID-19


```python
# Cast the index of dataframe (name of countries) into list
countries = list(dataset_deaths.index)

# List 10 countries with the highest deaths by COVID-19
count = 0
for country in countries:
    dataset_deaths.loc[country].plot(figsize=(14, 6), legend = True, title = 'Countries with the highest deaths by COVID-19')
    count += 1
    if count == 10:
        break

plt.legend()
```




    <matplotlib.legend.Legend at 0x2624dc88>




    
![png](assets/output_42_1.png)
    


*Countries with the highest deaths by COVID-19*

#### Look at data in some specific countries about deaths by COVID-19

##### Death by COVID-19 rate in Vietnam


```python
# Calculate the first derivative of the curve
dataset_deaths.loc["Vietnam"].diff().plot(figsize=(14, 6), legend = True, title = 'Death by COVID-19 rate in Vietnam')
```




    <AxesSubplot:title={'center':'Death by COVID-19 rate in Vietnam'}>




    
![png](assets/output_46_1.png)
    



```python
dataset_deaths.loc["Vietnam"].diff().max()
```




    3.0



The `max_death_rate` in Vietnam is **3.0**

##### Compare the deaths rate in some countries: Vietnam, United State, Italy, Japan and China


```python
# Calculate the first derivative of the curve
dataset_deaths.loc["US"].diff().plot(figsize=(14, 6), legend = True, color = 'orange', title = 'Compare the deaths rate in some countries: Vietnam, United State, Italy, Japan and China')
dataset_deaths.loc["Italy"].diff().plot(legend = True, color = 'purple')
dataset_deaths.loc["China"].diff().plot(legend = True, color = 'brown')
dataset_deaths.loc["Japan"].diff().plot(legend = True, color = 'green')
dataset_deaths.loc["Vietnam"].diff().plot(legend = True)
```




    <AxesSubplot:title={'center':'Compare the deaths rate in some countries: Vietnam, United State, Italy, Japan and China'}>




    
![png](assets/output_50_1.png)
    



```python
dataset_deaths.loc["US"].diff().max()
```




    2609.0



The `max_death_rate` in US is **2609.0**

#### Calculate the `max_death_rate` of all countries and add to `dataset_deaths`


```python
# Cast the index of dataframe (name of countries) into list
countries = list(dataset_deaths.index)

# Calculate the max_death_rate of all countries
max_death_rate = []
for country in countries:
    max_death_rate.append(dataset_deaths.loc[country].diff().max())

#Add new column
dataset_deaths["max_death_rate"] = max_death_rate

dataset_deaths.head(1000)
```




<!-- <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1/22/20</th>
      <th>1/23/20</th>
      <th>1/24/20</th>
      <th>1/25/20</th>
      <th>1/26/20</th>
      <th>1/27/20</th>
      <th>1/28/20</th>
      <th>1/29/20</th>
      <th>1/30/20</th>
      <th>1/31/20</th>
      <th>...</th>
      <th>9/10/20</th>
      <th>9/11/20</th>
      <th>9/12/20</th>
      <th>9/13/20</th>
      <th>9/14/20</th>
      <th>9/15/20</th>
      <th>9/16/20</th>
      <th>9/17/20</th>
      <th>9/18/20</th>
      <th>max_death_rate</th>
    </tr>
    <tr>
      <th>Country/Region</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>US</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>191766</td>
      <td>192979</td>
      <td>193693</td>
      <td>194071</td>
      <td>194493</td>
      <td>195781</td>
      <td>196763</td>
      <td>197633</td>
      <td>198570</td>
      <td>2609.0</td>
    </tr>
    <tr>
      <th>Brazil</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>129522</td>
      <td>130396</td>
      <td>131210</td>
      <td>131625</td>
      <td>132006</td>
      <td>133119</td>
      <td>134106</td>
      <td>134935</td>
      <td>135793</td>
      <td>1595.0</td>
    </tr>
    <tr>
      <th>India</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>76271</td>
      <td>77472</td>
      <td>78586</td>
      <td>79722</td>
      <td>80776</td>
      <td>82066</td>
      <td>83198</td>
      <td>84372</td>
      <td>84372</td>
      <td>2003.0</td>
    </tr>
    <tr>
      <th>Mexico</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>69649</td>
      <td>70183</td>
      <td>70604</td>
      <td>70821</td>
      <td>71049</td>
      <td>71678</td>
      <td>71978</td>
      <td>72179</td>
      <td>72803</td>
      <td>1092.0</td>
    </tr>
    <tr>
      <th>United Kingdom</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>41697</td>
      <td>41703</td>
      <td>41712</td>
      <td>41717</td>
      <td>41726</td>
      <td>41753</td>
      <td>41773</td>
      <td>41794</td>
      <td>41821</td>
      <td>1224.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Cambodia</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Holy See</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Saint Kitts and Nevis</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Timor-Leste</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Dominica</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>188 rows × 242 columns</p>
</div> -->




```python
# Keep needed column
dataset_deaths = pd.DataFrame(dataset_deaths["max_death_rate"])

dataset_deaths.head(1000)
```




<!-- <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>max_death_rate</th>
    </tr>
    <tr>
      <th>Country/Region</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>US</th>
      <td>2609.0</td>
    </tr>
    <tr>
      <th>Brazil</th>
      <td>1595.0</td>
    </tr>
    <tr>
      <th>India</th>
      <td>2003.0</td>
    </tr>
    <tr>
      <th>Mexico</th>
      <td>1092.0</td>
    </tr>
    <tr>
      <th>United Kingdom</th>
      <td>1224.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>Cambodia</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Holy See</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Saint Kitts and Nevis</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Timor-Leste</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Dominica</th>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>188 rows × 1 columns</p>
</div> -->



### Recovered

#### Import file

Import `time_series_covid19_recovered_global.csv`


```python
# Importing data
dataset_recovered = pd.read_csv("time_series_covid19_recovered_global.csv")

# Delete useless column
dataset_recovered.drop(["Lat", "Long", "Province/State"],axis=1, inplace=True)

# Set index
# dataset.set_index("Country/Region", inplace=True)
dataset_recovered = dataset_recovered.groupby("Country/Region").sum()

dataset_recovered.head(1000)
```




<!-- <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1/22/20</th>
      <th>1/23/20</th>
      <th>1/24/20</th>
      <th>1/25/20</th>
      <th>1/26/20</th>
      <th>1/27/20</th>
      <th>1/28/20</th>
      <th>1/29/20</th>
      <th>1/30/20</th>
      <th>1/31/20</th>
      <th>...</th>
      <th>9/9/20</th>
      <th>9/10/20</th>
      <th>9/11/20</th>
      <th>9/12/20</th>
      <th>9/13/20</th>
      <th>9/14/20</th>
      <th>9/15/20</th>
      <th>9/16/20</th>
      <th>9/17/20</th>
      <th>9/18/20</th>
    </tr>
    <tr>
      <th>Country/Region</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Afghanistan</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>31048</td>
      <td>31129</td>
      <td>31154</td>
      <td>31234</td>
      <td>31638</td>
      <td>32073</td>
      <td>32098</td>
      <td>32503</td>
      <td>32505</td>
      <td>32576</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>6284</td>
      <td>6346</td>
      <td>6443</td>
      <td>6494</td>
      <td>6569</td>
      <td>6615</td>
      <td>6668</td>
      <td>6733</td>
      <td>6788</td>
      <td>6831</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>33379</td>
      <td>33562</td>
      <td>33723</td>
      <td>33875</td>
      <td>34037</td>
      <td>34204</td>
      <td>34385</td>
      <td>34517</td>
      <td>34675</td>
      <td>34818</td>
    </tr>
    <tr>
      <th>Andorra</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>938</td>
      <td>938</td>
      <td>943</td>
      <td>943</td>
      <td>943</td>
      <td>945</td>
      <td>945</td>
      <td>1054</td>
      <td>1054</td>
      <td>1164</td>
    </tr>
    <tr>
      <th>Angola</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1245</td>
      <td>1277</td>
      <td>1288</td>
      <td>1289</td>
      <td>1301</td>
      <td>1324</td>
      <td>1332</td>
      <td>1401</td>
      <td>1405</td>
      <td>1443</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>West Bank and Gaza</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>18466</td>
      <td>18821</td>
      <td>19788</td>
      <td>19979</td>
      <td>20082</td>
      <td>21406</td>
      <td>21804</td>
      <td>22209</td>
      <td>23060</td>
      <td>23333</td>
    </tr>
    <tr>
      <th>Western Sahara</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Yemen</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1209</td>
      <td>1211</td>
      <td>1211</td>
      <td>1211</td>
      <td>1212</td>
      <td>1215</td>
      <td>1219</td>
      <td>1221</td>
      <td>1221</td>
      <td>1221</td>
    </tr>
    <tr>
      <th>Zambia</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>11839</td>
      <td>11899</td>
      <td>11899</td>
      <td>12007</td>
      <td>12260</td>
      <td>12380</td>
      <td>12590</td>
      <td>12869</td>
      <td>13029</td>
      <td>13207</td>
    </tr>
    <tr>
      <th>Zimbabwe</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>5542</td>
      <td>5635</td>
      <td>5660</td>
      <td>5675</td>
      <td>5678</td>
      <td>5690</td>
      <td>5783</td>
      <td>5823</td>
      <td>5841</td>
      <td>5883</td>
    </tr>
  </tbody>
</table>
<p>188 rows × 241 columns</p>
</div> -->




```python
# This dataset_recovered_filtered will be used with dataset_cases_filtered we created in Confirmed Cases Import file
dataset_recovered_filtered = pd.DataFrame(dataset_recovered["9/18/20"])
# Rename
dataset_recovered_filtered = dataset_recovered_filtered.rename(columns={'9/18/20': 'recovered'})
```

Sort the `dataset_recovered` descending by the value in the newest column.


```python
dataset_recovered = dataset_recovered.sort_values(by=['9/18/20'], ascending=False)
dataset_recovered.head(1000)
```




<!-- <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1/22/20</th>
      <th>1/23/20</th>
      <th>1/24/20</th>
      <th>1/25/20</th>
      <th>1/26/20</th>
      <th>1/27/20</th>
      <th>1/28/20</th>
      <th>1/29/20</th>
      <th>1/30/20</th>
      <th>1/31/20</th>
      <th>...</th>
      <th>9/9/20</th>
      <th>9/10/20</th>
      <th>9/11/20</th>
      <th>9/12/20</th>
      <th>9/13/20</th>
      <th>9/14/20</th>
      <th>9/15/20</th>
      <th>9/16/20</th>
      <th>9/17/20</th>
      <th>9/18/20</th>
    </tr>
    <tr>
      <th>Country/Region</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>India</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3471783</td>
      <td>3542663</td>
      <td>3624196</td>
      <td>3702595</td>
      <td>3780107</td>
      <td>3859399</td>
      <td>3942360</td>
      <td>4025079</td>
      <td>4112551</td>
      <td>4112551</td>
    </tr>
    <tr>
      <th>Brazil</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3611632</td>
      <td>3657701</td>
      <td>3695158</td>
      <td>3723206</td>
      <td>3723206</td>
      <td>3770138</td>
      <td>3811505</td>
      <td>3845464</td>
      <td>3873934</td>
      <td>3897539</td>
    </tr>
    <tr>
      <th>US</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2387479</td>
      <td>2403511</td>
      <td>2417878</td>
      <td>2434658</td>
      <td>2451406</td>
      <td>2474570</td>
      <td>2495127</td>
      <td>2525573</td>
      <td>2540334</td>
      <td>2556465</td>
    </tr>
    <tr>
      <th>Russia</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>854069</td>
      <td>859961</td>
      <td>865646</td>
      <td>871000</td>
      <td>873684</td>
      <td>876152</td>
      <td>881693</td>
      <td>887457</td>
      <td>893145</td>
      <td>898420</td>
    </tr>
    <tr>
      <th>Colombia</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>552885</td>
      <td>569479</td>
      <td>582694</td>
      <td>592820</td>
      <td>599385</td>
      <td>606925</td>
      <td>607978</td>
      <td>610078</td>
      <td>615457</td>
      <td>615457</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Holy See</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
    </tr>
    <tr>
      <th>Western Sahara</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Sweden</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Serbia</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>MS Zaandam</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>188 rows × 241 columns</p>
</div> -->



#### Recovered in specific countries


```python
dataset_recovered.loc["Vietnam"].plot(figsize=(14, 6), legend = True, title = 'Recovered in Vietnam, Singapore, Japan and China')
dataset_recovered.loc["Singapore"].plot()
dataset_recovered.loc["Japan"].plot()
dataset_recovered.loc["China"].plot()
# Identify which line belong to which country
plt.legend()
```




    <matplotlib.legend.Legend at 0x261b7328>




    
![png](assets/output_62_1.png)
    


*Recovered in Vietnam, Singapore, Japan and China*

#### Countries with the highest recovered


```python
# Cast the index of dataframe (name of countries) into list
countries = list(dataset_recovered.index)

# List 10 countries with the highest recovered
count = 0
for country in countries:
    dataset_recovered.loc[country].plot(figsize=(14, 6), legend = True, title = 'Countries with the highest recovered')
    count += 1
    if count == 10:
        break

plt.legend()
```




    <matplotlib.legend.Legend at 0x26c43cb8>




    
![png](assets/output_65_1.png)
    


*Countries with the highest recovered*

#### Look at data in some specific countries about recovered

##### Recovery rate in Vietnam


```python
# Calculate the first derivative of the curve
dataset_recovered.loc["Vietnam"].diff().plot(figsize=(14, 6), legend = True, title = 'Recovered rate in Vietnam')
```




    <AxesSubplot:title={'center':'Recovered rate in Vietnam'}>




    
![png](assets/output_69_1.png)
    



```python
dataset_recovered.loc["Vietnam"].diff().max()
```




    59.0



The `max_recovery_rate` in Vietnam is **59.0**

##### Recovery rate in United State


```python
# Calculate the first derivative of the curve
dataset_recovered.loc["US"].diff().plot(figsize=(14, 6), legend = True, color = 'orange', title = 'Recovery rate in United State')
```




    <AxesSubplot:title={'center':'Recovery rate in United State'}>




    
![png](assets/output_73_1.png)
    



```python
dataset_recovered.loc["US"].diff().max()
```




    103921.0



The `max_recovery_rate` in US is **103921.0**

#### Calculate the `max_recovery_rate` of all countries and add to `dataset_recovered`


```python
# Cast the index of dataframe (name of countries) into list
countries = list(dataset_recovered.index)

# Calculate the max_recovery_rate
max_recovery_rate = []
for country in countries:
    max_recovery_rate.append(dataset_recovered.loc[country].diff().max())

#Add new column
dataset_recovered["max_recovery_rate"] = max_recovery_rate

dataset_recovered.head(1000)
```




<!-- <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1/22/20</th>
      <th>1/23/20</th>
      <th>1/24/20</th>
      <th>1/25/20</th>
      <th>1/26/20</th>
      <th>1/27/20</th>
      <th>1/28/20</th>
      <th>1/29/20</th>
      <th>1/30/20</th>
      <th>1/31/20</th>
      <th>...</th>
      <th>9/10/20</th>
      <th>9/11/20</th>
      <th>9/12/20</th>
      <th>9/13/20</th>
      <th>9/14/20</th>
      <th>9/15/20</th>
      <th>9/16/20</th>
      <th>9/17/20</th>
      <th>9/18/20</th>
      <th>max_recovery_rate</th>
    </tr>
    <tr>
      <th>Country/Region</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>India</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3542663</td>
      <td>3624196</td>
      <td>3702595</td>
      <td>3780107</td>
      <td>3859399</td>
      <td>3942360</td>
      <td>4025079</td>
      <td>4112551</td>
      <td>4112551</td>
      <td>87472.0</td>
    </tr>
    <tr>
      <th>Brazil</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3657701</td>
      <td>3695158</td>
      <td>3723206</td>
      <td>3723206</td>
      <td>3770138</td>
      <td>3811505</td>
      <td>3845464</td>
      <td>3873934</td>
      <td>3897539</td>
      <td>140050.0</td>
    </tr>
    <tr>
      <th>US</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2403511</td>
      <td>2417878</td>
      <td>2434658</td>
      <td>2451406</td>
      <td>2474570</td>
      <td>2495127</td>
      <td>2525573</td>
      <td>2540334</td>
      <td>2556465</td>
      <td>103921.0</td>
    </tr>
    <tr>
      <th>Russia</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>859961</td>
      <td>865646</td>
      <td>871000</td>
      <td>873684</td>
      <td>876152</td>
      <td>881693</td>
      <td>887457</td>
      <td>893145</td>
      <td>898420</td>
      <td>12375.0</td>
    </tr>
    <tr>
      <th>Colombia</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>569479</td>
      <td>582694</td>
      <td>592820</td>
      <td>599385</td>
      <td>606925</td>
      <td>607978</td>
      <td>610078</td>
      <td>615457</td>
      <td>615457</td>
      <td>23868.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Holy See</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>Western Sahara</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Sweden</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Serbia</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4125.0</td>
    </tr>
    <tr>
      <th>MS Zaandam</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>188 rows × 242 columns</p>
</div> -->




```python
# Keep needed column
dataset_recovered = pd.DataFrame(dataset_recovered["max_recovery_rate"])

dataset_recovered.head(1000)
```




<!-- <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>max_recovery_rate</th>
    </tr>
    <tr>
      <th>Country/Region</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>India</th>
      <td>87472.0</td>
    </tr>
    <tr>
      <th>Brazil</th>
      <td>140050.0</td>
    </tr>
    <tr>
      <th>US</th>
      <td>103921.0</td>
    </tr>
    <tr>
      <th>Russia</th>
      <td>12375.0</td>
    </tr>
    <tr>
      <th>Colombia</th>
      <td>23868.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>Holy See</th>
      <td>10.0</td>
    </tr>
    <tr>
      <th>Western Sahara</th>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Sweden</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Serbia</th>
      <td>4125.0</td>
    </tr>
    <tr>
      <th>MS Zaandam</th>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>188 rows × 1 columns</p>
</div> -->



### GDP per capita

#### Import file

Import `gdp-per-capita-worldbank.csv`


```python
# Importing data
dataset_country = pd.read_csv("gdp-per-capita-worldbank.csv")

# Filter
dataset_country = dataset_country[dataset_country.Year.eq(2017)]

# Delete useless column
dataset_country.drop(["Code", "Year"],axis=1, inplace=True)

# Rename
dataset_country = dataset_country.rename(columns={'GDP per capita, PPP (constant 2011 international $)': 'gdp_per_capita'})

# Set index
dataset_country.set_index("Entity", inplace=True)

dataset_country.head(1000)
```




<!-- <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gdp_per_capita</th>
    </tr>
    <tr>
      <th>Entity</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Afghanistan</th>
      <td>1803.987487</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>11803.430594</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>13913.839363</td>
    </tr>
    <tr>
      <th>Angola</th>
      <td>5819.494971</td>
    </tr>
    <tr>
      <th>Antigua and Barbuda</th>
      <td>21490.942659</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>Vanuatu</th>
      <td>2921.908676</td>
    </tr>
    <tr>
      <th>Vietnam</th>
      <td>6171.884192</td>
    </tr>
    <tr>
      <th>World</th>
      <td>15469.207236</td>
    </tr>
    <tr>
      <th>Zambia</th>
      <td>3689.250826</td>
    </tr>
    <tr>
      <th>Zimbabwe</th>
      <td>1899.774977</td>
    </tr>
  </tbody>
</table>
<p>231 rows × 1 columns</p>
</div> -->



### Join all dataset into one


```python
data = dataset_cases
data = data.join(dataset_deaths, how="inner")
data = data.join(dataset_recovered, how="inner")
data.head(1000)
```




<!-- <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>max_infection_rate</th>
      <th>max_death_rate</th>
      <th>max_recovery_rate</th>
    </tr>
    <tr>
      <th>Country/Region</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MS Zaandam</th>
      <td>7.0</td>
      <td>2.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Western Sahara</th>
      <td>4.0</td>
      <td>1.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Holy See</th>
      <td>3.0</td>
      <td>0.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>Saint Kitts and Nevis</th>
      <td>5.0</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>Laos</th>
      <td>3.0</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Colombia</th>
      <td>15318.0</td>
      <td>442.0</td>
      <td>23868.0</td>
    </tr>
    <tr>
      <th>Russia</th>
      <td>11656.0</td>
      <td>232.0</td>
      <td>12375.0</td>
    </tr>
    <tr>
      <th>Brazil</th>
      <td>69074.0</td>
      <td>1595.0</td>
      <td>140050.0</td>
    </tr>
    <tr>
      <th>India</th>
      <td>97894.0</td>
      <td>2003.0</td>
      <td>87472.0</td>
    </tr>
    <tr>
      <th>US</th>
      <td>77255.0</td>
      <td>2609.0</td>
      <td>103921.0</td>
    </tr>
  </tbody>
</table>
<p>188 rows × 3 columns</p>
</div> -->




```python
data = data.join(dataset_country, how="inner")
data.head(1000)
```




<!-- <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>max_infection_rate</th>
      <th>max_death_rate</th>
      <th>max_recovery_rate</th>
      <th>gdp_per_capita</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Saint Kitts and Nevis</th>
      <td>5.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>24654.385401</td>
    </tr>
    <tr>
      <th>Laos</th>
      <td>3.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>6397.359825</td>
    </tr>
    <tr>
      <th>Grenada</th>
      <td>6.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>13593.876918</td>
    </tr>
    <tr>
      <th>Dominica</th>
      <td>5.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>9673.366962</td>
    </tr>
    <tr>
      <th>Saint Lucia</th>
      <td>6.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>12951.838877</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Peru</th>
      <td>21358.0</td>
      <td>4143.0</td>
      <td>18627.0</td>
      <td>12236.706152</td>
    </tr>
    <tr>
      <th>Colombia</th>
      <td>15318.0</td>
      <td>442.0</td>
      <td>23868.0</td>
      <td>13254.949218</td>
    </tr>
    <tr>
      <th>Russia</th>
      <td>11656.0</td>
      <td>232.0</td>
      <td>12375.0</td>
      <td>24765.953634</td>
    </tr>
    <tr>
      <th>Brazil</th>
      <td>69074.0</td>
      <td>1595.0</td>
      <td>140050.0</td>
      <td>14103.451531</td>
    </tr>
    <tr>
      <th>India</th>
      <td>97894.0</td>
      <td>2003.0</td>
      <td>87472.0</td>
      <td>6426.674406</td>
    </tr>
  </tbody>
</table>
<p>161 rows × 4 columns</p>
</div>
 -->

```python
# Remove the rows don't have data
data = data[data.gdp_per_capita.notnull()]
data = data[data.gdp_per_capita != 0]
data.head(1000)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>max_infection_rate</th>
      <th>max_death_rate</th>
      <th>max_recovery_rate</th>
      <th>gdp_per_capita</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Saint Kitts and Nevis</th>
      <td>5.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>24654.385401</td>
    </tr>
    <tr>
      <th>Laos</th>
      <td>3.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>6397.359825</td>
    </tr>
    <tr>
      <th>Grenada</th>
      <td>6.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>13593.876918</td>
    </tr>
    <tr>
      <th>Dominica</th>
      <td>5.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>9673.366962</td>
    </tr>
    <tr>
      <th>Saint Lucia</th>
      <td>6.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>12951.838877</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Peru</th>
      <td>21358.0</td>
      <td>4143.0</td>
      <td>18627.0</td>
      <td>12236.706152</td>
    </tr>
    <tr>
      <th>Colombia</th>
      <td>15318.0</td>
      <td>442.0</td>
      <td>23868.0</td>
      <td>13254.949218</td>
    </tr>
    <tr>
      <th>Russia</th>
      <td>11656.0</td>
      <td>232.0</td>
      <td>12375.0</td>
      <td>24765.953634</td>
    </tr>
    <tr>
      <th>Brazil</th>
      <td>69074.0</td>
      <td>1595.0</td>
      <td>140050.0</td>
      <td>14103.451531</td>
    </tr>
    <tr>
      <th>India</th>
      <td>97894.0</td>
      <td>2003.0</td>
      <td>87472.0</td>
      <td>6426.674406</td>
    </tr>
  </tbody>
</table>
<p>161 rows × 4 columns</p>
</div>


### Calculate Correlation Matrix

Correlation coefficients between `max_infection_rate`, `max_death_rate`, `max_recovery_rate` and `gdp_per_capita`


```python
# Correlation Matrix
data.corr()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>max_infection_rate</th>
      <th>max_death_rate</th>
      <th>max_recovery_rate</th>
      <th>gdp_per_capita</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>max_infection_rate</th>
      <td>1.000000</td>
      <td>0.582025</td>
      <td>0.798164</td>
      <td>-0.013862</td>
    </tr>
    <tr>
      <th>max_death_rate</th>
      <td>0.582025</td>
      <td>1.000000</td>
      <td>0.436201</td>
      <td>-0.003541</td>
    </tr>
    <tr>
      <th>max_recovery_rate</th>
      <td>0.798164</td>
      <td>0.436201</td>
      <td>1.000000</td>
      <td>-0.009103</td>
    </tr>
    <tr>
      <th>gdp_per_capita</th>
      <td>-0.013862</td>
      <td>-0.003541</td>
      <td>-0.009103</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>


### Visualization from data

#### Recovery Rate

Join the `dataset_cases_filtered` and `dataset_recovered_filtered` into `data_filtered`


```python
# Join the dataset_cases_filtered and dataset_recovered_filtered into data_filtered
data_filtered = dataset_cases_filtered
data_filtered = data_filtered.join(dataset_recovered_filtered, how="inner")
data_filtered.head(1000)
```


<!-- <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>confirmed_cases</th>
      <th>recovered</th>
    </tr>
    <tr>
      <th>Country/Region</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Afghanistan</th>
      <td>38883</td>
      <td>32576</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>12073</td>
      <td>6831</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>49413</td>
      <td>34818</td>
    </tr>
    <tr>
      <th>Andorra</th>
      <td>1564</td>
      <td>1164</td>
    </tr>
    <tr>
      <th>Angola</th>
      <td>3848</td>
      <td>1443</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>West Bank and Gaza</th>
      <td>34401</td>
      <td>23333</td>
    </tr>
    <tr>
      <th>Western Sahara</th>
      <td>10</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Yemen</th>
      <td>2024</td>
      <td>1221</td>
    </tr>
    <tr>
      <th>Zambia</th>
      <td>14022</td>
      <td>13207</td>
    </tr>
    <tr>
      <th>Zimbabwe</th>
      <td>7647</td>
      <td>5883</td>
    </tr>
  </tbody>
</table>
<p>188 rows × 2 columns</p>
</div> -->


Calculate the `recovery_rate` column by dividing the `recovered` by the `confirmed_cases`


```python
temp_data_filtered = data_filtered.sort_values(by=['confirmed_cases'], ascending=False)
temp_data_filtered["confirmed_cases"].plot(kind = 'bar', figsize=(42, 6), legend = True, title = 'Confirmed cases of all countries')
```


    <AxesSubplot:title={'center':'Confirmed cases of all countries'}, xlabel='Country/Region'>

    
![png](assets/output_92_1.png)
    


*Confirmed cases of all countries*


```python
# Calculate the recovery_rate column by dividing the recovered by the confirmed_cases
# recovery_rate <= 1
data_filtered["recovery_rate"] = data_filtered["recovered"] / data_filtered["confirmed_cases"]

# Sort the data_filtered by the value of recovery_rate
data_filtered = data_filtered.sort_values(by=['recovery_rate'], ascending=False)

data_filtered.head(1000)
```




<!-- <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>confirmed_cases</th>
      <th>recovered</th>
      <th>recovery_rate</th>
    </tr>
    <tr>
      <th>Country/Region</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Saint Kitts and Nevis</th>
      <td>17</td>
      <td>17</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Holy See</th>
      <td>12</td>
      <td>12</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Saint Vincent and the Grenadines</th>
      <td>64</td>
      <td>64</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Grenada</th>
      <td>24</td>
      <td>24</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Cambodia</th>
      <td>275</td>
      <td>274</td>
      <td>0.996364</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Netherlands</th>
      <td>94345</td>
      <td>2716</td>
      <td>0.028788</td>
    </tr>
    <tr>
      <th>United Kingdom</th>
      <td>388416</td>
      <td>2213</td>
      <td>0.005697</td>
    </tr>
    <tr>
      <th>Sweden</th>
      <td>88237</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>MS Zaandam</th>
      <td>9</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Serbia</th>
      <td>32757</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>188 rows × 3 columns</p>
</div> -->


```python
data_filtered["recovery_rate"].plot(kind = 'bar', figsize=(42, 6), legend = True, title = 'Recovery rate of all countries')
```

    <AxesSubplot:title={'center':'Recovery rate of all countries'}, xlabel='Country/Region'>

    
![png](assets/output_95_1.png)
    


*Recovery rate of all countries*

#### Graph: `max_infection_rate` and `max_death_rate`


```python
x = data["max_infection_rate"]
y = data["max_death_rate"]
# sns.scatterplot(x,y)
# np.log(y) make the graph easier to read, larger scale
sns.scatterplot(x, y)
```

    c:\users\admin\appdata\local\programs\python\python38-32\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    <AxesSubplot:xlabel='max_infection_rate', ylabel='max_death_rate'>

    
![png](assets/output_98_2.png)
    


```python
sns.regplot(x, y)
```

    c:\users\admin\appdata\local\programs\python\python38-32\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    <AxesSubplot:xlabel='max_infection_rate', ylabel='max_death_rate'>


![png](assets/output_99_2.png)
    


#### Graph: `max_infection_rate` and `max_recovery_rate`


```python
x = data["max_infection_rate"]
y = data["max_recovery_rate"]
# sns.scatterplot(x,y)
# np.log(y) make the graph easier to read, larger scale
sns.scatterplot(x, y)
```

    c:\users\admin\appdata\local\programs\python\python38-32\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    <AxesSubplot:xlabel='max_infection_rate', ylabel='max_recovery_rate'>


![png](assets/output_101_2.png)
    

```python
sns.regplot(x, y)
```

    c:\users\admin\appdata\local\programs\python\python38-32\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    <AxesSubplot:xlabel='max_infection_rate', ylabel='max_recovery_rate'>


![png](assets/output_102_2.png)
    


#### Graph: `gdp_per_capita` and `max_infection_rate`


```python
x = data["gdp_per_capita"]
y = data["max_infection_rate"]
# sns.scatterplot(x,y)
# np.log(y) make the graph easier to read, larger scale
sns.scatterplot(x, np.log(y))
```

    c:\users\admin\appdata\local\programs\python\python38-32\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    <AxesSubplot:xlabel='gdp_per_capita', ylabel='max_infection_rate'>


![png](assets/output_104_2.png)
    

```python
sns.regplot(x, np.log(y))
```

    c:\users\admin\appdata\local\programs\python\python38-32\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    <AxesSubplot:xlabel='gdp_per_capita', ylabel='max_infection_rate'>


![png](assets/output_105_2.png)


<!-- > #### The result shows that people who are living developed country (higher GDP per capita) are more prone to get infection of SarsCoV2 virus -->

#### Graph: `gdp_per_capita` and `max_recovery_rate`


```python
x = data["gdp_per_capita"]
y = data["max_recovery_rate"]
# sns.scatterplot(x,y)
# np.log(y) make the graph easier to read, larger scale
sns.scatterplot(x, y)
```

    c:\users\admin\appdata\local\programs\python\python38-32\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    <AxesSubplot:xlabel='gdp_per_capita', ylabel='max_recovery_rate'>

    
![png](assets/output_108_2.png)
    

```python
sns.regplot(x, y)
```

    c:\users\admin\appdata\local\programs\python\python38-32\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    <AxesSubplot:xlabel='gdp_per_capita', ylabel='max_recovery_rate'>


![png](assets/output_109_2.png)
    


## COVID-19 virus diagnosis from radiography
<!-- COVID-19 virus diagnosis from radiography -->
### Sample Gathering
```
└── 📁dataset_radiography
    ├── 📁data_for_training
    │   ├── 📁covid_infected
    │   │   └── More than 800 JPEG images
    │   └── 📁normal
    │       └── More than 800 JPEG images
    └── 📁data_for_testing
        ├── 📁covid_infected
        │   └── 100 JPEG images
        └── 📁normal
            └── 100 JPEG images
```
More than 1600 samples, include 800 X-ray JPEG images of COVID-19 infected patients and 800 X-ray JPEG images of normal people are used to train the model. We keep the remain 100 images of each category for testing after the model was trained.
### Model Trainging
Teachable Machine is using a technique called [transfer learning](https://www.tensorflow.org/tutorials/images/transfer_learning). There is a pre-trained model. Somebody else already trained a model on many, many samples, and this model is called [MobileNet](https://arxiv.org/pdf/1704.04861.pdf). And the training data for this model is called [ImageNet](http://www.image-net.org/papers/imagenet_cvpr09.pdf).

[MobileNet](https://arxiv.org/pdf/1704.04861.pdf) is a model that runs fast, works in the browser, and knows about a lot of different kinds of things. But it doesn't necessarily know whether the X-ray image belongs to a COVID-19 infected patient or a normal person. Maybe it's in there, maybe it's not. But what we could do is take the fact that it's learned how to boil the essence of images down into a bunch of different numbers, to then retrain it with our own images.

And this allows us to train a machine learning model with 1600 samples from the dataset we have gathered. Because if we were training an image classification model from scratch without this base model, you probably would need a much larger data set.
### Model Embedding

Import the `ml5.js` and `p5.js`

``` html
<!-- p5 -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/0.9.0/p5.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/0.9.0/addons/p5.dom.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/0.9.0/addons/p5.sound.min.js"></script>
<!-- ml5 -->
<script src="https://unpkg.com/ml5@0.4.3/dist/ml5.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Open+Sans:ital,wght@0,300;0,400;0,600;0,700;0,800;1,300;1,400;1,600;1,700;1,800&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Roboto:ital,wght@0,100;0,300;0,400;0,500;0,700;0,900;1,100;1,300;1,400;1,500;1,700;1,900&display=swap" rel="stylesheet">
```

Embed the model

``` Javascript
// Classifier Variable
    let classifier;
    // Model URL
    let imageModelURL = 'https://teachablemachine.withgoogle.com/models/V4ZNTcbKF/';
    // Video
    let video;
    let flippedVideo;
    // To store the classification
    let label = "Waiting for capture device";
    let label1 = "Waiting for capture device";
    let label2 = "Waiting for capture device";
    let label3 = "Waiting for capture device";
    let confidence = 0;
    let confidence1 = 0.0;
    let confidence2 = 0.0;
    let confidence3 = 0.0;
    // Some variables
    var w = window.innerWidth;
    var h = window.innerHeight;
    var videoWidth = w*0.3;
    var videoHeight = videoWidth/4*3;
    var lineHeight = 32;
    // Load the model first
    function preload() {
        classifier = ml5.imageClassifier(imageModelURL + 'model.json');
    }

    var canvas;
    function setup() {
        canvas = createCanvas(w, h);
        canvas.position(0,0);
        canvas.style('z-index', '-1');

        // Create the video
        video = createCapture(VIDEO);
        video.size(videoWidth, videoHeight);
        video.hide();

        // flippedVideo = ml5.flipImage(video);
        // Start classifying
        classifyVideo();
    }

    function draw() {
        background(0);
        // Draw the video
        

        rect(w/2 - videoWidth - 55, h/2 - videoHeight/2 - 5, videoWidth+9, videoHeight+9);
        image(flippedVideo, w/2 - videoWidth - 50, h/2 - videoHeight/2);

        
        stroke('transparent');
        // Vertically
        rect(w/2 - videoWidth/8*1 - 50, h/2 - videoHeight/2, 0, videoHeight);
        rect(w/2 - videoWidth/8*2 - 50, h/2 - videoHeight/2, 0, videoHeight);
        rect(w/2 - videoWidth/8*3 - 50, h/2 - videoHeight/2, 0, videoHeight);
        rect(w/2 - videoWidth/8*4 - 50, h/2 - videoHeight/2, 0, videoHeight);
        rect(w/2 - videoWidth/8*5 - 50, h/2 - videoHeight/2, 0, videoHeight);
        rect(w/2 - videoWidth/8*6 - 50, h/2 - videoHeight/2, 0, videoHeight);
        rect(w/2 - videoWidth/8*7 - 50, h/2 - videoHeight/2, 0, videoHeight);
        // Horizontally
        rect(w/2 - videoWidth - 50, h/2 + videoHeight/6*1, videoWidth, 0);
        rect(w/2 - videoWidth - 50, h/2 + videoHeight/6*2, videoWidth, 0);
        rect(w/2 - videoWidth - 50, h/2 + videoHeight/6*0, videoWidth, 0);
        rect(w/2 - videoWidth - 50, h/2 - videoHeight/6*1, videoWidth, 0);
        rect(w/2 - videoWidth - 50, h/2 - videoHeight/6*2, videoWidth, 0);
        stroke('black');

        // Draw the label
        fill(255);
        textSize(22);
        textAlign(LEFT);

        text(label, w/2, h/2 - videoHeight/2 + lineHeight);
        text(confidence.toFixed(5) + "%", w/2, h/2 - videoHeight/2 + lineHeight*2);
        rect(w/2, h/2 - videoHeight/2 + lineHeight*2.5, confidence * videoWidth /100, lineHeight/2);

        text(label1, w/2, h/2 - videoHeight/2 + lineHeight*4.5);
        text(confidence1.toFixed(5) + "%", w/2, h/2 - videoHeight/2 + lineHeight*5.5);
        rect(w/2, h/2 - videoHeight/2 + lineHeight*6, confidence1 * videoWidth /100, lineHeight/2);

        text(label2, w/2, h/2 - videoHeight/2 + lineHeight*8);
        text(confidence2.toFixed(5) + "%", w/2, h/2 - videoHeight/2 + lineHeight*9);
        rect(w/2, h/2 - videoHeight/2 + lineHeight*9.5, confidence2 * videoWidth /100, lineHeight/2);

        textSize(14);
        textAlign(LEFT);
        text("Important Notice: Keep the X-ray image filling the whole frame", w/2 - videoWidth - 50, h/2 - videoHeight/2 - 20);

        textSize(18);
        textAlign(CENTER);
        text("Diagnosis from Radiography", width / 2, height - lineHeight*3.5);
        textSize(36);
        text(label, width / 2, height - lineHeight*2);

    }

    // Get a prediction for the current video frame
    function classifyVideo() {
        flippedVideo = ml5.flipImage(video);
        classifier.classify(flippedVideo, gotResult);
        flippedVideo.remove();
    }

    // When we get a result
    function gotResult(error, results) {
        // If there is an error
        if (error) {
        console.error(error);
        return;
        }
        // The results are in an array ordered by confidence.
        // console.log(results[0]);
        label = results[0].label;
        label1 = results[1].label;
        label2 = results[2].label;
        // label3 = results[3].label;
        confidence = results[0].confidence * 100;
        confidence1 = results[1].confidence * 100;
        confidence2 = results[2].confidence * 100;
        // confidence3 = results[3].confidence;
        // Classifiy again!
        classifyVideo();
    }
```
### Testing
![png](assets/diag-normal.gif)
![png](assets/diag-covid.gif)

## Results
#### Analysis Conclusion
- The result shows that people who are living in a more developed country (higher GDP per capita) are more prone to get infection of the COVID-19 virus.
- The maximum recovery rate of a country is not affected by its development. Developed countries are as vulnerable to COVID-19 pandemic as developing countries.

#### Diagnosis Accuracy
196/200 (98%) test cases have the right results. The trained model has recognized whether an X-ray image belongs to a COVID-19 infected patient or a non-infected person.

#### Strengths of the method
- Based on the existing open-source platform, we can quickly build a machine learning model that can diagnose the COVID-19 disease from the radiography images.
- The technique used here is transfer learning, we have MobileNet helps us to boil the essence of images down into a bunch of different numbers, and then we just retrain it with our own samples.

#### Limitations of the method
- The practical application of this research absolutely depends on the sample resource, namely the X-ray images of COVID-19 infected patient and non-infected person, both must be taken in the same condition and environment.

#### Important Notice
⚠ This research is intended for **educational purposes only**. As the "Limitations of the method" mentioned, the practical application of this research absolutely depends on the sample resource, and the trained model in this project does not satisfy it. Please do not use the results without real clinical researches!

## References
- Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko,
Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam. [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf)
- Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li and Li Fei-Fei Dept. of Computer Science, Princeton University. [ImageNet: A Large-Scale Hierarchical Image Database.](http://www.image-net.org/papers/imagenet_cvpr09.pdf)
- World Health Organization. Coronavirus. https://www.who.int/health-topics/coronavirus
- Centers for Disease Control and Prevention. COVID-19 Overview and Infection Prevention and Control Priorities in non-US Healthcare Settings. https://www.cdc.gov/coronavirus/2019-ncov/hcp/non-us-settings/overview/index.html
- Wikipedia. Coronavirus disease 2019. https://en.wikipedia.org/wiki/Coronavirus_disease_2019
- ExpertSystem. What is Machine Learning. https://expertsystem.com/machine-learning-definition/
- Microsoft. What is a machine learning model? https://docs.microsoft.com/en-us/windows/ai/windows-ml/what-is-a-machine-learning-model
- TensorFlow. Why TensorFlow. https://www.tensorflow.org/about
- Google. Teachable Machine 2.0. https://blog.google/technology/ai/teachable-machine/
- Python. What is Python? Executive Summary. https://www.python.org/doc/essays/blurb/
- RTInsight. Why Python is Essential for Data Analysis. https://www.rtinsights.com/why-python-is-essential-for-data-analysis/
- The Jupyter Notebook. https://jupyter.org/index.html

______________________
> Please keep my name if you want to bring this article anywhere else.
> [Bui Quang Bao](https://buiquangbao.github.io/)