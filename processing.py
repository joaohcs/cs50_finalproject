### Reference Script for Processing the Database and Returning Cohort Graphs ###

## Basic Rules:
# File must have two columns: 'Date' and 'ID'
# ID can be any unique identifier (CPF, RG, Unique ID...)
# Date can be, well, any date
# File extension can be .csv, .xlsx, .xlsm, .xlsb, .xls


## Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import random
from datetime import timedelta
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
import openai
import os
from dotenv import load_dotenv

## API Configuration for LLM Usage
load_dotenv('secrets.env')

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


## Receives CSV or Excel file and returns Pandas Dataframe
def load_data(data_path):
    # Identify format - checking file extension
    _, file_extension = os.path.splitext(data_path)

    # csv
    if file_extension == '.csv':
        data = pd.read_csv(data_path)
    
        return data
    
    else:
    # excel extensions
        try:
            data = pd.read_excel(data_path)
            
            return data
        
        except ValueError as e:
            return f"Wrong type of Data, file must be .csv, .xlsx, .xls or .xlsm.\nError: {e}"


## Receives Raw Pandas Dataframe and returns DF ready for plotting insights
def format_data(data):

    ## Identify Columns (Which one is an ID and which is Date)
    date_column = None
    id_column = None

    for column in data.columns:
        # Try to convert the column to datetime type
        try:
            converted = pd.to_datetime(data[column])
            
            # Check for conversion success. If successful, assume it is Date column
            if converted.notnull().all():
                date_column = column
        
        except (ValueError, TypeError):
            pass

    id_column = [col for col in data.columns if col != date_column][0]

    ## Transforming date column to datetime
    data[date_column] = pd.to_datetime(data[date_column])

    ## Creating cohorts
    data['cohort'] = data.groupby(id_column)[date_column].transform('min')
    
    ## Creating column with year-month for the 'event' and cohort
    # Note that 'event' depends on what each line of the spreadsheet means in the user context
    # It can be either sales made by brokers, subscription purchases etc...
    data['event_yearmonth'] = data[date_column].dt.to_period('M') 
    data['cohort_yearmonth'] = data['cohort'].dt.to_period('M')

    ## Creating new dataframe with cohorts information
    # Grouping the data by cohort and event year-month, and counting the number of unique ids
    # That means we'll have a dataframe with the first column being the cohort, the second column being the event year_month and the number of unique ids for that combination
    # Example: 
    # 0 - Cohort (2024-01) - Event (2024-01) - n_ids (15)
    # 1 - Cohort (2024-01) - Event (2024-02) - n_ids (10)
    # We had 15 customers from the january cohort buying in january, and  10 customers from jan cohort also buying in february.
    cohort_data = data.groupby(['cohort_yearmonth', 'event_yearmonth']).agg(n_unique_ids=(id_column, 'nunique')).reset_index()
    
    ## Creates a period_number column with the difference in months between the event and the cohort date.
    cohort_data['period_number'] = (cohort_data['event_yearmonth'] - cohort_data['cohort_yearmonth']).apply(lambda x: x.n)
    
    ## Pivot cohort data - creating a matrix in which cohorts are lines and period numbers are columns. n_unique_ids is the value for each combination of cohort and period number
    # Example:
    # period_number     0   1
    # cohort_yearmonth
    # 2024-01           15  10
    # Note that it is precisely the structure of a retention matrix used in cohort analysis
    cohort_pivot = cohort_data.pivot_table(index='cohort_yearmonth', columns='period_number', values='n_unique_ids', aggfunc='sum')
    cohort_sizes = cohort_pivot.iloc[:, 0] # Gets the first column (period_number=0) - basically the size of each cohort

    ## Dividing each value in the cohort line by the initial size of the cohort
    # Example:
    # period_number     0      1
    # cohort_yearmonth
    # 2024-01           15/15  10/15
    # 2024-02           25/25  5/25
    # Result:
    # period_number     0      1
    # cohort_yearmonth
    # 2024-01           1      0.67
    # 2024-02           1      0.20
    # Note how that is precisely the structure of a retention matrix in a 0-1 scale - you can multiply by 100 to get %
    retention_matrix = cohort_pivot.divide(cohort_sizes, axis=0)

    return retention_matrix, cohort_sizes


## Receives formatted DataFrame and generate graphs
def plot_graphs(retention_matrix, cohort_sizes, use_percentage=True):

    # Color map options for graph
    pink_colors = ['#ffffff', '#f9ccec', '#f487cf', '#e40494']  
    blue_colors = ['#ffffff', '#cce5ff', '#87b8f4', '#0478e4']
    green_colors = ['#ffffff', '#ccffd9', '#87f4b8', '#04e464']
    purple_colors = ['#ffffff', '#e5ccff', '#b887f4', '#7804e4']

    # Choosing a random color map from options - we will let user decide on it later
    map = random.choice([pink_colors, blue_colors, green_colors, purple_colors])

    # Creating personalized color map
    color_cmap = LinearSegmentedColormap.from_list("Custom", map, N=256)


    # Plotting the matrix
    # Defining the format (% or not)
    fmt = ".0%" if use_percentage else "d" # "d" makes it display the raw values

    # Calculate vmin and vmax for graph
    flattened_values = retention_matrix.values.flatten()
    print(f'flattened values {flattened_values}')
    filtered_values = flattened_values[(flattened_values != 1) & ~np.isnan(flattened_values)]   # Exclude value 1 and nan
    print(f'filtered values {filtered_values}')
    vmin = filtered_values.min()
    vmax = filtered_values.max()
    vmin = round((vmin + 0.05*(vmax - vmin)), 2)
    vmax = round((vmax - 0.05*(vmax - vmin)), 2)
    print(f'vmin {vmin} e vmax {vmax}')

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(retention_matrix, annot=True, fmt=fmt, cmap=color_cmap, vmax=vmax, vmin=vmin, ax=ax) # Mudar vmax a depender dos valores de retenção para melhor visualização

    # Adding cohort sizes in the graph
    for i, size in enumerate(cohort_sizes):
        ax.annotate(f'n:\n{int(size)}', xy=(0, i), xycoords='data', textcoords='offset points', xytext=(10, -15),
                    ha='center', va='center', color='white', weight='bold', fontsize=9, bbox=dict(facecolor='white', edgecolor='none', alpha=0.1))

    ax.set_title('Cohort Analysis - Retention of Unique IDs')
    ax.set_ylabel('Cohort (Month of first "Event")')
    ax.set_xlabel('Months after first "Event"')

    # Save the figure
    static_dir = 'static'
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
    
    plt.savefig(os.path.join(static_dir, 'retention_heatmap.png'))

    # Return the figure object
    return fig


## Receives graphs and generate insights
def llm_analysis(retention_matrix, prompt):
    
    # Generate Response by feeding the retention matrix created
        # Convert retention matrix to a string
    retention_matrix_str = retention_matrix.to_string()

    # Initializing OpenAI with API key
    llm = OpenAI(api_key=OPENAI_API_KEY, max_tokens=-1)

    # Create the prompt
    question = f"""{prompt}\n\nRetention Matrix:\n{retention_matrix_str}\n
    """

    template = """
    Question: {question}
    
    Answer: Let's analyze it step by step. I will give you an analysis of a venture capitalist.
    """

    prompt_final = PromptTemplate.from_template(template)

    # Langchain chain
    llm_chain = prompt_final | llm

    # Query the LLM
    response = llm_chain.invoke(question)

    return response


## Receives CSV or Excel file, calls all functions, returns graphs plotted and LLM analysis
def process_csv(file):
    file_path = file
    print('--File Path--\n')
    print(file_path)
    data =  load_data(file_path)
    print('\n--Load Data Result--\n')
    print(data)
    retention_matrix, cohort_sizes = format_data(data)
    print('\n--Retention Matrix and Cohort Sizes--\n')
    print(retention_matrix, cohort_sizes)
    fig = plot_graphs(retention_matrix, cohort_sizes, use_percentage=True)
    print('\n --Fig generated')

    prompt = """
            You are a seasoned Venture Capital Analyst in charge of conducting a Cohort Analysis from startup data 
            and creating a Report from it. Act as a investor that is also a strategist and is trying to help.

            Objective: To provide a comprehensive analysis of retention trends over time, helping to understand the effectiveness of our retention strategies and identify opportunities for improvement.

            Data Provided:
            Dataframe as string: The cohort retention matrix with specific retention rates for each cohort and period.
    
            Analysis Request:
            Please analyze the provided cohort retention matrix, focusing on the following aspects:

            Retention Trends: Identify and discuss the overall trends in retention across different cohorts and periods.
            Cohort Comparisons: Compare the retention rates between cohorts. Highlight any significant differences or patterns.
            Implications for Business Strategy: Explain what these retention trends imply for our retention strategies. Suggest actionable insights and strategies to improve retention rates!!
            Investment Justification: Provide a clear, data-driven justification for why these retention trends indicate a promising investment opportunity or why they do not. Include potential areas for further development.
            Conclusion: Summarizing the main insights in bullet points and next steps as investor or as the entrepreneur.
            Output Structure:
            Retention Trends Analysis
            Cohort Comparisons
            Implications for Business Strategy
            Investment Justification
            Conclusion

    """
    llm_response = llm_analysis(retention_matrix, prompt)
    print(llm_response)
    print(len(llm_response))

    plt.show()

    with open("output.txt", "w") as f:
        f.write(llm_response)
    print("Output written to output.txt")

    return fig, llm_response

# if __name__ == "__process_csv__":
#     fig, llm_response = process_csv()
#     with open("output.txt", "w") as f:
#         f.write(llm_response)
#     print("Output written to output.txt")
