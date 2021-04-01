import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as ss
from collections import Counter
import math


def axes_loc(i, width):
    """
    :desc: Yeild axis location for subplot position
    :params:
            i: plot number enumerated (int)
            width: number of subplots on x axis (int)

    :return: int, int: x and y location of subplot
    """

    x = i // width
    y = i % width
    yield x, y

def check_type(col_type):
    """
    :desc: Converting column type to relevant information for plotting variables 
    :params:
            col_type: type of column (str)

    :return: str: relevant type for plotting purposes
    """

    type_map = {'float64': 'NUM', 'int64': 'NUM', 'bool':'CAT', 'category':'CAT', 'object': 'CAT'}
    try:
        col_type = type_map[str(col_type)]
    except:
        print(f'Failed {col_type}')
    return col_type


def multiplots_freq(plot_df, filename, axis_width = 5, w = 20, h = 40):
    """
    :desc: Plotting multiple frequency plots using independent variables
    :params:
            plot_df: pandas dataframe
            filename: name of figure to be saved
            axis_width: number of subplots that should show on the x axis
            w: width of each plot
            h: height of each plot

    :return: None (Saves figure and shows in notebook/ lab env)
    """

    fig, axes = plt.subplots(math.ceil(len(plot_df.columns)/axis_width), axis_width)
    
    for i, col in enumerate(plot_df.columns):
        col_type = check_type(plot_df[col].dtype)
        
        axes_x_y = axes_loc(i, axis_width)
        x, y = next(axes_x_y)
        
        try:
            if col_type == 'CAT':
                plot_df[col].value_counts().plot(
                    kind='bar', ax=axes[x,y], rot=35, color=['0.5', 'salmon'], 
                    edgecolor='0.3', title = col, fontsize = 14)

            elif col_type == 'NUM':
                plot_df[col].plot(
                    kind='hist', ax=axes[x,y], color='LightBlue', 
                    edgecolor='0.5', title = col, fontsize = 14)

        except:
            print(f'{col} could not be plotted. Check data type.')
        
        axes[x,y].title.set_size(25)
        fig = plt.gcf()
        plt.subplots_adjust(left=0.125,bottom=0.05, right=0.9, top=0.90, wspace=0.2, hspace=0.5)
        fig.tight_layout()
        fig.set_size_inches(w, h, forward=True)
        fig.savefig(f'{filename}.png', dpi=100)


def multiplots_freq_target(plot_df, filename, bin_target = 'target', axis_width = 5, w = 20, h = 40):
    """
    :desc: Plotting multiple frequency plots using independent variables against the dependent variable
    		only suitable for binary targets currently
    :params:
            plot_df: pandas dataframe
            filename: name of figure to be saved
            bin_target: binary target (dependent) variable
            axis_width: number of subplots that should show on the x axis
            w: width of each plot
            h: height of each plot

    :return: None (Saves figure and shows in notebook/ lab env)
    """
    fig, axes = plt.subplots(math.ceil(len(plot_df.columns)/axis_width), axis_width)

    map_targets = {0: 'absense', 1:'presence'} # Remove hardcoded values
    #plot_df.plot_target = plot_df[bin_target].apply(lambda x: map_targets[x])
    targets = plot_df[bin_target].unique().tolist()

    for i, col in enumerate(plot_df.drop(bin_target, axis=1).columns):
        col_type = check_type(plot_df[col].dtype)

        df_target = pd.DataFrame()

        axes_x_y = axes_loc(i, axis_width)
        x, y = next(axes_x_y)

        try:
            if col_type == 'CAT':
                df_target[targets[0]] = plot_df[plot_df['target'] == targets[0]][col].value_counts()
                df_target[targets[1]] = plot_df[plot_df['target'] == targets[1]][col].value_counts()
                df_target.plot(
                    kind='bar', ax=axes[x,y], rot=35, color=['0.5', 'salmon'], 
                    edgecolor='0.3', title = col, fontsize = 14)

            elif col_type == 'NUM':
                sns.histplot(ax=axes[x,y], data = plot_df, x=col, hue="target", kde=True, stat='density')
                #axes[x,y].legend(['absense', 'presence'])
        except:
            print(f'{col} could not be plotted. Check data type.')

        axes[x,y].title.set_size(25)
        fig = plt.gcf()
        plt.subplots_adjust(left=0.125,bottom=0.05, right=0.9, top=0.90, wspace=0.2, hspace=0.5)
        fig.tight_layout()
        fig.set_size_inches(w, h, forward=True)
        fig.savefig(f'{filename}.png', dpi=100)

def conditional_entropy(x,
                        y,
                        nan_strategy='replace',
                        nan_replace_value=0.0,
                        log_base: float = math.e):

    if nan_strategy == 'replace':
        x, y = replace_nan_with_value(x, y, nan_replace_value)
    elif nan_strategy == 'drop':
        x, y = remove_incomplete_samples(x, y)
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x, y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0.0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y / p_xy, log_base)
    return entropy

def replace_nan_with_value(x, y, value):
    x = np.array([v if v == v and v is not None else value for v in x])  # NaN != NaN
    y = np.array([v if v == v and v is not None else value for v in y])
    return x, y

def remove_incomplete_samples(x, y):
    x = [v if v is not None else np.nan for v in x]
    y = [v if v is not None else np.nan for v in y]
    arr = np.array([x, y]).transpose()
    arr = arr[~np.isnan(arr).any(axis=1)].transpose()
    if isinstance(x, list):
        return arr[0].tolist(), arr[1].tolist()
    else:
        return arr[0], arr[1]
    
def theils_u(x, y):
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x
    
def correlation_ratio(categories, measurements):
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator/denominator)
    return eta

def create_corr_matrix(df):
    corr_matrix = {}
    for col_x in df.columns:
        temp_mat = []
        col_type_x = check_type(df[col_x].dtype)
        
        for col_y in df.columns:
            col_type_y = check_type(df[col_y].dtype)
            
            if col_type_x == 'NUM' and col_type_y == 'NUM':
                x = np.corrcoef(df[col_x], df[col_y])
                x = x[0][1]
            elif col_type_x == 'NUM' and col_type_y == 'CAT':
                x = correlation_ratio(df[col_y], df[col_x].values)
            elif col_type_x == 'CAT' and col_type_y == 'NUM':
                x = correlation_ratio(df[col_x], df[col_y].values)
            elif col_type_x == 'CAT' and col_type_y == 'CAT':
                x = theils_u(df[col_x], df[col_y])
                
            temp_mat.append(x)
        corr_matrix[col_x] = temp_mat
    return corr_matrix
