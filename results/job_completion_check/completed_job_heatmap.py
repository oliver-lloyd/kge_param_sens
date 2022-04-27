import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import date
from numpy import nan
from matplotlib.colors import LinearSegmentedColormap


def process_job_info(file_name):

    indexes = {
        # Corresponding indexes of split_name for each possible input file type
        'csv': [2, 3, 4, 5, -4],
        'sh': [0, 1, 2, 3, -3]
    }

    file_name = file_name.replace('fb15k-237', 'fb15k_237')
    split_name = file_name.split('-')
    file_type = file_name.split('.')[-1]

    dataset = split_name[indexes[file_type][0]]
    model = split_name[indexes[file_type][1]]
    training_type = split_name[indexes[file_type][2]]
    loss_func = split_name[indexes[file_type][3]][:indexes[file_type][4]]

    return [dataset, model, training_type, loss_func]


if __name__ == '__main__':

    # Construct df of jobs with returned results
    jobs = pd.DataFrame(columns=['dataset', 'model', 'training_type', 'loss', 'finished'])
    for loc in os.listdir('trace_csvs'):
        if loc.endswith('.csv'):
            job_info = process_job_info(loc)
            job_info.append(True)
            jobs.loc[len(jobs)] = job_info

    # Add errored jobs to df
    for loc in os.listdir('errored_jobs'):
        job_info = process_job_info(loc)
        job_info.append(False)
        jobs.loc[len(jobs)] = job_info

    # Create grid for job completion heatmap
    train_x_loss = pd.Series([row.training_type + ', ' + row.loss for i, row in jobs.iterrows()]).unique()
    train_x_loss.sort()
    data_x_model = [model + ', ' + dataset for model in jobs.model.unique() for dataset in jobs.dataset.unique()]
    data_x_model.sort()
    grid = pd.DataFrame(columns=train_x_loss, index=data_x_model)

    # Fill in grid, storing uncomplete job names in jobs_remaining.txt
    for model_dataset in grid.index:
        model, dataset = model_dataset.split(', ')
        subdf = jobs.loc[jobs.model == model].loc[jobs.dataset == dataset]
        for train_loss in grid.columns:
            train, loss = train_loss.split(', ')
            row = subdf.loc[jobs.training_type == train].loc[jobs.loss == loss]
            if len(row) > 0 and row.finished.iloc[0]:
                grid[train_loss][model_dataset] = 1
            else:
                grid[train_loss][model_dataset] = 0

    if nan in grid.values:
        grid.replace(nan, 0, inplace=True)

    # Plot grid
    cmap = LinearSegmentedColormap.from_list(name='redgreen', colors=['red', 'green'])

    plt.figure(figsize=(7, 9))
    sns.heatmap(grid.astype(float), 
                cmap=cmap, 
                linewidths=1.0,
                cbar=False)
    plt.xticks(rotation=25, fontsize=8)
    plt.yticks(fontsize=8)
    plt.title(f'Gridsearch jobs completed as of {date.today().strftime("%d/%m/%y")}')
    plt.ylabel('Model, Dataset')
    plt.xlabel('Training type, Loss function')
    plt.tight_layout()
    #plt.show()
    plt.savefig('job_completion_check/complete_jobs_heatmap.png')