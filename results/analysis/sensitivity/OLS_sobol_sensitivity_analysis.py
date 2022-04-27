import pandas as pd
import numpy as np
import os


from SALib.sample import saltelli
from SALib.analyze import sobol
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':
    pd.options.mode.chained_assignment = None  # Suppress pd printed warnings
    
    # Load all results
    print('Loading LibKGE trial data...')
    traces_path = '../../trace_csvs/'
    all_results = pd.DataFrame()
    for loc in os.listdir(traces_path):
        if loc.endswith('.csv'):
            subdf = pd.read_csv(traces_path + loc)
            subdf['loss_func'] = loc.split('-')[-1].replace('.csv', '')
            all_results = all_results.append(subdf).reset_index(drop=True)

    # Convert weight reg column to 0/1
    all_results['lookup_embedder.regularize_args.weighted'] = all_results['lookup_embedder.regularize_args.weighted'].astype(int)

    # Change WN18RR name
    all_results.dataset.loc[all_results.dataset == 'wnrr'] = 'wn18rr'

    # Remove TransH and RotatE (models didn't run on all three datasets)
    all_results = all_results.loc[all_results.model != 'rotate']
    all_results = all_results.loc[all_results.model != 'transh']

    # Combine split columns
    print('Combining columns split by model...')
    model_names = list(all_results.model.unique())
    for column in all_results.columns:
        if column.split('.')[0] in model_names:
            actual_column = '.'.join(column.split('.')[1:])
            all_results[actual_column] = None
            for column2 in all_results.columns:
                if actual_column in column2 and actual_column != column2:
                    for i, val in enumerate(all_results[column2]):
                        if pd.notna(val):
                            all_results[actual_column][i] = val
                    all_results.drop(columns=[column2], inplace=True)

    # Prepare data
    print('Dummifying categorical inputs...')
    all_results = all_results.loc[pd.notna(all_results.job_type)]
    parameter_names = ['job_type'] + list(all_results.columns[15:])
    x_vars = [col for col in parameter_names if not any(pd.isnull(all_results[col]))]

    for var in x_vars:
        if all_results.dtypes[var] not in [float, int]:
            dummy_df = pd.get_dummies(all_results[var])
            dummy_df.columns = [var + '_dummy:' + str(col) for col in dummy_df.columns]
            all_results.drop(columns=[var], inplace=True)
            all_results = pd.concat([all_results, dummy_df], axis=1)
            x_vars.remove(var)
            x_vars = x_vars + list(dummy_df.columns)

    x_df_all = all_results[x_vars]

    # Select top trials for each dataset
    dfs = {}
    top_x_percent = 5
    print(f'Selecting top {top_x_percent}% of trials by performance...')

    for dataset in all_results.dataset.unique():
        filtered_df = all_results.loc[all_results.dataset == dataset]
        MRR_threshold = np.percentile(filtered_df.metric, 100-top_x_percent)
        filtered_df = filtered_df.loc[filtered_df.metric >= MRR_threshold]
        dfs[dataset] = filtered_df
        print(f'Dataset {dataset}: MRR threshold = {MRR_threshold}, leaves {len(filtered_df)} trials.')

    # Run OLS for each subdf
    print('Running regression...')
    models = {}

    for dataset in dfs:
        x = dfs[dataset][x_vars]
        y = dfs[dataset]['metric']
        model = LinearRegression(normalize=True).fit(x, y)
        models[dataset] = model


    # Get input var bounds
    print('Defining bounds for Sobol sampling...')
    types_df = pd.DataFrame([(column, x_df_all.dtypes[column]) for column in x_df_all.columns], columns=['col_name', 'dtype'])

    bounds = {
        'train.batch_size': [128, 1024],
        'train.optimizer_args.lr': [0.0003, 1],
        'train.lr_scheduler_args.patience': [0, 10],
        'lookup_embedder.dim': [16, 512],
        'lookup_embedder.initialize_args.normal_.std': [0.00001, 1.0],
        'lookup_embedder.initialize_args.uniform_.a': [-1.0, -0.00001]
    }

    dummies_index = {}
    for i, row in types_df.iterrows():
        if row['dtype'] != float:
            bounds[row['col_name']] = [0, 1]
            dummy_var = row['col_name'].split(':')[0]
            if dummy_var not in dummies_index:
                dummies_index[dummy_var] = [i]
            else:
                dummies_index[dummy_var].append(i)

    # Define the model inputs
    print('Performing Sobol sampling...')
    problem = {
        'num_vars': len(x_df_all.columns),
        'names': x_df_all.columns,
        'bounds': [bounds[key] for key in bounds]
    }

    # Generate samples
    param_values = saltelli.sample(problem, 2048)

    # Round dummy var sets to 0 or 1 based on max roll. Note this takes a while to run if debug = True
    #TODO: parallelise this
    print('Arg-min/maxing samples for dummy variables. This may take a moment...')
    debug = False
    for sample in param_values:
        for dummy in dummies_index:
            check_index = dummies_index[dummy] # Get indices of linked dummy vars
            if len(check_index) == 1:
                value = sample[check_index[0]]
                sample[check_index[0]] = round(value)
            else:
                check_values = sample[check_index] # Get sampled values (0:1) of each dummy
                index_of_max = np.argmax(check_values) # Get index of dummy with highest value
                for j, idx in enumerate(check_index):
                    if j == index_of_max: # Set max value dummy to 1
                        sample[idx] = 1
                    else:
                        sample[idx] = 0 # Set non-max dummies to 0
                if debug:
                    sample_verify = np.array(sample)
                    print(f'Inserted booleans: {sample[check_index]}')
                    print(f'Replaced sampled values: {sample_verify[[check_index]]}')
                    print()

    # Get model predictions
    print(f'Getting model predictions for Sobol samples')
    results = {}
    for dataset in models:
        
        # Run model
        Y = models[dataset].predict(param_values)

        # Perform analysis
        Si = sobol.analyze(problem, Y)

        # Store
        results[dataset] = Si        


    # Convert to df and save
    print('Saving...')
    out_df = pd.DataFrame(columns=['dataset', 'parameter', 's1_value', 's1_conf', 'st_value', 'st_conf', ])

    for dataset in results:
        sens_result = results[dataset]
        for i, param in enumerate(x_vars):
            row = [dataset, param]
            for level in ['S1', 'ST']:
                row.append(results[dataset][level][i])
                row.append(results[dataset][level + '_conf'][i])
            out_df.loc[len(out_df)] = row
        
        # Save interaction matrices
        matrix = results[dataset]['S2']
        pd.DataFrame(matrix, columns=x_vars, index=x_vars).to_csv(f'interactions/{dataset}_interaction_matrix_values.csv')
        confidence_matrix = results[dataset]['S2_conf']
        pd.DataFrame(confidence_matrix, columns=x_vars, index=x_vars).to_csv(f'interactions/{dataset}_interaction_matrix_conf.csv')

    # Analyse for possible interactions
    out_df['possible_interaction']= False
    for i, row in out_df.iterrows():
        min_ST = row['st_value'] - row['st_conf']
        max_S1 = row['s1_value'] + row['s1_conf']
        if max_S1 < min_ST:
            out_df.possible_interaction[i] = True

    out_df.to_csv('param_sensitivity_results.csv', index=False)
    print('Complete.')