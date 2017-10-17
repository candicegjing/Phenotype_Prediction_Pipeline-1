
import os
import pandas as pd
from   sklearn import linear_model

import knpackage.toolbox as kn

def run_lasso_predict(run_parameters):

    gene_file = run_parameters['spreadsheet_name_full_path'     ]
    sign_file = run_parameters['response_name_full_path'        ]
    test_file = run_parameters['test_spreadsheet_name_full_path']

    gene_df   = kn.get_spreadsheet_df(gene_file)
    sign_df   = kn.get_spreadsheet_df(sign_file)
    test_df   = kn.get_spreadsheet_df(test_file)

    row_names = test_df.columns

    gene_mat  = gene_df.values
    sign_mat  = sign_df.values[0]
    test_mat  = test_df.values

    reg_model          = linear_model.Lasso()
    response_predict   = reg_model.fit( gene_mat.T, sign_mat).predict(test_mat.T)
    predict_df         = pd.DataFrame(response_predict.T, index=row_names, columns=['predict'])
    write_predict_data(predict_df, run_parameters)
    

def run_elastic_predict(run_parameters):

    gene_file = run_parameters['spreadsheet_name_full_path'     ]
    sign_file = run_parameters['response_name_full_path'        ]
    test_file = run_parameters['test_spreadsheet_name_full_path']

    gene_df   = kn.get_spreadsheet_df(gene_file)
    sign_df   = kn.get_spreadsheet_df(sign_file)
    test_df   = kn.get_spreadsheet_df(test_file)

    row_names = test_df.columns

    gene_mat  = gene_df.values
    sign_mat  = sign_df.values[0]
    test_mat  = test_df.values

    reg_model         = linear_model.ElasticNetCV()
    response_predict  = reg_model.fit( gene_mat.T, sign_mat).predict(test_mat.T)
    predict_df        = pd.DataFrame(response_predict.T, index=row_names, columns=['predict'])
    write_predict_data(predict_df, run_parameters)


def write_predict_data(predict_df, run_parameters):

    test_spreadsheet_name_full_path = run_parameters['test_spreadsheet_name_full_path']
    results_directory               = run_parameters['results_directory']
    _, output_file_name             = os.path.split(test_spreadsheet_name_full_path)

    output_file_name, _             = os.path.splitext(output_file_name)
    output_file_name                = os.path.join(results_directory, output_file_name)
    output_file_name                = kn.create_timestamped_filename(output_file_name) + '.tsv'

    predict_df.to_csv(output_file_name, sep='\t', header=True, index=True)

