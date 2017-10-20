
import os
import pandas as pd
import numpy as np
import pickle
import knpackage.toolbox as kn

from   sklearn import linear_model


def run_LassoCV(run_parameters):
    """
    """

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

    min_alpha     = run_parameters['min_alpha'    ]
    max_alpha     = run_parameters['max_alpha'    ]
    n_alpha       = run_parameters['n_alpha'      ]
    fit_intercept = run_parameters['fit_intercept']
    normalize     = run_parameters['normalize'    ]
    max_iter      = run_parameters['max_iter'     ]
    tol           = run_parameters['tolerance'    ]
    alphas        = np.linspace( min_alpha, max_alpha, num=n_alpha )

    reg_model     = linear_model.LassoCV( alphas        = alphas
                                        , fit_intercept = fit_intercept
                                        , normalize     = normalize
                                        , max_iter      = max_iter
                                        , tol           = tol
                                        , cv            = 5           )

    reg_model.fit( gene_mat.T, sign_mat)

    filename      = os.path.join(run_parameters['results_directory'], 'lasso_model.pkl') 
    pickle.dump(reg_model, open(filename, 'wb'))

    predict_mat   = reg_model.predict(test_mat.T)
    predict_df    = pd.DataFrame(predict_mat.T, index=row_names, columns=['prediction'])

    write_predict_data(predict_df, run_parameters)
    

def run_ElasticNetCV(run_parameters):
    """
    """

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

    min_l1        = run_parameters['min_l1'       ]
    max_l1        = run_parameters['max_l1'       ]
    n_l1          = run_parameters['n_l1'         ]
    min_alpha     = run_parameters['min_alpha'    ]
    max_alpha     = run_parameters['max_alpha'    ]
    n_alpha       = run_parameters['n_alpha'      ]
    fit_intercept = run_parameters['fit_intercept']
    eps           = run_parameters['eps'          ]    
    normalize     = run_parameters['normalize'    ]
    max_iter      = run_parameters['max_iter'     ]
    tol           = run_parameters['tolerance'    ]

    l1_ratio      = np.linspace( min_l1   , max_l1   , num = n_l1    )
    alphas        = np.linspace( min_alpha, max_alpha, num = n_alpha )

    reg_model = linear_model.ElasticNetCV( l1_ratio      = l1_ratio
                                         , alphas        = alphas
                                         , fit_intercept = fit_intercept
                                         , eps           = eps
                                         , normalize     = normalize
                                         , max_iter      = max_iter
                                         , tol           = tol
                                         , cv            = 5        )

    reg_model.fit(gene_mat.T, sign_mat)

    filename    = os.path.join(run_parameters['results_directory'], 'elastic_net_model.pkl') 
    pickle.dump(reg_model, open(filename, 'wb'))

    predict_mat = reg_model.predict(test_mat.T)
    predict_df  = pd.DataFrame(predict_mat.T, index=row_names, columns=['prediction'])

    write_predict_data(predict_df, run_parameters)


def write_predict_data(predict_df, run_parameters):
    """
    """

    test_file          = run_parameters['test_spreadsheet_name_full_path']
    results_directory  = run_parameters['results_directory'              ]
    method             = run_parameters['method'                         ]

    _, output_file     = os.path.split(test_file)
    output_file, _     = os.path.splitext(output_file)
    output_file        = os.path.join(results_directory, output_file + '_' + method)
    output_file        = kn.create_timestamped_filename(output_file) + '.tsv'

    predict_df.to_csv(output_file, sep='\t', header=True, index=True, float_format='%g')

