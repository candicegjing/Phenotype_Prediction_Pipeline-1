# this is the main function of PPP

def LassoCV(run_parameters):
    from phenotype_prediction_toolbox import run_LassoCV
    run_LassoCV(run_parameters)
    

def ElasticNetCV(run_parameters):
    from phenotype_prediction_toolbox import run_ElasticNetCV
    run_ElasticNetCV(run_parameters)

SELECT = { 'ElasticNetCV': ElasticNetCV 
         , 'LassoCV'     : LassoCV     }

def main():
    import sys
    from knpackage.toolbox import get_run_directory_and_file
    from knpackage.toolbox import get_run_parameters

    run_directory, run_file = get_run_directory_and_file(sys.argv)
    run_parameters = get_run_parameters(run_directory, run_file)
    SELECT[run_parameters['method']](run_parameters)


if __name__ == "__main__":
    main()
