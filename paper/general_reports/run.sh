
python3 ../codes/plot_solution_features.py --inputfile ../Genetic_algo/output_yelp/random_cc_output4/features/selected_solutions_clustering_coeff_0/merged.csv --outputdir ../Genetic_algo/output_yelp/random_cc_output4/feature_plots --opti_func cc

python3 ../codes/test.py --dataset yelp --imagedir /users/sghosh15/Genetic_algo/output_yelp/random_cc_output4 --featuredir /users/sghosh15/Genetic_algo/output_yelp/random_cc_output4/feature_plots

mv *.pdf pdfs



