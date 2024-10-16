from tabulate import tabulate

def evaluate_models(model_results):
    # Prepare data for tabulation
    models = list(model_results.keys())
    train_scores = [results[0] for results in model_results.values()]  # Assuming train score is the first element
    test_scores = [results[1] for results in model_results.values()]   # Assuming test score is the second element
    
    # Prepare the table data
    table_data = [
        ["Model"] + models,
        ["Train Score"] + [f"{score:.4f}" for score in train_scores],
        ["Test Score"] + [f"{score:.4f}" for score in test_scores]
    ]
    
    # Print the table
    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))