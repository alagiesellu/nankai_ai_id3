from id3 import construct_and_evaluate

for x in range(10):
    results, iteration = construct_and_evaluate(data_source="dataset/Tennis.csv", label="Play", iteration=1000)

    print(results)
