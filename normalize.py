def normalize(data, column_index):
    X = data
    column = [x[column_index] for x in X]
    min_column = min(column)
    max_column = max(column)
    normalized_column = []

    for i in range(len(column)):
        normalized_column.append((column[i] - min_column)/(max_column - min_column))

    for x in range(len(X)):
        X[x][column_index] = round(normalized_column[x], 4)

    return X