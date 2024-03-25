epsilon = 10**(-6)


def read_a_file_dict(file_path):
    matrix = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        n = int(lines[0].strip())
        for line in lines[1:]:
            values = line.strip().split(',')
            value = float(values[0])
            i = int(values[1])
            j = int(values[2]) if values[2] != '' else None
            if value != 0:
                if i not in matrix:
                    matrix[i] = {}
                if j is not None:
                    if j in matrix[i]:
                        matrix[i][j] += value
                    else:
                        matrix[i][j] = value
    return n, matrix


def sum_of_matrix(a, b):
    matrix_result = {}
    for key in a.keys():
        matrix_result[key] = {}

        for key1 in a[key].keys():
            if key1 in matrix_result[key].keys():
                matrix_result[key][key1] += a[key][key1]
            else:
                matrix_result[key][key1] = a[key][key1]

        for key1 in b[key].keys():
            if key1 in matrix_result.get(key, {}):
                matrix_result[key][key1] += b[key][key1]
            else:
                matrix_result[key][key1] = b[key][key1]

    matrix_result = {key: {k: v for k, v in value.items() if v != 0} for key, value in matrix_result.items()}

    return matrix_result


def check_result(matrix_result, matrix):
    for key in matrix_result.keys():
        for key1 in matrix_result[key].keys():
            if abs(matrix_result[key][key1] - matrix[key][key1]) >= epsilon:
                raise Exception("Abs value is bigger than epsilon")

    return True


file_path1 = "a.txt"
file_path2 = "b.txt"
file_path3 = "aplusb.txt"
n1, matrix1 = read_a_file_dict(file_path1)
n2, matrix2 = read_a_file_dict(file_path2)
n3, aplusb = read_a_file_dict(file_path3)

print("Dimension n =", n1)
sum_result = sum_of_matrix(matrix1, matrix2)
if check_result(sum_result, aplusb):
    print("Sum of a.txt + b.txt is aplusb.txt")
