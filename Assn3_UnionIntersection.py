A = {'x1': 0.2, 'x2': 0.4, 'x3': 0.6, 'x4': 0.8}
B = {'x1': 0.3, 'x2': 0.5, 'x3': 0.7, 'x4': 0.9}

R = {('x1', 'y1'): 0.2, ('x1', 'y2'): 0.4, ('x2', 'y1'): 0.6, ('x2', 'y2'): 0.8}
S = {('x1', 'y1'): 0.3, ('x1', 'y2'): 0.5, ('x2', 'y1'): 0.7, ('x2', 'y2'): 0.9}

def fuzzy_union(A, B):
    union = {}
    for key in A.keys():
        union[key] = max(A[key], B[key])
    return union

# Example usage
union_result = fuzzy_union(A, B)
print("Union:", union_result)


def fuzzy_intersection(A, B):
    intersection = {}
    for key in A.keys():
        intersection[key] = min(A[key], B[key])
    return intersection

# Example usage
intersection_result = fuzzy_intersection(A, B)
print("Intersection:", intersection_result)


def fuzzy_complement(A):
    complement = {}
    for key in A.keys():
        complement[key] = 1 - A[key]
    return complement

# Example usage
complement_result_A = fuzzy_complement(A)
complement_result_B = fuzzy_complement(B)
print("Complement A:", complement_result_A)
print("Complement B:", complement_result_B)


def fuzzy_difference(A, B):
    difference = {}
    for key in A.keys():
        difference[key] = min(A[key], 1 - B[key])
    return difference

# Example usage
difference_result = fuzzy_difference(A, B)
print("Difference:", difference_result)

def cartesian_product(A,B):
    product={}
    for key1 in A.keys():
        for key2 in B.keys():
            product[(key1,key2)]=min(A[key1],B[key2])
    return product

R1 = cartesian_product(A,B)
print(" Cartesian Product of A and B", R1)
R2 = cartesian_product(B,A)
print(" Cartesian Product of B and A", R2)

composition = {}
for (a, b) in R1:
    for (b2, c) in R2:
        if b == b2:  # Matching common element (b)
            key = (a, c)
            composition[key] = max(composition.get(key, 0), min(R1[(a, b)], R2[(b2, c)]))

print("Max-Min Composition of R1 and R2:", composition)