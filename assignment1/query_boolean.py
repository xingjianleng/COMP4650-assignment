import pickle


def intersect_query(doc_list1, doc_list2):
    # in your run_boolean_query implementation
    # for full marks this should be the O(n + m) intersection algorithm for sorted lists
    # using data structures such as sets or dictionaries in this function will not score full marks
    rtn = []
    ptr1, ptr2 = 0, 0
    while ptr1 < len(doc_list1) and ptr2 < len(doc_list2):
        if doc_list1[ptr1] == doc_list2[ptr2]:
            rtn.append(doc_list1[ptr1])
            ptr1 += 1
            ptr2 += 1
        elif doc_list1[ptr1] < doc_list2[ptr2]:
            ptr1 += 1
        else:
            ptr2 += 1
    # no intersection if one list run out of elements
    return rtn


def union_query(doc_list1, doc_list2):
    # in your run_boolean_query implementation
    # for full marks this should be the O(n + m) union algorithm for sorted lists
    # using data structures such as sets or dictionaries in this function will not score full marks
    rtn = []
    ptr1, ptr2 = 0, 0
    while ptr1 < len(doc_list1) and ptr2 < len(doc_list2):
        if doc_list1[ptr1] == doc_list2[ptr2]:
            rtn.append(doc_list1[ptr1])
            ptr1 += 1
            ptr2 += 1
        while ptr1 < len(doc_list1) and ptr2 < len(doc_list2) and doc_list1[ptr1] < doc_list2[ptr2]:
            rtn.append(doc_list1[ptr1])
            ptr1 += 1
        while ptr1 < len(doc_list1) and ptr2 < len(doc_list2) and doc_list1[ptr1] > doc_list2[ptr2]:
            rtn.append(doc_list2[ptr2])
            ptr2 += 1
    # if either list have remaining elements, append them to return list
    while ptr1 < len(doc_list1):
        rtn.append(doc_list1[ptr1])
        ptr1 += 1
    while ptr2 < len(doc_list2):
        rtn.append(doc_list2[ptr2])
        ptr2 += 1
    return rtn


def run_boolean_query(query, index):
    """Runs a boolean query using the index.

    Args:
        query (str): boolean query string
        index (dict(str : list(tuple(int, int)))): The index aka dictionary of posting lists

    Returns:
        list(int): a list of doc_ids which are relevant to the query
    """
    ptr = 1
    query_lst = query.split()
    result_lst = [doc_id for doc_id, _ in index[query_lst[0]]]
    while ptr < len(query_lst):
        connective = query_lst[ptr]
        assert connective in ("AND", "OR")
        next_lst = [doc_id for doc_id, _ in index[query_lst[ptr + 1]]]
        if connective == "AND":
            result_lst = intersect_query(result_lst, next_lst)
        else:
            result_lst = union_query(result_lst, next_lst)
        ptr += 2
    return result_lst


# load the stored index
(index, doc_freq, doc_ids, num_docs) = pickle.load(open("stored_index.pik", "rb"))

print("Index length:", len(index))
if len(index) != 906290:
    print("Warning: the length of the index looks wrong.")
    print("Make sure you are using `process_tokens_original` when you build the index.")
    raise Exception()

# the list of queries asked for in the assignment text
queries = [
    "Welcoming",
    "Australasia OR logistic",
    "heart AND warm",
    "global AND space AND wildlife",
    "engine OR origin AND record AND wireless",
    "placement AND sensor OR max AND speed"
]

# run each of the queries and print the result
ids_to_doc = {v: k for k, v in doc_ids.items()}
for q in queries:
    res = run_boolean_query(q, index)
    res.sort(key=lambda x: ids_to_doc[x])
    print(q)
    for r in res:
        print(ids_to_doc[r])
