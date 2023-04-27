import pickle

def fixing_duplications(duplicate_filename,original_filename):
    # dumping the original data back from duplicate
    with open(duplicate_filename, 'rb') as f:
        # load the data from the file into memory
        duplicate_data = pickle.load(f)
    with open(original_filename, 'wb') as f:
        # load the data from the file into memory
        pickle.dump(duplicate_data, f)

def multiplying_data(duplicate_filename,original_filename):
    with open(original_filename, 'rb') as f:
        # load the data from the file into memory
        data_original = pickle.load(f)

    data_multiplied = []
    for i in range(10):
        data_multiplied.extend(data_original)

    with open(duplicate_filename, 'wb') as f:
        # load the data from the file into memory
        pickle.dump(data_multiplied, f)

def small_training_data(small_data_filename,original_filename,sample_factor):
    with open(original_filename, 'rb') as f:
        # load the data from the file into memory
        data_original = pickle.load(f)

    data_sampaled = data_original[:int(len(data_original)//sample_factor)]
    print("data_smapled:",data_sampaled[:2])
    # vocab = {}
    # vocab_size = 0
    # for visit in data_sampaled:
    #     for diagnosis in visit:
    #         if diagnosis not in vocab:
    #             vocab_size+=1
    #         vocab[diagnosis] = vocab_size - 1
    # Find the unique indices in the input data
    unique_indices = list(set([item for sublist in data_sampaled for item in sublist]))

    # Create a dictionary mapping each unique index to a smaller index
    vocab = {unique_indices[i]: i for i in range(len(unique_indices))}
    max_diganosis = 0

    # Iterate over the input data and replace each index with its corresponding value in the vocabulary
    for i in range(len(data_sampaled)):
        for j in range(len(data_sampaled[i])):
            data_sampaled[i][j] = vocab[data_sampaled[i][j]]
            if data_sampaled[i][j] > max_diganosis:
                max_diganosis = data_sampaled[i][j]

    # for j in range(len(data_sampaled)):
    #     for i in range(j):
    #         print(data_sampaled[j][i])
    #         data_sampaled[j][i] = vocab[data_sampaled[j][i]]
    #         # if data_sampaled[j][i] > max_diganosis:
    #         #     max_diganosis = data_sampaled[j][i]
    print(vocab)
    print("vocab_size is:",len(vocab),max_diganosis)
    with open(small_data_filename, 'wb') as f:
        # load the data from the file into memory
        pickle.dump(data_sampaled, f)


if __name__ == '__main__':
    filename = './data/med2vec.seqs'
    small_data_filename = "./data/small_sample/med2vec.seqs"
    small_training_data(small_data_filename,filename,4)
    # open the file in text mode
    # multiplying_data("./data/duplicates/med2vec.seqs",'data/med2vec.seqs')
    #
    # # data_original = data[:(len(data)//10)]
    # # filename = './data/med2vec.seqs'
    # # with open(filename, 'wb') as f:
    # #     pickle.dump(data_original, f)
    #
    # with open('./data/duplicates/med2vec.seqs', 'rb') as f:
    #     # load the data from the file into memory
    #     data_multiplied = pickle.load(f)
    #
    # with open('data/med2vec.seqs', 'rb') as f:
    #     # load the data from the file into memory
    #     data_original = pickle.load(f)
    #
    # print("new data len and first 10:",len(data_multiplied),data_multiplied[:10])
    # print("data len and first 10:",len(data_original),data_original[:10])
