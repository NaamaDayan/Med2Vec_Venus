import ast
import pickle
import pandas as pd


def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def decode_data(data, vocab):
    decoded_data = []
    count = 0
    for visits in data:
        count += 1
        decoded_visits = []
        for code in visits:
            if code == -1:
                decoded_visits.append(['-1'])
            else:
                diagnosis = vocab.iloc[code]['Diagnosis']
                # print('diagnosis is:',diagnosis[0])
                decoded_visits.append(diagnosis)
        decoded_data.append(decoded_visits)
    return decoded_data

def convert_voacb(vocab,col):
    # access only to first value as no need in more for now
    vocab[col] = vocab[col].apply(lambda x: ast.literal_eval(x)[0])
    return vocab

def saving_decoded_data(decoded_data,filename):
    with open(filename, 'wb') as f:
        # load the data from the file into memory
        pickle.dump(decoded_data, f)

def get_top_50_embeddings(data):
    # Flatten the list of lists into a 1D list
    flattened_data = [x for row in data for x in row]

    # Remove any occurrences of -1
    filtered_data = [x for x in flattened_data if x != -1]

    # Count the number of occurrences of each integer
    integer_counts = {}
    for integer in filtered_data:
        if integer in integer_counts:
            integer_counts[integer] += 1
        else:
            integer_counts[integer] = 1

    # Sort the integers by frequency in the data
    sorted_integers = sorted(integer_counts.items(), key=lambda x: x[1], reverse=True)

    # Get the top 50 unique integers (or all of them if there are less than 50)
    top_integers = [x[0] for x in sorted_integers][:50]

    # Convert each integer to a list with a single item
    top_integers_list = [[i] for i in top_integers]

    return top_integers_list






if __name__ == '__main__':
    vocab = pd.read_csv('./data/vocab_med2vec_new_large_data_no_rare_diagnosis')
    print(len(vocab))
    data = load_data('data/data_large/med2vec.seqs')
    # # print("data size is:",len(data))
    # # print(vocab[:10])
    # vocab = convert_voacb(vocab,'Diagnosis')
    # vocab.to_csv('originaL_vocab.csv', index=False)
    print(vocab[:10])
    # # print(data[:10])
    top_words_to_decode = get_top_50_embeddings(data)
    print("top words are:",top_words_to_decode)
    saving_decoded_data(top_words_to_decode,'./data/top_frequency_embeddings.seqs')

    # # print(top_words_to_decode)
    decoded_data = decode_data(data, vocab)
    # # print(decoded_data[:10])
    # top_diagnoses_similarity_vocab = load_data('data/top_frequency_similarity_diagnosis.seqs')
    # sorted_vocab = sorted(top_diagnoses_similarity_vocab.items(), key=lambda x: x[0], reverse=True)
    # # sorted_vocab is a list of tuples sorted based on keys in descending order
    # decoded_data_flat = [val[0] for val in decoded_data]
    # # print(decoded_data_flat[:10])
    # # print(top_diagnoses_similarity_vocab)
    # new_vocab = {decoded_data_flat[t]: top_diagnoses_similarity_vocab[t] for t in top_diagnoses_similarity_vocab}
    # for key in new_vocab.keys():
    #     new_vocab[key] = [item for sublist in decode_data([new_vocab[key]], vocab) for item in sublist]
    # # new_dict = {t[0]: t[1] for t in new_vocab}
    # print(new_vocab)
    # with pd.ExcelWriter('top_diagnosis_similarities.xlsx') as writer:
    #     for key, value in new_vocab.items():
    #         sheet_name = key.replace('/', '_').replace('\\', '_').replace('?', '_').replace('*', '_') \
    #             .replace('[', '_').replace(']', '_').replace(':', '_')
    #         df = pd.DataFrame(value)
    #         df.index.name = key  # set the index name to the key
    #         df.to_excel(writer, sheet_name=sheet_name, startrow=1,
    #                     header=False)  # write to sheet, start at row 2 and exclude headers
    #         writer.sheets[sheet_name].cell(row=1, column=1).value = key
            # saving_decoded_data(decoded_data,'./data/decoded_data')
    # embeddings = load_data('./data/top_embedding')
    #
    # # now decoding the enbeddings to see if related:
    # decoded_embeddings = (embeddings, vocab)
    # print(decoded_embeddings)
    saving_decoded_data(decoded_data,'./data/decoded_large_data')

    decoded_data = load_data('./data/decoded_large_data')
    print("data is:",data[:10])
    print(decoded_data[:10])
    print(len(decoded_data))


