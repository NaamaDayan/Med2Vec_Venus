import collections
import itertools
import pickle
import random

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import Tensor

from model.metric import recall_k
from model.med2vec import Med2Vec
from sklearn.metrics.pairwise import cosine_similarity
pd.set_option('display.max_columns', None)
pd.set_option('display.width',None)
pd.set_option('display.max_rows', None)
torch.set_default_tensor_type(torch.cuda.FloatTensor)
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def load_best_model(save_path):
    """
    Loads the best saved model from a directory.

    Args:
    model_class (type): The class of the model to load.
    save_dir (str): The directory containing the saved model.
    device (torch.device): The device to use for computation.

    Returns:
    model (torch.nn.Module): The loaded model.
    """
    checkpoint = torch.load(save_path)
    model_args = checkpoint['model_args']
    data_loader = checkpoint['data_loader']
    model = Med2Vec(**model_args)

    model.load_state_dict(checkpoint['model_state_dict'])
    best_val_loss = checkpoint['best_val_loss']
    data_for_testing = checkpoint['validation_data_for_testing']
    emb_w = checkpoint['embeddings']
    return model, best_val_loss, data_for_testing,data_loader,model_args,emb_w

def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def saving_decoded_data(decoded_data, filename):
    with open(filename, 'wb') as f:
        # load the data from the file into memory
        pickle.dump(decoded_data, f)

def jaccard_similarity_score(tensor1, tensor2, vocab_size, device):
    # Convert indices to binary tensors
    binary_tensor1 = torch.zeros((1, vocab_size), dtype=torch.long, device=device)
    binary_tensor1[0, torch.LongTensor(tensor1.to(device).long())] = 1
    binary_tensor2 = torch.zeros((1, vocab_size), dtype=torch.long, device=device)
    binary_tensor2[0, torch.LongTensor(tensor2.to(device).long())] = 1
    # Calculate Jaccard similarity coefficient
    intersection = (binary_tensor1 * binary_tensor2).sum()
    union = binary_tensor1.sum() + binary_tensor2.sum() - intersection
    similarity = intersection / union

    return similarity.item()

def visits_generator(vocab_size, visit_min_diagnosis, visit_max_diagnosis, similarity):
    # Calculate the number of diagnoses in each visit
    visit1_length = random.randint(visit_min_diagnosis, visit_max_diagnosis)
    visit2_length = random.randint(visit_min_diagnosis, visit_max_diagnosis)


    visit1_length = max(visit1_length, visit2_length)
    visit2_length = visit1_length
    shared_diagnoses = round(min(visit1_length, visit2_length) * similarity)

    # Generate the shared diagnoses
    shared_diagnoses_indices = random.sample(range(vocab_size), shared_diagnoses)

    # Generate the diagnoses for visit1
    visit1_indices = []
    visit1_binary = torch.zeros(vocab_size)

    for i in range(visit1_length):
        if i < shared_diagnoses:
            # Add a shared diagnosis
            visit1_indices.append(shared_diagnoses_indices[i])
            visit1_binary[shared_diagnoses_indices[i]] = 1
        else:
            # Add a random diagnosis that is not in the shared diagnoses
            random_index = random.choice([i for i in range(vocab_size) if i not in shared_diagnoses_indices])
            visit1_indices.append(random_index)
            visit1_binary[random_index] = 1

    # Generate the diagnoses for visit2
    visit2_indices = []
    visit2_binary = torch.zeros(vocab_size)
    for i in range(visit2_length):
        if i < shared_diagnoses:
            # Add a shared diagnosis
            visit2_indices.append(shared_diagnoses_indices[i])
            visit2_binary[shared_diagnoses_indices[i]] = 1
        else:
            # Add a random diagnosis that is not in the shared diagnoses
            random_index = random.choice([i for i in range(vocab_size) if i not in shared_diagnoses_indices])
            visit2_indices.append(random_index)
            visit2_binary[random_index] = 1

    # Convert binary tensors to one-hot tensors
    visit1_one_hot = torch.unsqueeze(visit1_binary, 0)
    visit2_one_hot = torch.unsqueeze(visit2_binary, 0)
    return visit1_one_hot, visit2_one_hot


def visits_embedded(model, visit):
    probits, emb_w = model(visit)
    return probits, emb_w

def get_top_diagnosis(probits,emb_w,k):
    _, tk = torch.topk(probits, k)
    return tk, emb_w



def sanity_checks(model,iterations,model_args,k,precentage_diff):
    mean_similarity = []
    mean_emb_cosine_similarity = []
    for i in range(1, precentage_diff+1):
        current_similarity = 0
        curren_emb_similarity = 0
        for iter in range(iterations):
            shared_diagnoses = i/precentage_diff
            visit_1, visit_2 = visits_generator(model_args['icd9_size'], 3, 10, shared_diagnoses)
            probits_1,emb = visits_embedded(model, visit_1)
            visit_1_top_k, _ = get_top_diagnosis(probits_1, emb, k)
            probits_2,_ = visits_embedded(model, visit_2)
            visit_2_top_k, _ = get_top_diagnosis(probits_2,emb,k)
            # getting metrices values
            emb1 = visit_1 @ emb.T
            emb2 = visit_2 @ emb.T
            # print("shapes are:",emb1.shape,emb.shape)
            similarity_score = jaccard_similarity_score(visit_1_top_k, visit_2_top_k,model_args['icd9_size'],model.device)
            emb_similarity_score = cosine_similarity(emb1.detach().numpy(), emb2.detach().numpy())
            # similarity_score_original = jaccard_similarity_score(visit_1,visit_2,model_args['icd9_size'],model.device)
            current_similarity += similarity_score
            curren_emb_similarity += emb_similarity_score
        mean_similarity.append(current_similarity/iterations)
        mean_emb_cosine_similarity.append(curren_emb_similarity/iterations)
    return mean_similarity, mean_emb_cosine_similarity


def recall_test_on_validation_data_on_best_model(model,test_data_indices,data_loader):
    mean_recall_r = 0
    mean_recall_t = 0
    # Evaluate the model on validation data using the recall_k function from metric.py
    model.eval()
    batch_count = 0
    emb_w = None
    with torch.no_grad():
        for batch_idx in test_data_indices:
            (x, ivec, jvec, mask, d) = data_loader.get_batch_data(batch_idx)
            batch_count += 1
            if len(x) < data_loader.batch_size:
                continue
            data, ivec, jvec, mask, d = x.to(model.device), ivec.to(model.device), jvec.to(model.device), mask.to(
                model.device), d.to(model.device)
            # print("shapes are:",x.shape)
            probits,emb_w = model(data.float(), d)
            recall_r, recall_t = recall_k(probits, data, mask, k=5, window=4)
            mean_recall_r += recall_r
            mean_recall_t += recall_t

        mean_recall_r /= batch_count
        mean_recall_t /= batch_count
    return mean_recall_r,mean_recall_t,emb_w

def get_top_k_indices(embedding, k=10, n=20):
    # Get the top k indices for each row in the embedding
    top_k_values, top_k_indices = torch.topk(embedding, k=k, dim=1)

    # Get the mean values for the top k indices for each row
    mean_top_k_values = torch.mean(top_k_values, dim=1)

    # Get the indices of the top n rows with the highest mean top k values
    top_n_indices = torch.topk(mean_top_k_values, k=n)[1]

    # Get the top k indices for the top n rows
    top_k_indices_for_top_n_rows = top_k_indices[top_n_indices]

    # Convert the tensor to a list of lists
    top_k_indices_for_top_n_rows = top_k_indices_for_top_n_rows.tolist()
    return top_k_indices_for_top_n_rows


def create_vocab_matching(top_indices, vocab_df):
    # Create an empty DataFrame with the desired columns
    vocab_matching = pd.DataFrame(columns=['Subcategory designation percentage',
                                            'Category designation percentage',
                                            'Block percentage',
                                            'Chapter percentage'])

    # Iterate over each key-value pair in top_indices
    for key, indices in top_indices.items():
        # Create a DataFrame of the related indices from vocab_df
        # print(indices)
        related_df = vocab_df[vocab_df['index'].isin(indices)]
        print(vocab_df.iloc[indices])
        print(indices)

        # Calculate the percentage of matching values in each column
        subcategory_count = (related_df['Subcategory designation'] == vocab_df.loc[key, 'Subcategory designation']).mean()
        category_count = (related_df['Category designation'] == vocab_df.loc[key, 'Category designation']).mean()
        block_count = (related_df['Block'] == vocab_df.loc[key, 'Block']).mean()
        chapter_count = (related_df['Chapter'] == vocab_df.loc[key, 'Chapter']).mean()
        # Divide the counts by the total percentages and add as a new row to vocab_matching
        vocab_matching.loc[key] = [subcategory_count,
                                   category_count,
                                   block_count,
                                   chapter_count]
    return vocab_matching

def get_top_50_words(data):
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

def plot_histograms(df, columns, bins ,k=3):
    # remove rows that have 0 in all columns
    # df = df.loc[(df != 0).any(axis=1)]
    num_columns = len(columns)
    fig, axes = plt.subplots(nrows=1, ncols=num_columns, figsize=(num_columns * 6, 5))
    plt.subplots_adjust(wspace=0.3)
    mean_score_for_chapter = np.mean(df['Chapter percentage'])
    for i, column in enumerate(columns):
        # create histogram of column values
        axes[i].hist(df[column], bins=bins)

        # calculate and display mean value of column
        mean_value = np.mean(df[column])
        axes[i].axvline(mean_value, color='r', linestyle='dashed', linewidth=1)
        axes[i].text(mean_value + (mean_value * 0.05), 0.9 * axes[i].get_ylim()[1], f'Mean = {mean_value:.2f}',
                     fontsize=12, color='r', ha='left')

        # set plot title and axis labels
        axes[i].set_title(f'{column}, k={k}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Count')
    plt.show()
    fig.savefig(f'histogram_{column}_{k}_min_similarity_med2vec_optuna_testing.png')
    plt.close()

    return mean_score_for_chapter

def get_top_similarity_diagnosis(matrix, top_embeddings,device,num_top=10):
    top_indices = {}
    for i, (key, emb) in enumerate(top_embeddings.items()):
        valid_indices = [j for j in range(len(matrix)) if j != key]
        scores = [(j, cosine_similarity(emb.detach().cpu().reshape(1, -1),
                                        matrix[j].detach().cpu().reshape(1, -1), len(matrix)))
                  for j in valid_indices]
        top_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        # cutting off at 100 for now as we don't look anyways at more than that when testing the model
        top_indices[key] = [j for j, score in top_scores if score > 0][:100]
    return top_indices

def save_words_and_similarities(data,emb_w,device):
    # extracting the most frequent diagnosis in the data
    top_50_words = get_top_50_words(data)
    # it's a list of list of words because of the convention of the data so
    top_50_words = [index for sublist in top_50_words for index in sublist]
    # creating dict of the top_words as keys and their embeddings as values
    top_frequency_embeddings = {index: emb_w.T[index] for index in top_50_words}
    # for each word(diagnosis) we want to get the most similar words to it in the embeddings space
    top_similarity_diagnosis = get_top_similarity_diagnosis(emb_w.T, top_frequency_embeddings,device=device)
    with open("top_50_words.pkl","wb") as f:
        pickle.dump(top_50_words, f)
    with open("top_frequency_embeddings.pkl","wb") as f:
        pickle.dump(top_frequency_embeddings, f)
    with open("top_similarity_diagnosis.pkl", "wb") as f:
        pickle.dump(top_similarity_diagnosis, f)

def load_words_and_similarities():
    top_50_words = load_data('top_50_words.pkl')
    top_frequency_embeddings = load_data('top_frequency_embeddings.pkl')
    top_similarity_diagnosis = load_data('top_similarity_diagnosis.pkl')
    return top_50_words, top_frequency_embeddings, top_similarity_diagnosis

def get_top_words_in_dict(top_similarity_diagnosis_cutoff):
    # Flatten the dictionary into a single list of all values
    all_values = list(itertools.chain.from_iterable(top_similarity_diagnosis_cutoff.values()))

    # Count the frequency of each value in the list
    value_counts = collections.Counter(all_values)

    # Get the top 5 most common values and their frequency
    top_values = value_counts.most_common(5)

    # Print the results
    print("Top 5 most common values:")
    for value, count in top_values:
        print(f"{value}: {count}")



def main():
    save_path = './saved/models/best_model.pt'
    model, val_loss, test_data_indices,data_loader, model_args,emb_w = load_best_model(save_path)
    print("model args are:", model_args)
    vocab_df = pd.read_csv('encoded_vocab_with_hierarchy_large_data.csv')
    # print("embeddings shape" ,emb_w.shape)
    data = load_data('data/data_large/med2vec.seqs')
    # save_words_and_similarities(data,emb_w,model.device)
    top_50_words, top_frequency_embeddings, top_similarity_diagnosis = load_words_and_similarities()
    # testing the model for the code embeddings
    mean_scores_per_run = []
    top_rows  = [1,3,5,10]
    for k in top_rows:
        # cutting of the top_similarity pair word as we want to test it only the top similar
        top_similarity_diagnosis_cutoff = {}
        for key, value in top_similarity_diagnosis.items():
            top_similarity_diagnosis_cutoff[key] = value[:k]
        # finding matching between the word and her top k most similar words in the data
        hierarchy_intersections = create_vocab_matching(top_similarity_diagnosis_cutoff,vocab_df)
        # plotting the histograms over the top 50 words and for different top k

        mean_score_for_chapter = plot_histograms(hierarchy_intersections, hierarchy_intersections.columns, 20,k)
        print("mean score for k=",k," is: ",mean_score_for_chapter)
        mean_scores_per_run.append(mean_score_for_chapter)
    return sum(mean_scores_per_run) / len(mean_scores_per_run)


    # # testing the model for the visit embeddings
    #
    # iterations = 1000
    # k = 10
    # precentage_diff = 10
    #
    # # sanity check that checks for generated visits with on going increases of their joined diagnosis that the similar
    # # the visits so has the jaccard similarity score in the top k values in the model are similar
    #
    # mean_similarity,mean_emb_similarity = sanity_checks(model, iterations, model_args, k, precentage_diff)
    # print("we got top k  mean similarity of:",mean_similarity)
    # print("we got mean embeddings similarity of:",mean_emb_similarity)
    #
    # # recalls of the model on the validation data
    # mean_recall_r, mean_recall_t,emb_w = recall_test_on_validation_data_on_best_model(model, test_data_indices, data_loader)
    # print("recalls for best model are:",mean_recall_r, mean_recall_t)
    # # indices_list = get_top_k_indices(emb_w, 10, 20)
    # # # saving the indices lists that will be uploaded in the decoder:
    # # with open("./data/top_embedding", 'wb') as f:
    # #     # load the data from the file into memory
    # #     pickle.dump(indices_list, f)


if __name__ == '__main__':
    main()