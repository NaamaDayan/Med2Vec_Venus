import ast
import subprocess
import sys

# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])
#
# subprocess.call(["pip", "install","openpyxl"])

# install('--up')


import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
pd.set_option('display.max_columns', None)
pd.set_option('display.width',None)
pd.set_option('display.max_rows', None)


# def calc_code_vectors_similarities(vocab_df, top_rows):
#     vocab = vocab_df.index.values
#     vocab_embeddings = vocab_df.values
#
#     # compute cosine similarities
#     cosine_similarities = cosine_similarity(vocab_embeddings)
#     np.fill_diagonal(cosine_similarities, 0)
#
#     # get top 10 indices for each vector
#     top_indices = {}
#     mean_top_indices = {}
#     for i in range(len(vocab)):
#         top_indices[i] = cosine_similarities[i].argsort()[-top_rows - 1:-1][::-1]
#         mean_top_indices[i] = np.mean(cosine_similarities[i][top_indices[i]])
#
#     return top_indices

def get_top_similarity_diagnosis(matrix, top_embeddings,device,num_top=10):
    top_indices = {}
    for i, emb in enumerate(top_embeddings):
        valid_indices = [j for j in range(len(matrix)) if j != i]
        # scores = [(j, cosine_similarity(matrix[i].detach().cpu().reshape(1, -1),
        #                                  matrix[j].detach().cpu().reshape(1, -1))[0][0]) for j in valid_indices]
        # scores = [(j, jaccard_similarity_score(matrix[i].detach().reshape(1, -1),
        #                                        matrix[j].detach().reshape(1, -1), len(matrix), device=device))
        #           for j in valid_indices]
        scores = [(j, cosine_similarity(matrix[i].detach().cpu().reshape(1, -1),
                                        matrix[j].detach().cpu().reshape(1, -1), len(matrix))) for j in valid_indices]
        top_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        top_indices[i] = [j for j, score in top_scores if score > 0][:num_top]
    return top_indices


def create_vocab_matching(top_indices, vocab_df):
    # Create an empty dataframe to store the matching percentages
    vocab_matching = pd.DataFrame(columns=['Subcategory designation',
                                           'Category designation',
                                           'Block',
                                           'Chapter'])
    for diagnosis in top_indices.keys():
        # Get the related indices from vocab_df
        values = top_indices[diagnosis]
        related_df = vocab_df.loc[vocab_df['Diagnosis'].isin(values)]

        # print("related_df:")
        # print(related_df)
        # print("vocab df chapter for this one:")
        # print(diagnosis)
        # print(vocab_df.loc[vocab_df['Diagnosis']==diagnosis])
        # Calculate the percentage of matching values in each column
        subcategory_count = (related_df['Subcategory designation'].isin(vocab_df['Subcategory designation']
                                                                        .loc[vocab_df['Diagnosis'] == diagnosis])).mean()
        category_count = (related_df['Category designation'].isin(vocab_df['Category designation']
                                                                  .loc[vocab_df['Diagnosis'] == diagnosis])).mean()
        block_count = (related_df['Block'].isin(vocab_df['Block']
                                                .loc[vocab_df['Diagnosis'] == diagnosis])).mean()
        chapter_count = (related_df['Chapter'].isin(vocab_df['Chapter'].loc[vocab_df['Diagnosis'] == diagnosis])).mean()

        # Add the percentages as a new row to vocab_matching
        row = [subcategory_count, category_count, block_count, chapter_count]
        vocab_matching.loc[diagnosis] = row
        # print("vocab matching:")
        # print(vocab_matching[:5])

    return vocab_matching

    # for diagnosis in top_indices.keys():
    #     # Get the related indices from vocab_df
    #     values = top_indices[diagnosis]
    #     related_df = vocab_df.loc[vocab_df['Diagnosis'].apply(lambda x: any([v in x for v in values]))]
    #     # Calculate the percentage of matching values in each column
    #     # print(diagnosis)
    #     # print(vocab_df[vocab_df['Diagnosis']
    #     #              .apply(lambda x: diagnosis in x)].iloc[0]['Subcategory designation'])
    #     # print(related_df[related_df['Subcategory designation'] == vocab_df[vocab_df['Diagnosis']
    #     #              .apply(lambda x: diagnosis in x)].iloc[0]['Subcategory designation']])
    #
    #
    #     subcategory_count = (related_df['Subcategory designation'] == vocab_df[vocab_df['Diagnosis']
    #                          .apply(lambda x: diagnosis in x)].iloc[0]['Subcategory designation']).mean()
    #     category_count = (related_df['Category designation'] == vocab_df[vocab_df['Diagnosis']
    #                       .apply(lambda x: diagnosis in x)].iloc[0]['Category designation']).mean()
    #     block_count = (related_df['Block'] == vocab_df[vocab_df['Diagnosis']
    #                       .apply(lambda x: diagnosis in x)].iloc[0]['Block']).mean()
    #     chapter_count = (related_df['Chapter'] == vocab_df[vocab_df['Diagnosis']
    #                       .apply(lambda x: diagnosis in x)].iloc[0]['Chapter']).mean()
    #
    #     # Add the percentages as a new row to vocab_matching
    #     row = [subcategory_count, category_count, block_count, chapter_count]
    #     vocab_matching.loc[diagnosis] = row
    #
    # return vocab_matching

def plot_histograms(df, columns, bins ,k=10):
    # remove rows that have 0 in all columns
    num_columns = len(columns)
    fig, axes = plt.subplots(nrows=1, ncols=num_columns, figsize=(num_columns * 6, 5))
    plt.subplots_adjust(wspace=0.3)
    # print(df[:3])
    mean_score_for_chapter = np.mean(df['Chapter'])
    print(mean_score_for_chapter)
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

    plt.savefig(f'histogram_{column}_{k}_min_similarity_Glove_large_data.png')
    plt.show()
    return mean_score_for_chapter

def get_top_50_words(corpus):
    word_freq = {}
    corpus = [d for d in corpus if d != '<EOS>']
    for word in corpus:
        if word not in word_freq:
            word_freq[word] = 0
        word_freq[word] += 1
    top_words = sorted(word_freq, key=word_freq.get, reverse=True)[:50]
    return top_words

def main():
    # Load the new GloVe embeddings
    with open("repo_glove_visits_large_data.pkl", "rb") as f:
        newglove = pickle.load(f)

    # Load the tokenized corpus which is the data himself
    with open("repo_corpus_large_data.pkl", "rb") as f:
        corpus = pickle.load(f)
    vocab_df = pd.read_csv('encoded_vocab_with_hierarchy_large_data.csv')


    # Get the embedding for each word in the vocabulary
    vocab_embeddings = {}
    for word in newglove:
        vocab_embeddings[word] = newglove[word]
    # print("embeddings:", vocab_embeddings['CYST AND PSEUDOCYST OF PANCREAS'])

    # Get the top 50 words from the corpus
    top_words = get_top_50_words(corpus)
    # creating the dict where keys are the top frequency words(diagnoses) and the values are their embeddings
    word_vectors = [vocab_embeddings[word] for word in top_words]
    # calculating the cosine_similarity of the top_words with all other words in the data
    cos_sim_matrix = cosine_similarity(word_vectors, list(vocab_embeddings.values()))
    print(cos_sim_matrix.shape)
    # Create a dictionary to hold the top similar codes for each diagnosis
    similar_codes_dict = {word: [] for word in top_words}
    # Populate the dictionary with the top similar codes
    num_rows = [1,3,5,10]
    for k in num_rows:
        for i, word in enumerate(top_words):
            print(i,word)
            sim_scores = cos_sim_matrix[i]
            top_sim_indices = sim_scores.argsort()[-k-1:-1]
            top_sim_words = [list(vocab_embeddings.keys())[j] for j in top_sim_indices]
            for w in top_sim_words:
                similar_codes_dict[word].append(w)

        hierarchy_intersections = create_vocab_matching(similar_codes_dict, vocab_df)
        # print(hierarchy_intersections)
        mean_score_for_chapter = plot_histograms(hierarchy_intersections, hierarchy_intersections.columns, 10, k)

        # # Save the dictionary as an Excel file
        writer = pd.ExcelWriter('similar_codes_Glove_large_data.xlsx')
        for word, sim_codes in similar_codes_dict.items():
            sheet_name = word.replace('/', '_').replace('\\', '_').replace('?', '_').replace('*', '_') \
                .replace('[', '_').replace(']', '_').replace(':', '_')
            df = pd.DataFrame({'Similar Codes': sim_codes})
            # Insert the word as the first row of the dataframe
            df = pd.concat([pd.Series([word]), df['Similar Codes']], axis=0, ignore_index=True)
            df.to_excel(writer, sheet_name=sheet_name, index=False,
                        header=False)
            # set header to False to avoid writing column name
        writer.save()




if __name__ == '__main__':
    main()