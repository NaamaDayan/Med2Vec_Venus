import random
import pickle

def generate_sentences(num_sen,max_sen_length,min_sen_length,vocab_size,min_gap_length,max_gap_length):
    # generate the sequences
    sentences = []
    i = 0
    while i < num_sen:
        seq_length = random.randint(min_sen_length, max_sen_length)
        seq = [random.randint(0, vocab_size-1) for _ in range(seq_length)]

        sentences.append(seq)
        i += 1
        if i % random.randint(min_gap_length, max_gap_length) == 0 and i != num_sen:
            sentences.append([-1])
            i += 1

    return sentences



if __name__ == '__main__':
    sentences =  generate_sentences(num_sen=50,max_sen_length = 5,min_sen_length = 2,vocab_size = 10,min_gap_length = 2,max_gap_length=8)
    filename = './data/generated/med2vec.seqs'
    with open(filename, 'wb') as f:
        pickle.dump(sentences, f)


