import csv
import numpy as np
import argparse

VECTOR_LEN = 300   # Length of glove vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and glove_embeddings.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################


def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the glove
    embeddings.

    Parameters:
        file (str): File path to the glove embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding glove
        embedding np.ndarray.
    """
    #same length array for each values
    glove_map = dict()
    with open(file, encoding='utf-8') as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            glove_map[word] = np.array(embedding, dtype=float)
    return glove_map


if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("validation_input", type=str, help='path to validation input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the input .tsv file')
    parser.add_argument("feature_dictionary_in", type=str, 
                        help='path to the GloVe feature dictionary .txt file')
    parser.add_argument("train_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the training data should be written')
    parser.add_argument("validation_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the validation data should be written')
    parser.add_argument("test_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the test data should be written')
    args = parser.parse_args()


def feature_engineer_main(raw_data,dictionary_raw,output_file):
    feature_dict=load_feature_dictionary(dictionary_raw)
    data_arr=load_tsv_dataset(raw_data)
    label_lst=extract_label(data_arr) #extract label values and put it to lst form
    sentence_lst=parse_words(data_arr) #parse each sentence component to words to compare
    word_vector_lst=[] #lst of word vector arrays
    for i in range(len(data_arr)):
        temp=vectorize(sentence_lst[i],feature_dict) #return array for that word
        word_vector_lst.append(temp)
    


    with open(output_file,'w') as f_out:
        for i in range(len(word_vector_lst)):
            label_val=label_lst[i]
            f_out.write(label_val + '\t' )
            for j in range(len(word_vector_lst[0])):
                vec_value=word_vector_lst[i][j]
                f_out.write(f'{vec_value:.6f}' + '\t')
            f_out.write('\n')

        
    return word_vector_lst

def extract_label(data_arr):
    result=[]
    for i in range(len(data_arr)):
        str_label = str(round(float(data_arr[i][0]), 6))
        formatted_label= str_label + "000000"  
        formatted_label = formatted_label[:formatted_label.index('.') + 7] 
        result.append(formatted_label)
    return result

def parse_words(data_arr):
    result=[]
    for i in range(len(data_arr)):
        temp=data_arr[i][1].split(' ')
        result.append(temp)
    return result

def vectorize(sentence,feature_dict):
    #print(sentence)
    count=0
    final_vector=[0]*feature_dict['tomato']
    for word in sentence:
        possible_vec=feature_dict.get(word,None)
        if isinstance(possible_vec,np.ndarray):
            final_vector+=possible_vec
            count+=1
    if count!=0:
        final_vector=final_vector/count 
    return np.round(final_vector, decimals=6)




feature_engineer_main(args.train_input,args.feature_dictionary_in,args.train_out)
feature_engineer_main(args.test_input,args.feature_dictionary_in,args.test_out)
feature_engineer_main(args.validation_input,args.feature_dictionary_in,args.validation_out)