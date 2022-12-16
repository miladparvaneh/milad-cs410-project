import numpy as np
from pathlib import Path
from collections import defaultdict


def normalize(input_matrix):
    """
    Normalizing the 2d input_matrix along the rows (summing to 1)
    """

    row_sums = input_matrix.sum(axis=1)
    try:
        assert (np.count_nonzero(row_sums) == np.shape(row_sums)[0])
    except Exception:
        raise Exception("Error while normalizing. Row(s) sum to zero")
    new_matrix = input_matrix / row_sums[:, np.newaxis]
    return new_matrix


class Corpus(object):
    """
    The corpus of documents
    """

    def __init__(self, documents_path):
        """
        Initializing parameters
        """
        self.documents = []
        self.vocabulary = []
        self.likelihoods = []
        self.documents_path = documents_path
        self.term_doc_matrix = None
        self.document_topic_prob = None  # P(z | d)
        self.topic_word_prob = None  # P(w | z)
        self.topic_prob = None  # P(z | d, w)

        self.number_of_documents = 0
        self.vocabulary_size = 0

        # Employing list of stop words instead of background language model
        self.stop_words = []
        f = open('stopwords.txt', "r")
        for w in f:
            self.stop_words.append(w.rstrip('\n'))
        f.close()

    def build_corpus(self):
        """
        Reading documents, fill in 'self.documents' as a list of list-of-word
        Example: self.documents = [["the", "day", "is", "nice", "the", ...], [], []...]
        Updating 'self.number_of_documents'
        """
        doc_counter = -1
        for child in sorted(Path(self.documents_path).iterdir()):
            if child.is_file() and child.suffix == '.txt':
                doc_text = child.read_text()
                doc_counter += 1
                self.documents.append([])
                for w in doc_text.split():
                    w = w.rstrip('.').rstrip(',').lower()
                    if w != '[sound]' and w != '[music]' and w != '>>' and w != '[inaudible]':
                        if w not in self.stop_words:
                            self.documents[doc_counter].append(w)

        self.number_of_documents = len(self.documents)

    def build_vocabulary(self):
        """
        Forming a list of unique words in the whole corpus, and putting it in 'self.vocabulary'
        Example: ["rain", "the", ...]
        Updating 'self.vocabulary_size'
        """
        for child in sorted(Path(self.documents_path).iterdir()):
            if child.is_file() and child.suffix == '.txt':
                doc_text = child.read_text()
                for w in doc_text.split():
                    w = w.rstrip('.').rstrip(',').lower()
                    if w in self.vocabulary or w == '[sound]' or w == '[music]' or w == '>>' or w == '[inaudible]' or w in self.stop_words:
                        continue
                    else:
                        self.vocabulary.append(w)

        self.vocabulary_size = len(self.vocabulary)

    def build_term_doc_matrix(self):
        """
        Constructing the term-document matrix. Each row represents a document.
        Each column represents a vocabulary term.
        self.term_doc_matrix[i][j] is the count of term j in document i
        """
        self.term_doc_matrix = np.zeros((self.number_of_documents, self.vocabulary_size))
        doc_counter = -1
        for child in sorted(Path(self.documents_path).iterdir()):
            if child.is_file() and child.suffix == '.txt':
                doc_text = child.read_text()
                doc_counter += 1
                for w in doc_text.split():
                    w = w.rstrip('.').rstrip(',').lower()
                    if w != '[sound]' and w != '[music]' and w != '>>' and w != '[inaudible]':
                        if w not in self.stop_words:
                            self.term_doc_matrix[doc_counter][self.vocabulary.index(w)] += 1
        print(self.term_doc_matrix)

    def initialize_randomly(self, number_of_topics):
        """
        Randomly initialize the matrices: document_topic_prob and topic_word_prob
        which hold the probability distributions for P(z | d) and P(w | z): self.document_topic_prob,
        and self.topic_word_prob
        """
        temp1 = np.random.random_sample(size=(self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(temp1)

        temp2 = np.random.random_sample(size=(number_of_topics, self.vocabulary_size))
        self.topic_word_prob = normalize(temp2)

    def initialize_uniformly(self, number_of_topics):
        """
        Initializes the matrices: self.document_topic_prob and self.topic_word_prob with a uniform
        probability distribution.
        """
        self.document_topic_prob = np.ones((self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)

        self.topic_word_prob = np.ones((number_of_topics, len(self.vocabulary)))
        self.topic_word_prob = normalize(self.topic_word_prob)

    def initialize(self, number_of_topics, random=False):
        """
        Call the functions to initialize the matrices document_topic_prob and topic_word_prob
        """
        print("Initializing...")

        if random:
            self.initialize_randomly(number_of_topics)
        else:
            self.initialize_uniformly(number_of_topics)

    def expectation_step(self):
        """
        The E-step updates P(z | w, d)
        """
        print("E step:")
        self.topic_prob = {}
        for i in range(self.number_of_documents):
            temp1 = np.transpose(self.topic_word_prob)
            temp2 = np.transpose(np.multiply(temp1, self.document_topic_prob[i, :]))
            row_sums = temp2.sum(axis=1)
            new_matrix = temp2 / row_sums[:, np.newaxis]
            self.topic_prob[i] = new_matrix

    def maximization_step(self):
        """
        The M-step updates P(w | z)
        """
        print("M step:")

        # update P(w | z)
        for i in range(self.number_of_documents):
            temp = self.term_doc_matrix[i, :]
            temp1 = np.multiply(self.topic_prob[i], temp)
            row_sums = temp1.sum(axis=1)
            all_sum = np.sum(row_sums)
            row_sums = row_sums / all_sum
            self.document_topic_prob[i, :] = row_sums

        # update P(z | d)
        for i in range(self.number_of_documents):
            temp = self.term_doc_matrix[i, :]
            temp = temp[:, np.newaxis]
            temp = np.transpose(temp)
            if i == -1:
                print(np.shape(temp))
            temp2 = np.multiply(self.topic_prob[i], temp)
            self.topic_word_prob = np.add(self.topic_word_prob, temp2)

        self.topic_word_prob = normalize(self.topic_word_prob)
        row_sums = np.sum(self.topic_word_prob, axis=0)
        new_matrix = self.topic_word_prob / row_sums[np.newaxis, :]
        self.topic_word_prob = new_matrix

    def calculate_likelihood(self):
        """
        Calculating the current log-likelihood of the model using
        the model's updated probability matrices
        Append the calculated log-likelihood to 'self.likelihoods'
        """
        log_like = 0
        for i in range(self.number_of_documents):
            temp1 = np.transpose(self.topic_word_prob)
            if i == -1:
                print(temp1)
            temp2 = np.transpose(np.multiply(temp1, self.document_topic_prob[i, :]))
            row_sums = temp2.sum(axis=0)
            if i == -1:
                print(row_sums)
            log_term = np.log(row_sums)
            if i == -1:
                print(log_term)
                print(self.term_doc_matrix[i, :])
            log_like += np.dot(log_term, self.term_doc_matrix[i, :])
            self.likelihoods.append(log_like)

        return log_like

    def plsa(self, number_of_topics, max_iter):
        """
        Model topics.
        """
        print("EM iteration begins...")

        # build term-doc matrix
        self.build_term_doc_matrix()

        # Create the counter arrays.

        # P(z | d, w)
        self.topic_prob = np.zeros([self.number_of_documents, number_of_topics, self.vocabulary_size], dtype=np.float)

        # P(z | d) P(w | z)
        self.initialize(number_of_topics, random=True)

        # Run the EM algorithm

        for iteration in range(max_iter):
            print("Iteration #" + str(iteration + 1) + "...")

            self.expectation_step()
            self.maximization_step()
            print(self.calculate_likelihood())

            f = open('topic_word_prob.txt', 'w')
            f.write(repr(self.topic_word_prob))
            f.close()

            self.topic_max_word = []
            topic_max_word_prob = np.argmax(self.topic_word_prob, axis=1)
            print('topic_max_word_prob ' + repr(topic_max_word_prob))
            for i in range(topic_max_word_prob.shape[0]):
                self.topic_max_word.append(self.vocabulary[topic_max_word_prob[i]])

            topic_word_dic = defaultdict(lambda: defaultdict(float))
            if iteration == max_iter - 1:
                for i in range(number_of_topics):
                    for j in range(len(self.vocabulary)):
                        topic_word_dic[i][self.vocabulary[j]] = self.topic_word_prob[i, j]

                    print('Topic ' + repr(i))
                    print(sorted(topic_word_dic[i], key=topic_word_dic[i].get, reverse=True)[0:7])

        seg_transcript = []
        doc_counter = -1
        for doc in self.documents:
            seg_transcript.append([])
            doc_counter += 1
            seg_transcript[doc_counter].append([])
            seg_counter = 0
            topic_max_word_visited = []
            for w in doc:
                if w in self.topic_max_word and w not in topic_max_word_visited:
                    topic_max_word_visited.append(w)
                    seg_transcript[doc_counter][seg_counter].append(w)
                    seg_transcript[doc_counter].append([])
                    seg_counter += 1
                else:
                    seg_transcript[doc_counter][seg_counter].append(w)

        f = open('seg_transcript.txt', 'w')
        f.write(repr(seg_transcript))
        f.close()


def main():
    documents_path = 'transcripts/lecture_01/'
    corpus = Corpus(documents_path)  # instantiate corpus
    corpus.build_corpus()
    corpus.build_vocabulary()
    print(corpus.vocabulary)
    print("Vocabulary size:" + str(len(corpus.vocabulary)))
    print("Number of documents:" + str(len(corpus.documents)))
    number_of_topics = 10
    max_iterations = 2000
    corpus.plsa(number_of_topics, max_iterations)


if __name__ == '__main__':
    main()
