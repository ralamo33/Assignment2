import inspect, sys, hashlib

# Hack around a warning message deep inside scikit learn, loaded by nltk :-(
#  Modelled on https://stackoverflow.com/a/25067818
import warnings
with warnings.catch_warnings(record=True) as w:
    save_filters=warnings.filters
    warnings.resetwarnings()
    warnings.simplefilter('ignore')
    import nltk
    warnings.filters=save_filters
try:
    nltk
except NameError:
    # didn't load, produce the warning
    import nltk

from nltk.corpus import brown
from nltk.tag import map_tag, tagset_mapping
import math

if map_tag('brown', 'universal', 'NR-TL') != 'NOUN':
    # Out-of-date tagset, we add a few that we need
    tm=tagset_mapping('en-brown','universal')
    tm['NR-TL']=tm['NR-TL-HL']='NOUN'

class HMM:
    def __init__(self, train_data, test_data):
        """
        Initialise a new instance of the HMM.

        :param train_data: The training dataset, a list of sentences with tags
        :type train_data: list(list(tuple(str,str)))
        :param test_data: the test/evaluation dataset, a list of sentence with tags
        :type test_data: list(list(tuple(str,str)))
        """
        self.train_data = train_data
        self.test_data = test_data

        # Emission and transition probability distributions
        self.emission_PD = None
        self.transition_PD = None
        self.states = []

        self.viterbi = []
        self.backpointer = []

    # Compute emission model using ConditionalProbDist with a LidstoneProbDist estimator.
    #   To achieve the latter, pass a function
    #    as the probdist_factory argument to ConditionalProbDist.
    #   This function should take 3 arguments
    #    and return a LidstoneProbDist initialised with +0.01 as gamma and an extra bin.
    #   See the documentation/help for ConditionalProbDist to see what arguments the
    #    probdist_factory function is called with.
    def emission_model(self, train_data):
        """
        Compute an emission model using a ConditionalProbDist.

        :param train_data: The training dataset, a list of sentences with tags
        :type train_data: list(list(tuple(str,str)))
        :return: The emission probability distribution and a list of the states
        :rtype: Tuple[ConditionalProbDist, list(str)]
        """
        # Don't forget to lowercase the observation otherwise it mismatches the test data
        # Do NOT add <s> or </s> to the input sentences
        data = [[(tag, word.lower()) for word, tag in sentence] for sentence in train_data]
        flat_data = []
        for d in data:
            for item in d:
                flat_data.append(item)
        emission_FD = nltk.ConditionalFreqDist(flat_data)
        #Should I be using emission_FD for the FreqDistributions???/
        self.emission_PD = nltk.ConditionalProbDist(emission_FD, self.LidstoneProbDistFactory)
        self.states = list(self.emission_PD.keys())
        return self.emission_PD, self.states

    def LidstoneProbDistFactory(self, freqdist, gamma=0.01, extra_bins=1):
        return nltk.LidstoneProbDist(freqdist, gamma, freqdist.B() + extra_bins)


    # Access function for testing the emission model
    # For example model.elprob('VERB','is') might be -1.4
    def elprob(self,state,word):
        """
        The log of the estimated probability of emitting a word from a state

        :param state: the state name
        :type state: str
        :param word: the word
        :type word: str
        :return: log base 2 of the estimated emission probability
        :rtype: float
        """
        probability = self.emission_PD[state].prob(word)
        ans = math.log(probability, 2)
        return ans

    # Compute transition model using ConditionalProbDist with a LidstonelprobDist estimator.
    # See comments for emission_model above for details on the estimator.
    def transition_model(self, train_data):
        """
        Compute an transition model using a ConditionalProbDist.

        :param train_data: The training dataset, a list of sentences with tags
        :type train_data: list(list(tuple(str,str)))
        :return: The transition probability distribution
        :rtype: ConditionalProbDist
        """
#        raise NotImplementedError('HMM.transition_model')
        # TODO: prepare the data
        data = []

        # The data object should be an array of tuples of conditions and observations,
        # in our case the tuples will be of the form (tag_(i),tag_(i+1)).
        # DON'T FORGET TO ADD THE START SYMBOL </s> and the END SYMBOL </s>
        for s in train_data:
            for i, word_tag in enumerate(s):
                tag = word_tag[1]
                if i == 0:
                    data.append(("<s>", tag))
                if i != len(s) - 1:
                    data.append((tag, s[i + 1][1]))
                else:
                    data.append((tag, "</s>"))
        # TODO compute the transition model
        transition_FD = nltk.ConditionalFreqDist(data)
        self.transition_PD = nltk.ConditionalProbDist(transition_FD, self.LidstoneProbDistFactory)

        return self.transition_PD

    # Access function for testing the transition model
    # For example model.tlprob('VERB','VERB') might be -2.4
    def tlprob(self,state1,state2):
        """
        The log of the estimated probability of a transition from one state to another

        :param state1: the first state name
        :type state1: str
        :param state2: the second state name
        :type state2: str
        :return: log base 2 of the estimated transition probability
        :rtype: float
        """
        prob = self.transition_PD[state1].prob(state2)
        return math.log(prob, 2)

    # Train the HMM
    def train(self):
        """
        Trains the HMM from the training data
        """
        self.emission_model(self.train_data)
        self.transition_model(self.train_data)

    # Part B: Implementing the Viterbi algorithm.

    # Initialise data structures for tagging a new sentence.
    # Describe the data structures with comments.
    # Use the models stored in the variables: self.emission_PD and self.transition_PD
    # Input: first word in the sentence to tag
    def initialise(self, observation):
        """
        Initialise data structures for tagging a new sentence.

        :param observation: the first word in the sentence to tag
        :type observation: str
        Data Structures
        :param viterbi: Dictionary of tags to a list of cost corresponding to the steps of tagging.
        So viterbi["NOUN"][4] would be the cost of a NOUN at step 4.
        :type viterbi: Dictionary(String -> list(float))
        :param backtrace: Dictionary of tags to a list of tags corresponding to the steps of tagging.
        So backtrace["VERB"][3] would be the backtrace of a VERB at step 3.
        :type backtrace: Dictionary(String -> list(String))
        """
        start_state = "<s>"
        self.viterbi = dict()
        self.backpointer = dict()
        for state in self.states:
            #tlprob and elprob return positive logarithms of the probability, they must be negated to become costs.
            self.viterbi.update({state : [-self.tlprob(start_state, state) - self.elprob(state, observation)]})
            self.backpointer.update({state : [start_state]})


    # Tag a new sentence using the trained model and already initialised data structures.
    # Use the models stored in the variables: self.emission_PD and self.transition_PD.
    # Update the self.viterbi and self.backpointer datastructures.
    # Describe your implementation with comments.
    def tag(self, observations):
        """
        Tag a new sentence using the trained model and already initialised data structures.

        :param observations: List of words (a sentence) to be tagged
        :type observations: list(str)
        :return: List of tags corresponding to each word of the input
        """
#        raise NotImplementedError('HMM.tag')
        tags = []
        if self.backpointer == None or self.viterbi == None:
            raise AttributeError("Either backpointer or viterbi have not been intialized yet, remember to run initalise"
                                 "before tag.")
        #ToDO: Verb at step 5 should cost 56
        #The step that self.viterbi and self.backpointer are currently on
        step = 1
        for t in observations:
            for s in self.states:
                cost_given_word = -self.elprob(s, t)
                last_state_cost = 0
                last_state = None
                #Choose the last state with the least cost
                for context in self.states:
                    #self.tlprob returns the log of the probability, so it must be negated to be the cost
                    cost_of_transition = -self.tlprob(context, s)
                    cost_of_context = self.get_viterbi_value(context, step) + cost_of_transition
                    if last_state is None:
                        last_state = context
                        last_state_cost = cost_of_context
                    elif cost_of_context < last_state_cost:
                        last_state = context
                        last_state_cost = cost_of_context
                cost_total = cost_given_word + last_state_cost
                #Update viterbi and backpointer with the cost and last_state of the next step.
                self.viterbi[s].append(cost_total)
                self.backpointer[s].append(last_state)
            #We have now gone one further step in the viterbi algorithm
            step += 1

        # TODO
        # Add a termination step with cost based solely on cost of transition to </s> , end of sentence.
        termination_state = "</s>"
        before_termination_state = None
        before_termination_cost = 0
        for s in self.states:
            cost_of_context = self.get_viterbi_value(s, step) - self.tlprob(s, termination_state)
            #The chance that the termination symbol will be used at the sentence end is 100%, meaning cost is 0
            cost_total = cost_of_context
            #Add this cost to the next step in the viterbi
            #self.viterbi[s].append(cost_total)
            if before_termination_state is None or cost_total < before_termination_cost:
                before_termination_state = s
                before_termination_cost = cost_total
        step += 1
        #self.viterbi[termination_state] = before_termination_cost
        self.backpointer[termination_state]  = before_termination_state

        # TODO
        # Reconstruct the tag sequence using the backpointer list.
        # Return the tag sequence corresponding to the best path as a list.
        # The order should match that of the words in the sentence.

        #Start with the backpointer for the termination variable. From there work out the backpointer values to get
        #the tags.
        last_tag = before_termination_state
        tags.append(last_tag)
        for i in range(len(self.backpointer[last_tag]) - 1, 0, -1):
            last_tag = self.get_backpointer_value(last_tag, i)
            tags.append(last_tag)
        #Tags is now in reverse order. Fix that
        tags = tags[::-1]
        return tags



    # Access function for testing the viterbi data structure
    # For example model.get_viterbi_value('VERB',2) might be 6.42

    def get_viterbi_value(self, state, step):
        """
        Return the current value from self.viterbi for
        the state (tag) at a given step

        :param state: A tag name
        :type state: str
        :param step: The (0-origin) number of a step:  if negative,
          counting backwards from the end, i.e. -1 means the last (</s>) step
        :type step: int
        :return: The value (a cost) for state as of step
        :rtype: float
        """
        answer = self.viterbi[state][step - 1]
        return answer

    # Access function for testing the backpointer data structure
    # For example model.get_backpointer_value('VERB',2) might be 'NOUN'
    def get_backpointer_value(self, state, step):
        """
        Return the current backpointer from self.backpointer for
        the state (tag) at a given step

        :param state: A tag name
        :type state: str
        :param step: The (0-origin) number of a step:  if negative,
          counting backwards from the end, i.e. -1 means the last (</s>) step
        :type step: str
        :return: The state name to go back to at step-1
        :rtype: str
        """
        return self.backpointer[state][step - 1]

def answer_question4b():
    """
    Report a hand-chosen tagged sequence that is incorrect, correct it
    and discuss
    :rtype: list(tuple(str,str)), list(tuple(str,str)), str
    :return: your answer [max 280 chars]
    """
    raise NotImplementedError('answer_question4b')

    # One sentence, i.e. a list of word/tag pairs, in two versions
    #  1) As tagged by your HMM
    #  2) With wrong tags corrected by hand
    tagged_sequence = 'fixme'
    correct_sequence = 'fixme'
    # Why do you think the tagger tagged this example incorrectly?
    answer =  inspect.cleandoc("""\
    fill me in""")[0:280]

    return tagged_sequence, correct_sequence, answer

def answer_question5():
    """
    Suppose you have a hand-crafted grammar that has 100% coverage on
        constructions but less than 100% lexical coverage.
        How could you use a POS tagger to ensure that the grammar
        produces a parse for any well-formed sentence,
        even when it doesn't recognise the words within that sentence?

    :rtype: str
    :return: your answer [max 500 chars]
    """
    raise NotImplementedError('answer_question5')

    return inspect.cleandoc("""\
    fill me in""")[0:500]

def answer_question6():
    """
    Why else, besides the speedup already mentioned above, do you think we
    converted the original Brown Corpus tagset to the Universal tagset?
    What do you predict would happen if we hadn't done that?  Why?

    :rtype: str
    :return: your answer [max 500 chars]
    """
    raise NotImplementedError('answer_question6')

    return inspect.cleandoc("""\
    fill me in""")[0:500]

# Useful for testing
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    # http://stackoverflow.com/a/33024979
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def answers():
    global tagged_sentences_universal, test_data_universal, \
           train_data_universal, model, test_size, train_size, ttags, \
           correct, incorrect, accuracy, \
           good_tags, bad_tags, answer4b, answer5

    # Load the Brown corpus with the Universal tag set.
    tagged_sentences_universal = brown.tagged_sents(categories='news', tagset='universal')

    # Divide corpus into train and test data.
    test_size = 500
    train_size = len(tagged_sentences_universal) - 500

    test_data_universal = tagged_sentences_universal[-500:]
    train_data_universal = tagged_sentences_universal[:-500]

    if hashlib.md5(''.join(map(lambda x:x[0],train_data_universal[0]+train_data_universal[-1]+test_data_universal[0]+test_data_universal[-1])).encode('utf-8')).hexdigest()!='164179b8e679e96b2d7ff7d360b75735':
        print('!!!test/train split (%s/%s) incorrect, most of your answers will be wrong hereafter!!!'%(len(train_data_universal),len(test_data_universal)),file=sys.stderr)

    # Create instance of HMM class and initialise the training and test sets.
    model = HMM(train_data_universal, test_data_universal)

    # Train the HMM.
    model.train()

    # Some preliminary sanity checks
    # Use these as a model for other checks
    e_sample=model.elprob('VERB','is')
    if not (type(e_sample)==float and e_sample<=0.0):
        print('elprob value (%s) must be a log probability'%e_sample,file=sys.stderr)

    t_sample=model.tlprob('VERB','VERB')
    if not (type(t_sample)==float and t_sample<=0.0):
           print('tlprob value (%s) must be a log probability'%t_sample,file=sys.stderr)

    if not (type(model.states)==list and \
            len(model.states)>0 and \
            type(model.states[0])==str):
        print('model.states value (%s) must be a non-empty list of strings'%model.states,file=sys.stderr)

    print('states: %s\n'%model.states)

    ######
    # Try the model, and test its accuracy [won't do anything useful
    #  until you've filled in the tag method
    ######
    s='the cat in the hat came back'.split()
    model.initialise(s[0])
    ttags = model.tag(s)
    print("Tagged a trial sentence:\n  %s"%list(zip(s,ttags)))

    v_sample=model.get_viterbi_value('VERB',5)
    if not (type(v_sample)==float and 0.0<=v_sample):
           print('viterbi value (%s) must be a cost'%v_sample,file=sys.stderr)

    b_sample=model.get_backpointer_value('VERB',5)
    if not (type(b_sample)==str and b_sample in model.states):
           print('backpointer value (%s) must be a state name'%b_sample,file=sys.stderr)


    # check the model's accuracy (% correct) using the test set
    correct = 0
    incorrect = 0

    for sentence in test_data_universal:
        s = [word.lower() for (word, tag) in sentence]
        model.initialise(s[0])
        tags = model.tag(s)

        for ((word,gold),tag) in zip(sentence,tags):
            if tag == gold:
                correct += 1
            else:
                incorrect += 1

    accuracy = (correct / (correct + incorrect)) * 100
    print('Tagging accuracy for test set of %s sentences: %.4f'%(test_size,accuracy))

    # Print answers for 4b, 5 and 6
    bad_tags, good_tags, answer4b = answer_question4b()
    print('\nA tagged-by-your-model version of a sentence:')
    print(bad_tags)
    print('The tagged version of this sentence from the corpus:')
    print(good_tags)
    print('\nDiscussion of the difference:')
    print(answer4b[:280])
    answer5=answer_question5()
    print('\nFor Q5:')
    print(answer5[:500])
    answer6=answer_question6()
    print('\nFor Q6:')
    print(answer6[:500])

if __name__ == '__main__':
#    if len(sys.argv)>1 and sys.argv[1] == '--answers':
        import adrive2_embed
        from autodrive_embed import run, carefulBind
        with open("userErrs.txt","w") as errlog:
            run(globals(),answers,adrive2_embed.a2answers,errlog)
#    else:
#        answers()
