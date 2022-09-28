import json
from collections import Counter, defaultdict


class RuleWriter(object):
    """
    This class is for writing rules in a format 
    the judging software can read
    Usage might look like this:

    rule_writer = RuleWriter()
    for lhs, rhs, prob in out_rules:
        rule_writer.add_rule(lhs, rhs, prob)
    rule_writer.write_rules()

    """
    def __init__(self):
        self.rules = []

    def add_rule(self, lhs, rhs, prob):
        """Add a rule to the list of rules
        Does some checking to make sure you are using the correct format.

        Args:
            lhs (str): The left hand side of the rule as a string
            rhs (Iterable(str)): The right hand side of the rule. 
                Accepts an iterable (such as a list or tuple) of strings.
            prob (float): The conditional probability of the rule.
        """
        assert isinstance(lhs, str)
        assert isinstance(rhs, list) or isinstance(rhs, tuple)
        assert not isinstance(rhs, str)
        nrhs = []
        for cl in rhs:
            assert isinstance(cl, str)
            nrhs.append(cl)
        assert isinstance(prob, float)

        self.rules.append((lhs, nrhs, prob))

    def write_rules(self, filename="q1.json"):
        """Write the rules to an output file.

        Args:
            filename (str, optional): Where to output the rules. Defaults to "q1.json".
        """
        json.dump(self.rules, open(filename, "w"))


# load the parsed sentences
psents = json.load(open("parsed_sents_list.json", "r"))
# psents = [['A', ['B', ['C', 'blue']], ['B', 'cat']]] # test case

# print a few parsed sentences
# NOTE: you can remove this if you like
# for sent in psents[:10]:
#     print(sent)


# TODO: estimate the conditional probabilities of the rules in the grammar
def parse_sent(sent, rules=defaultdict(list)):
    # Parser for each sentence
    # DFS approach to visit all nodes
    curr_node = sent[0]
    children = sent[1:]

    # lexicon, RHS is a single word
    if len(children) == 1 and type(children[0]) == str:
        rules[curr_node].append(tuple(children))

    # CFG rule
    else:
        # for each children, append current CFG rule,
        rules[curr_node].append(tuple([child[0] for child in children]))
        for child in children:
            # for each children, parse the next rule
            parse_sent(child, rules=rules)
    return rules


def parse_psents(psents):
    # Iteratively parse each sentence, put results in the rules dictionary
    rules = defaultdict(list)
    for sent in psents:
        parse_sent(sent=sent, rules=rules)
    return rules


def rules_to_writer(rules):
    writer_obj = RuleWriter()
    # add LHS to RHS mapping
    for lhs, rhs_lst in rules.items():
        counter = Counter(rhs_lst)
        for possible_rhs, freq in counter.items():
            writer_obj.add_rule(lhs, possible_rhs, freq / len(rhs_lst))
    return writer_obj

# the RuleWriter object for formated writing
writer_obj = rules_to_writer(parse_psents(psents))
    

# TODO: write the rules to the correct output file using the write_rules method
writer_obj.write_rules()
