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
# psents = json.load(open("parsed_sents_list.json", "r"))
psents = [['A', ['B', ['C', 'blue']], ['B', 'cat']]] # test case

# print a few parsed sentences
# NOTE: you can remove this if you like
# for sent in psents[:10]:
#     print(sent)

# TODO: estimate the conditional probabilities of the rules in the grammar
class Rules:
    def __init__(self):
        self.rules = defaultdict([])

    def add_rule(self, lhs, rhs):
        if isinstance(rhs, list):
            rhs = tuple(rhs)
        assert isinstance(lhs, str) and isinstance(rhs, tuple)
        self.rules[lhs].append(rhs)

    def transform(self):
        writer_obj = RuleWriter()
        for lhs, rhs_lst in self.rules.items():
            counter = Counter(rhs_lst)
            for possible_rhs, freq in counter.items():
                writer_obj.add_rule(lhs, possible_rhs, freq / len(rhs_lst))
        return writer_obj


def parsing(sent):
    # function for parsing a single sentence
    pass

rules = Rules()
rules.add_rule("A", ["B", "B"])
rules.add_rule("B", ["C"])
rules.add_rule("B", ["cat"])
rules.add_rule("C", ["blue"])
    

# TODO: write the rules to the correct output file using the write_rules method
rules.transform().write_rules()
