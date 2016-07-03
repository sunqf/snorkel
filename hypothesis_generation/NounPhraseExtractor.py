from ddlite_matcher import *

class NounPhraseExtractor(CandidateExtractor):
    """Candidates correspond to all noun phrases as extracted via TextBlob"""
    def init(self):
        self.label = self.opts['label']
        self.ignore_case = self.opts.get('ignore_case', True)
	self.nlp = self.opts['nlp_method']

    def match_tokens(self, list1, list2, noun_phrases, text_lower_tokens,sentence_pointer,np_len, text):
	if len(list1) != len(list2):
		print list1, list2
		print noun_phrases, text_lower_tokens
		print sentence_pointer, np_len
		print text
		assert False
	flag = True
	for i in range(len(list1)):
		if not list1[i].startswith(list2[i]):
			flag = False
			break;
	return flag

    def valid_noun_phrase(self, np, np_pos, pre_token, post_token):
	# check if candidate is followed by = sign
	char_filter = ["=",":"]
	for c in char_filter:
		if c in post_token:
			return False
	# if none of the pos tags for np contains a noun as tagged by corenlp then not valid
	NN_Pos_found = False
	for pos in np_pos:
		if pos.startswith('NN'):
			NN_Pos_found = True
			break;
	if not NN_Pos_found:
		return False
	# if some filters are off
	filters_in = ["<",">","=",'%']
	filters_in.extend(["significant", "statistical", "different", "percentage","p-value","correlation","figure", "group", "effect", "level", "rate", "number", "test", "regression", "control", "group","study","deviation"])
	filters_exact = ["ci","or","i2","hr","wt","tau"]
	for f in filters_in:
		if f in np:
			return False
	for f in filters_exact:
		if f == np:
			return False
	return True

    def _apply(self, s, idxs=None):
        """
        Analyze a sentence with TextBlob to extract all noun phrases.
        """
        # grab word tokens (match_attrib set to words)
        words = s.words
        # conver word to text
        text = s.text.lower()
        text = text.replace('-LRB-','(').replace('-RRB-',')')
	text_lower_tokens = [w.lower() for w in s.words]
        assert len(text_lower_tokens) == len(words)
        # extract noun phrases from text
        blob = self.nlp(text)
        noun_phrases = []
	for np in blob.noun_chunks:
		noun_phrases.append(np.text)
	
	# match noun phrases to raw sentnece tokens - use two pointers to iterate over noun phrases and raw text tokens
	sentence_pointer= 0
	noun_phrases_pointer = 0
	while noun_phrases_pointer < len(noun_phrases):
	    np = noun_phrases[noun_phrases_pointer]
	    np_tokens = np.split(' ')
	    np_len = len(np_tokens)
	    raw_tokens = text_lower_tokens[sentence_pointer:sentence_pointer+np_len]
	    if not self.match_tokens(raw_tokens, np_tokens, noun_phrases, text_lower_tokens, sentence_pointer,np_len, s.text):
		# No match found progress raw sentence pointer
	       	sentence_pointer += 1
	    else:
		# Match found set idxs and progress both pointers
		# examine if match is a valid noun phrase
		if self.valid_noun_phrase(np, s.poses[sentence_pointer:sentence_pointer+np_len], s.words[max(sentence_pointer-1,0)], s.words[min(len(s.words)-1,sentence_pointer+np_len)]):
			idxs = range(sentence_pointer, sentence_pointer+np_len, 1)
			noun_phrases_pointer += 1
			sentence_pointer += np_len
			yield idxs, self.label
		else:
			noun_phrases_pointer += 1
			sentence_pointer += np_len

