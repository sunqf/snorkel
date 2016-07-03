from ddlite import *
'''
This class is exactly the same as Relations. The only 
different is that it does not return pairs in the crossproduct
that have the same indices or contain exactly the same tokens.
'''
class HypothesesCandidates(Candidates):
  def __init__(self, content, matcher1=None, matcher2=None):
    if matcher1 is not None and matcher2 is not None:
      if not issubclass(matcher1.__class__, CandidateExtractor):
        warnings.warn("matcher1 is not a CandidateExtractor subclass")
      if not issubclass(matcher2.__class__, CandidateExtractor):
        warnings.warn("matcher2 is not a CandidateExtractor subclass")
      self.e1 = matcher1
      self.e2 = matcher2
    super(HypothesesCandidates, self).__init__(content)
  
  def __getitem__(self, i):
    return Relation(self, i)  
  
  def _apply(self, sent):
    xt = corenlp_to_xmltree(sent)
    for e1_idxs, e1_label in self.e1.apply(sent):
      for e2_idxs, e2_label in self.e2.apply(sent):
	min_e1 = min(e1_idxs)
	max_e1 = max(e1_idxs)
	min_e2 = min(e2_idxs)
	max_e2 = max(e2_idxs)
	if sent.words[min_e1:max_e1] != sent.words[min_e2:max_e2] and max_e1 < min_e2: #### THIS IS THE ONLY DIFFERENCE FROM RELATIONS
	        yield relation_internal(e1_idxs, e2_idxs, e1_label, e2_label, sent, xt)
  
  def _get_features(self, method='treedlib'):
    get_feats = compile_relation_feature_generator()
    f_index = defaultdict(list)
    for j,cand in enumerate(self._candidates):
      for feat in get_feats(cand.root, cand.e1_idxs, cand.e2_idxs):
        f_index[feat].append(j)
    return f_index
    
  def generate_mindtagger_items(self, samp, probs):
    for i, p in zip(samp, probs):
      item = self[i]      
      yield dict(
        ext_id          = item.id,
        doc_id          = item.doc_id,
        sent_id         = item.sent_id,
        words           = json.dumps(corenlp_cleaner(item.words)),
        e1_idxs         = json.dumps(item.e1_idxs),
        e1_label        = item.e1_label,
        e2_idxs         = json.dumps(item.e2_idxs),
        e2_label        = item.e2_label,
        probability     = p
      )
      
  def mindtagger_format(self):
    s1 = """
         <mindtagger-highlight-words index-array="item.e1_idxs" array-format="json" with-style="background-color: yellow;"/>
         <mindtagger-highlight-words index-array="item.e2_idxs" array-format="json" with-style="background-color: cyan;"/>
         """
    s2 = """
         <strong>{{item.e1_label}} -- {{item.e2_label}}</strong>
         """
    return {'style_block' : s1, 'title_block' : s2}
